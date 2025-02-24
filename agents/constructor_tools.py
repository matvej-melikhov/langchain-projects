from pydantic import BaseModel, Field
from typing import Union, Any

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

from langchain_core.tools import tool
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent

from prompts import contextualize_q_system_prompt, qa_system_prompt

import sys; sys.path.append('..')
import os
import utils


# pydantic model for the tool arguments

class EvalMathByPythonArgs(BaseModel):
    expression: str = Field('Выражение для вычисления на Python')


class SearchInWikipedia(BaseModel):
    query: str = Field('Запрос для поиске в Wikipedia')


class RagToolInput(BaseModel):
    query: str = Field(description='Запрос для поиска информации в базе новостей Lenta.ru')


# RAG-tool

class RagTool(BaseTool):
    name: str = 'rag_tool'
    description: str = 'Инструмент для поиска релевантной информации в векторной БД новостей Lenta.ru'
    args_schema: type = RagToolInput

    # required for pydantic
    current_path: str = Field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    dataset_path: str = Field(default=None)
    index_dir: str = Field(default=None)
    docs: list[Any] = Field(default_factory=list)
    embeddings: HuggingFaceEmbeddings = Field(default=None)
    db: FAISS = Field(default=None)
    retriever: Any = Field(default=None)
    llm: Any = Field(default=None)
    contextualize_prompt: ChatPromptTemplate = Field(default=None)
    qa_prompt: ChatPromptTemplate = Field(default=None)
    rag_chain: Any = Field(default=None)

    def __init__(self):
        super().__init__()

        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.current_path, '../RAG/datasets/lenta-ru-news-short.csv')
        self.index_dir = os.path.join(self.current_path, '../RAG/faiss-index')

        self.docs = self._load_docs()
        self.embeddings = self._load_embeddings_model()
        self.db = self._load_db()

        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={'k': 1})
        self.llm = utils.init_gigachat_model()

        self.contextualize_prompt = self._get_contextualize_prompt()
        self.qa_prompt = self._get_qa_prompt()

        self.rag_chain = self._create_rag_chain()
        
    def _load_docs(self) -> list:
        loader = CSVLoader(
            file_path=self.dataset_path,
            csv_args={'fieldnames': ['Заголовок', 'Содержание', 'Тема', 'Теги', 'Дата']}
        )
        docs = loader.load()
        return docs
    
    def _load_embeddings_model(self) -> HuggingFaceEmbeddings:
        embeddings_model = HuggingFaceEmbeddings(
            model_name='cointegrated/LaBSE-en-ru',
            model_kwargs={'device': 'mps'},
            encode_kwargs={'normalize_embeddings': True},
        )
        return embeddings_model
    
    def _load_db(self) -> FAISS:
        db = FAISS.load_local(
            folder_path=self.index_dir,
            index_name='index',
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return db
    
    def _get_contextualize_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ('system', contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ])
    
    def _get_qa_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ('system', qa_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ])
    
    def _create_rag_chain(self) -> Any:
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=self.contextualize_prompt
        )
        qa_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.qa_prompt
        )
        return create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=qa_chain)
    
    def _run(self, query: str, chat_history: list = None) -> str:
        """Выполняет поиск и генерацию ответа на основе запроса и истории чата"""
        chat_history = chat_history or []  # empty list if chat_history is None
        result = self.rag_chain.invoke({'input': query, 'chat_history': chat_history})
        return result['answer']


# functions for the tools

@tool
def get_date_and_time() -> str:
    """Получает текущие дату и время"""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%d-%m-%Y %H:%M:%S")


@tool(args_schema=SearchInWikipedia)
def search_in_wikipedia(query: str, *args, **kwargs) -> str:
    """Выполняет поиск в Wikipedia по запросу"""
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return 'Нет информации по запросу'


@tool(args_schema=EvalMathByPythonArgs)
def eval_math_by_python(expression: str, *args, **kwargs) -> Union[int, float]:
    """Вычисляет математическое выражение с помощью Python"""
    import math

    try:
        return eval(expression)
    except Exception as e:
        return f'Невалидное выражение: {e}'


tools = [
    get_date_and_time,
    search_in_wikipedia,
    eval_math_by_python,
    RagTool(),
]

# better results with GigaChat-Max
llm = utils.init_gigachat_model() # llm = utils.init_local_ollama_model(model_name='phi4:latest')


system_prompt_template = '''\
Ты универсальный ассистент. У тебя есть инструменты: eval_math_by_python, search_in_wikipedia, get_date_and_time, rag_tool. \
Твоя задача — использовать инструмент КАЖДЫЙ РАЗ, когда запрос пользователя относится к его функционалу. \
Если подходящего инструмента нет, только тогда отвечай текстом.

Для каждого запроса выполняй следующие шаги:
1. Определи, какой инструмент нужен для ответа. Если подходит несколько, выбери наиболее релевантный или используй их последовательно.
2. Переформулируй запрос пользователя в формат, который ожидает инструмент:
   - eval_math_by_python: Python-выражение (например, "2 + 3").
   - search_in_wikipedia: ключевое слово или фраза для поиска (например, "Дональд Трамп").
   - get_date_and_time: не требует параметров, просто вызови инструмент.
   - rag_tool: запрос для поиска в базе новостей (например, "Новый год 2016 в Москве").
3. Верни вызов инструмента в формате JSON:
```
{{"tool": "имя_инструмента", "tool_input": {{"параметр": "значение"}}}}
```
Для get_date_and_time используй пустой tool_input: {{"tool": "get_date_and_time", "tool_input": {{}}}}.

Примеры:
- Запрос: "Сколько будет два в седьмой степени умножить на корень из 16?"
→ Инструмент: eval_math_by_python
→ Переформулировка: "2**7 * 16**0.5"
→ JSON: {{"tool": "eval_math_by_python", "tool_input": {{"expression": "2**7 * 16**0.5"}}}}

- Запрос: "Сколько лет Дональду Трампу?"
→ Инструменты: search_in_wikipedia (узнать дату рождения), get_date_and_time (текущая дата)
→ Переформулировка: "Дональд Трамп" для поиска, затем вызов без параметров
→ JSON: {{"tool": "search_in_wikipedia", "tool_input": {{"query": "Дональд Трамп"}}}} (затем отдельно get_date_and_time)

- Запрос: "Как в Москве встретили новый 2016 год?"
→ Инструмент: rag_tool
→ Переформулировка: "Новый год 2016 в Москве"
→ JSON: {{"tool": "rag_tool", "tool_input": {{"query": "Новый год 2016 в Москве"}}}}

- Запрос: "Привет, как дела?"
→ Инструмент не нужен
→ Ответ: "Привет! У меня всё хорошо, а у тебя?"

Используй инструменты максимально часто, даже если кажется, что можно ответить без них. Удачи!
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt_template),
        ('placeholder', '{chat_history}'),
        ('human', '{input}'),
        ('placeholder', '{agent_scratchpad}'),
    ]
)

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
)

# available only for models with tools-calling support, model should have .bind_tools() method
# better to use tools (if it's available) than create_structured_chat_agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory, # auto update chat messages after invoke calling
    handle_parsing_errors=True,
)

while True:
    query = input('Вопрос: ')
    if query.lower() == 'quit':
        break

    response = agent_executor.invoke({'input': query})['output']
    print(f'AI: {response}')