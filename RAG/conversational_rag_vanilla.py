from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
import sys
sys.path.append('..')  # add parent directory to path
import utils


current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, 'datasets/lenta-ru-news-short.csv')
index_dir = os.path.join(current_path, 'faiss-index')

# creating documents
loader = CSVLoader(
    file_path=dataset_path,
    csv_args={'fieldnames': ['Заголовок', 'Содержание', 'Тема', 'Теги', 'Дата']}
)

docs = loader.load()

# embeddings model
model_kwargs = {'device': 'mps'} # use GPU
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name='cointegrated/LaBSE-en-ru',
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# creating or loading vector store
if not os.path.exists(index_dir):
    db = FAISS.from_documents(documents=docs, embedding=embeddings)
    db.save_local(folder_path=index_dir, index_name='index')
else:
    db = FAISS.load_local(
        folder_path=index_dir,
        index_name='index',
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

# creating retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 1})

# creating LLM
llm = utils.init_gigachat_model()

# contextualizing question prompt
contextualize_q_system_prompt = (
    'Твоя задача — переформулировать вопрос пользователя с учетом истории сообщений так, '
    'чтобы он стал понятен без контекста предыдущих сообщений. '
    'НЕ ОТВЕЧАЙ на вопрос, а только переформулируй его! '
    'Результат должен содержать только переформулированный вопрос. '
    'Например, если в истории говорили про Трампа, а вопрос "Какой у него рост?", '
    'результат должен быть: "Переформулированный вопрос: Какой рост у Трампа?"'
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}. Переформулированный вопрос: '),
])

# QA-prompt
qa_system_prompt = (
    'Твоя задача — ответить на вопрос пользователя ТОЛЬКО на основе контекста, '
    'приведенного ниже. Ты не должен использовать свои общие знания '
    'или любые другие источники, кроме указанного контекста. '
    'Если в контексте нет информации для ответа, напиши ровно одну фразу: '
    '"Я не знаю ответа, так как в контексте нет нужной информации."'
    '\n\nКонтекст:\n{context}'
)
qa_prompt = ChatPromptTemplate.from_messages([
    ('system', qa_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}'),
])

# retrieving context by rephrased query
def get_context(reprased_query):
    docs = retriever.invoke(reprased_query)
    return '\n'.join([doc.page_content for doc in docs])

# rephrasing query based on previous messages
rephrase_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
)

# QA-chain
qa_chain = (
    RunnablePassthrough.assign(
        input=rephrase_chain,
        context=lambda x: get_context(x['input']),
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# QA-loop
print('Введите "quit" для завершения чата.\n')

chat_history = []

while True:
    query = input('Ваш вопрос: ')

    if query == 'quit':
        break

    answer = qa_chain.invoke(
        {'input': query,
         'chat_history': chat_history}
    )

    print(f'> {answer}\n')

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))