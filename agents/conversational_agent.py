from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import sys; sys.path.append('..')
import utils
from prompts import structured_chat_prompt_template


def get_current_datetime(*args, **kwargs) -> str:
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%d-%m-%Y %H:%M:%S")


def search_wikipedia(query: str) -> str:
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return 'Нет информации по запросу'
    

tools = [
    Tool(
        name='Time',
        func=get_current_datetime,
        description='Полезно, когда нужно узнать текущее время или дату.',
    ),
    Tool(
        name='Wikipedia',
        func=search_wikipedia,
        description='Полезно, когда нужно найти информацию в Википедии',
    )
]

human_template_prompt = '''Question: {input}
{agent_scratchpad}
'''

prompt = ChatPromptTemplate.from_messages([
    ('system', structured_chat_prompt_template),
    MessagesPlaceholder('chat_history'),
    ('human', human_template_prompt),
])

llm = utils.init_gigachat_model()
# llm = utils.init_local_ollama_model(model_name='gemma:latest')

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
)

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

while True:
    query = input('Вопрос: ')
    if query.lower() == 'quit':
        break

    response = agent_executor.invoke({'input': query})
    print('AI: ', response['output'])

    memory.chat_memory.add_message(HumanMessage(content=query))
    memory.chat_memory.add_message(AIMessage(content=response['output']))