from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import sys; sys.path.append('..')
from chains.simple_prompts import react_prompt_template
import utils


def get_current_time(*args, **kwargs) -> str:
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")


tools = [
    Tool(
        name='Time',
        func=get_current_time,
        description='Полезно, когда нужно узнать текущее время',
    ),
]

llm = utils.init_gigachat_model()

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_template,
    stop_sequence=True,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    # handle_parsing_errors=True
)

query = 'Который сейчас час?'
response = agent_executor.invoke({'input': query})

print(response)