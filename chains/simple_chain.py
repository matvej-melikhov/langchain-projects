from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
import utils


model = utils.init_gigachat_model()

# tuples to use as templates
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage('Отвечай пользователю, основываясь на предоставленных примерах'),
    HumanMessage('привет -> '),
    AIMessage('тевирп'),
    HumanMessage('например -> '),
    AIMessage('ремирпан'),
    HumanMessage('мадам -> '),
    AIMessage('мадам'),
    ('human', '{query} -> '),
])

palindrome_checking = RunnableLambda(lambda x: f'{x} (палиндром)' if x == x[::-1] else x)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | palindrome_checking
)

result1 = chain.invoke({'query': 'школа'})
print(result1)

result2 = chain.invoke({'query': 'летел'})
print(result2)