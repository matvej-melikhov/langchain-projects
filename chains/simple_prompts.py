from dotenv import load_dotenv
from os import environ as env

from langchain_gigachat.chat_models import GigaChat
from langchain.prompts import ChatPromptTemplate


load_dotenv()

model = GigaChat(
    model=env['MODEL'],
    scope=env['SCOPE'],
    credentials=env['CREDENTIALS'],
    verify_ssl_certs=False,
)

# tuples to use as templates
messages = [
    ('system', 'Ты должен отвечать как {role} на все вопросы пользователя'),
    ('human', 'Расскажи шутку про {topic}')
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
    'role': 'программист-зануда',
    'topic': 'мышь'
})

result = model.invoke(prompt).content
print(result)