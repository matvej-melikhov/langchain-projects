from dotenv import load_dotenv
from os import environ as env

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()

model = GigaChat(
    model=env['MODEL'],
    scope=env['SCOPE'],
    credentials=env['CREDENTIALS'],
    verify_ssl_certs=False,
)

messages1 = [
    SystemMessage('Ты очень эмоциональный бот, отвечай очень экспрессивно'),
    HumanMessage('Привет, кто ты такой?'),
]

messages2 = [
    SystemMessage('Ты должен выполнять математические операции по правилам, которые задает пользователь'),
    HumanMessage('Сколько будет 2 + 2?'),
    AIMessage('2 + 2 = 22'),
    HumanMessage('Сколько будет 11 + 7?'),
    AIMessage('11 + 7 = 117'),
    HumanMessage('Сколько будет 13 + 19?'),
]

result = model.invoke(messages2)
print(result.content)