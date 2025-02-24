from dotenv import load_dotenv
from os import environ as env

from langchain_gigachat.chat_models import GigaChat


load_dotenv()

model = GigaChat(
    model=env['MODEL'],
    scope=env['SCOPE'],
    credentials=env['CREDENTIALS'],
    verify_ssl_certs=False,
)

result = model.invoke('Привет, как тебя зовут?')
print(result.content)