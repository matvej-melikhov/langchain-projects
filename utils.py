from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_ollama.llms import OllamaLLM


def init_gigachat_model() -> GigaChat:
    load_dotenv(override=True)
    model = GigaChat(verify_ssl_certs=False)
    return model


def init_local_ollama_model(model_name: str = 'mistral:latest') -> OllamaLLM:
    model = OllamaLLM(model=model_name)
    return model   