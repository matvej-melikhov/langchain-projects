from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
import utils


def extract_prompt(pos: str, sentence: str):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage('Ты профессиональный лингвист.'),
        (
            'human',
            '''Выдели все слова части речи {pos} из предложения
            и приведи их в виде списка через запятую.\nПредложение: {sentence}'''
        )
    ]).format_prompt(pos=pos, sentence=sentence)
    return prompt

def agg_result(nouns: str, verbs: str) -> str:
    return f'Nouns: {nouns}\nVerbs: {verbs}'


model = utils.init_gigachat_model()

prompt = ChatPromptTemplate.from_messages([
    SystemMessage('Ты должен придумывать предложения длиной от 10 до 20 слов.'),
    ('human', 'Сочини предложение на тему {topic}'),
])

nouns_branch_chain = (
    RunnableLambda(lambda x: extract_prompt(pos='существительное', sentence=x))
    | model
    | StrOutputParser()
)

verbs_branch_chain = (
    RunnableLambda(lambda x: extract_prompt(pos='глагол', sentence=x))
    | model
    | StrOutputParser()
)

chain = (
    prompt
    | model
    | StrOutputParser()
    | RunnableParallel(branches={'nouns': nouns_branch_chain, 'verbs': verbs_branch_chain})
    | RunnableLambda(lambda x: agg_result(x['branches']['nouns'], x['branches']['verbs']))
)

result = chain.invoke({'topic': 'математики'})

print(result)

