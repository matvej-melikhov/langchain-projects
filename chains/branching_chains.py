from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch
import utils


model = utils.init_gigachat_model()

positive_feedback_template = ChatPromptTemplate.from_messages([
    SystemMessage('Ты должен отвечать очень вежливо и уважительно.'),
    (
        'human',
        'Поблагодари пользователя за положительный отзыв. Его отзыв: {feedback}'
    )
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    SystemMessage('Ты должен отвечать очень вежливо и уважительно.'),
    (
        'human',
        '''Пользователь не доволен, извинись за то, что не оправдали его ожиданий.
        Его отзыв: {feedback}'''
    )
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    SystemMessage('Ты должен отвечать очень вежливо и уважительно.'),
    (
        'human',
        'Ответь на нейтральный отзыв пользователя. Его отзыв: {feedback}'
    )
])

escalate_feedback_to_human = RunnableLambda(lambda x: 'Передаю запрос оператору...')

feedback_classification_template = ChatPromptTemplate.from_messages([
    SystemMessage('Ты очень тонко чувствуешь эмоциональный окрас текста.'),
    (
        'human',
        '''Определи эмоциональный окрас отзыва. Выбери из трех вариантов:
        positive, negative, neutral.\nОтзыв: {feedback}
        '''
    )
])

branches = RunnableBranch(
    (
        lambda x: 'positive' in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'neutral' in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_to_human
)

chain = (
    feedback_classification_template
    | model
    | StrOutputParser()
    | branches
)

review = 'Мне очень понравилось это заведение. Ставлю ему 9/10 баллов'
result = chain.invoke({'feedback': review})

print(result)