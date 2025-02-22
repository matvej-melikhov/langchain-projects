from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import os
import sys


current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, 'datasets/lenta-ru-news-short.csv')
index_dir = os.path.join(current_path, 'faiss-index')

# create documents
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

# create or load vector store
if not os.path.exists(index_dir):
    db = FAISS.from_documents(
        documents=docs, # 10_000 documents
        embedding=embeddings,
    )
    db.save_local(folder_path=index_dir, index_name='index')
else:
    db = FAISS.load_local(
        folder_path=index_dir,
        index_name='index',
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

# creating retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1},
)

# creating LLM
sys.path.append('..')  # add parent directory to path
import utils

llm = utils.init_gigachat_model()

# building RAG
system_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''Отвечай только с учетом полученного контекста.
        Если в нем нет ответа, скажи, что не знаешь.
        \n Контекст:\n{context}'''
    ),
    ('human', '{input}')
])

qa_chain = create_stuff_documents_chain(llm, system_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ask a question
'''
Статья:
'Нападающий Лионель Месси будет получать более миллиона фунтов стерлингов (1,3 миллиона долларов) 
в неделю по новому контракту с «Барселоной». Об этом сообщает The Sunday Times.
По информации источника, зарплата аргентинца составит 54,8 миллиона фунтов (70,6 миллиона долларов) в год.
За четыре года «Барселона» выплатит Месси около 220 миллионов фунтов (283,6 миллиона долларов) без учета бонусов.
При этом аргентинец станет первым футболистом в истории, чья зарплата превысит миллион фунтов в неделю.
5 июля Месси продлил контракт с «Барселоной». Новое соглашение действует до 2021 года.
Предыдущий договор истекал летом 2018 года. Во время переговоров о новом контракте появлялась
информация о желании аргентинского форварда покинуть каталонский клуб. 23 июня сообщалось,
что государственный прокурор Испании согласился заменить Месси наказание.
Вместо отбывания тюремного срока футболист заплатит штраф. Аргентинец и его отец
обвинялись в уклонении от уплаты налогов на сумму в 4,1 миллиона евро.
Месси выступает за «Барселону» с 2003 года. Нападающий пять раз выигрывал «Золотой мяч».'
'''

query = 'Сколько зарабатывает Лионель Месси в Барселоне?'
result = rag_chain.invoke({'input': query})['answer']

print(f'Question: {query}\n\nAnswer:\n> {result}\n____________________\n')

query = 'Сколько лет Месси играет за Барселону?'
result = rag_chain.invoke({'input': query})['answer']

print(f'Question: {query}\n\nAnswer:\n> {result}\n____________________\n')