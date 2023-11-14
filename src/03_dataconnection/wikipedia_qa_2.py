import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever

chat = ChatOpenAI()

retriever = WikipediaRetriever(  #← WikipediaRetrieverを初期化する
    lang="ja",  #← Wikipediaの言語を指定する
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Wikipediaの情報を元に質問に答えます。質問を入力してください。").send()

@cl.on_message
async def on_message(message: str):
    chain = RetrievalQA.from_chain_type(  #← RetrievalQAを初期化する
        llm=chat,  #← 質問を生成するための言語モデル
        chain_type="stuff",  #← チェーンの種類を指定する
        retriever=retriever,  #← チェーンの検索を行うためのRetrieverを指定
    )
    result = chain(message) #← RetrievalQAを実行する
    await cl.Message(content=result["result"]).send() #← チェーンを実行する
