from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever

chat = ChatOpenAI()

retriever = WikipediaRetriever(  #← WikipediaRetrieverを初期化する
    lang="ja",  #← Wikipediaの言語を指定する
    doc_content_chars_max=500  #← 取得するテキストの最大文字数を指定する
)

chain = RetrievalQA.from_llm(
    llm=chat,
    retriever=retriever,
    return_source_documents=True,
)
result = chain("バーボンウイスキーとは？") #← RetrievalQAを実行する
source_documents = result["source_documents"] #← 情報の取得元のドキュメントを取得する
print(f"検索結果: {len(source_documents)}件") #← 検索結果の件数を表示する
for document in source_documents:
    print("---------------取得したメタデータ---------------")
    print(document.metadata)
    print("---------------取得したテキスト---------------")
    print(document.page_content[:100])
print("---------------返答---------------")
print(result["result"]) #← 返答を表示する
