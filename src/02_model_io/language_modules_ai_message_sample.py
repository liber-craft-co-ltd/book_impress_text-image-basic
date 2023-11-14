result = chat( #← 実行する
    [
        HumanMessage(content="茶碗蒸しの作り方を教えて"),
        AIMessage(content="{ChatModelからの返答である茶碗蒸しの作り方}"),
        HumanMessage(content="英語に翻訳して"),
    ]
)
