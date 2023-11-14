result = chat(
    [
        SystemMessage(content="あなたは親しい友人です。返答は敬語を使わず、フランクに会話してください。"),  #← システムメッセージを使用して設定を追加
        HumanMessage(content="こんにちは！"),
    ]
)
