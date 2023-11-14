import openai  #← OpenAIが用意しているPythonパッケージをインポートする

response = openai.ChatCompletion.create(  #← OpenAIのAPIを呼び出すことですことで、言語モデルを呼び出している
    model="gpt-3.5-turbo",  #← 呼び出す言語モデルの名前
    messages=[
        {
            "role": "user",
            "content": "iPhone8のリリース日を教えて"  #← 入力する文章(プロンプト)
        },
    ]
)
print(response) #← 結果を表示
