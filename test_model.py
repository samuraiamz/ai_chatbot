from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3001/v1",
    api_key="freellmapi-01050f4834803d768e7ece602f68b8ee0694157474c18df1"
)

response = client.chat.completions.create(
    model="@cf/openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Explain what artificial intelligence is."
        }
    ]
)

print(response.choices[0].message.content)
