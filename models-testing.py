import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": "You are a free role AI support agent."
        },
        {
            "role": "user",
            "content": "My account is locked. How can I unlock it? answer it however youl like"
        }
    ],
    reasoning_effort="low",
    include_reasoning=False,
    temperature=0.3,
    max_completion_tokens=1024
)

print(response.choices[0].message.content)