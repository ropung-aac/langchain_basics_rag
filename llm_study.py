from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client  = OpenAI()

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [
    {"role":"user", "content": "사람의 병을 고치는 직업은 뭐야?"},
    {"role":"assistant", "content": "의사입니다."},
    {"role":"user", "content": "그 직업은 얼마나 공부해야해?"}
    ]
)

print(response.choices[0].message.content)
