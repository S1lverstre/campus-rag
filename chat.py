"""第2天里程碑：命令行聊天机器人（DeepSeek API）"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

history = [
    {"role": "system", "content": "你是安徽大学的校园助手，用简洁的中文回答。不知道就说不知道。"}
]

print("=== 校园助手（输入 exit 退出）===")
while True:
    q = input("你：").strip()
    if not q:
        continue
    if q.lower() == "exit":
        break
    history.append({"role": "user", "content": q})
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        temperature=0.7,
    )
    answer = resp.choices[0].message.content
    print("助手：" + answer + "\n")
    history.append({"role": "assistant", "content": answer})
