"""第3天里程碑：基于 FAISS 检索 + DeepSeek 生成的校园问答"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

print("=== 安大校园问答助手（输入 exit 退出）===\n")
while True:
    q = input("你：").strip()
    if not q:
        continue
    if q.lower() == "exit":
        break

    # 1. 检索：从知识库找最相关的5个片段
    hits = db.similarity_search(q, k=1)

    # 2. 组装 Prompt：资料 + 问题，要求模型只依据资料回答
    context = "\n\n".join(f"[片段{i+1}] {h.page_content}" for i, h in enumerate(hits))
    prompt = (
        "你是安徽大学校园助手。请仅根据以下资料回答问题，"
        "回答末尾注明参考了哪些片段；如果资料中没有答案，就回答"
        "\"资料中未找到，建议咨询学校相关部门\"。\n\n"
        f"资料：\n{context}\n\n问题：{q}"
    )

    # 3. 生成回答
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    print("\n助手：" + resp.choices[0].message.content)
    print("\n[引用来源]")
    for i, h in enumerate(hits, 1):
        print(f"  片段{i}（{h.metadata.get('source', '?')}）：{h.page_content[:60]}……")
    print()
