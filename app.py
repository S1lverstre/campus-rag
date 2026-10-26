"""校园知识库问答助手 - Streamlit 网页版"""
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="安大校园问答助手", page_icon="🎓", layout="centered")
st.title("🎓 安大校园问答助手")
st.caption("基于 RAG 的校园知识库问答（LangChain + FAISS + DeepSeek）")

@st.cache_resource
def load_resources():
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    return db, client

db, client = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if q := st.chat_input("请输入你的问题，如：校园卡丢了怎么补办？"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("检索知识库并生成回答中……"):
            hits = db.similarity_search(q, k=5)
            context = "\n\n".join(f"[片段{i+1}] {h.page_content}" for i, h in enumerate(hits))
            prompt = (
                "你是安徽大学校园助手。请仅根据以下资料回答问题，"
                "回答末尾注明参考了哪些片段；如果资料中没有答案，就回答"
                "\"资料中未找到，建议咨询学校相关部门\"。\n\n"
                f"资料：\n{context}\n\n问题：{q}"
            )
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            answer = resp.choices[0].message.content
            st.markdown(answer)

        with st.expander("🔍 查看引用来源"):
            for i, h in enumerate(hits, 1):
                src = os.path.basename(h.metadata.get("source", "?"))
                st.markdown(f"**片段{i}**（{src}）：{h.page_content[:100]}……")

    st.session_state.messages.append({"role": "assistant", "content": answer})
