"""第3天里程碑：把 data/ 下的文档切分、向量化并建立 FAISS 索引"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. 加载所有 txt 文档
docs = DirectoryLoader(
    "data", glob="**/*.txt",
    loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
).load()
print(f"共加载 {len(docs)} 篇文档")

# 2. 切分：每块约500字，相邻块重叠50字（保证语义不断裂）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"共切出 {len(chunks)} 个文本块")

# 3. 向量化：硅基流动的 bge-m3（免费，1024维）
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
    chunk_size=32,
)

# 4. 建立 FAISS 索引并保存到本地
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("索引已保存到 faiss_index/ 目录")
