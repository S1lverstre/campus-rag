FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# faiss_index 由 build_index.py 构建（需 SILICONFLOW_API_KEY），构建镜像时挂载或运行时生成
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true
