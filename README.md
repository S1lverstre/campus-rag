# 🎓 安大校园问答助手（Campus RAG Assistant）

[![CI](https://github.com/S1lverstre/campus-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/S1lverstre/campus-rag/actions/workflows/ci.yml)

基于 RAG（检索增强生成）的安徽大学校园知识库问答系统。用户用自然语言提问，系统从本地知识库中检索最相关的文档片段，交由大模型生成**带引用来源**的回答，知识库外问题会诚实拒答。

## ✨ 项目亮点

- **端到端 RAG 流程**：文档加载 → 切分 → 向量化 → FAISS 检索 → Prompt 组装 → LLM 生成
- **答案可溯源**：每个回答都标注引用了哪篇文档的哪些片段，便于核查
- **幻觉抑制**：通过 Prompt 约束 + 检索增强，知识库外问题拒答（"资料中未找到"）
- **自动化评估**：自建 20 题测试集（覆盖校园卡、校园网、宿舍、奖学金、转专业 5 类主题），关键词命中判定，准确率 **100%**（详见 `evaluation.md`）
- **参数对比实验**：对比 `chunk_size` 与召回条数 `k` 的不同配置，量化验证"块太大引入噪声、k 太小漏召回"，最终配置有实验依据（详见 `docs/实验记录-参数对比.md`）

## 🛠 技术栈

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.13 |
| 文档切分 | LangChain RecursiveCharacterTextSplitter（chunk=500, overlap=50） |
| Embedding | BAAI/bge-m3（1024 维，硅基流动 API） |
| 向量检索 | FAISS（余弦相似度，Top-5） |
| 大模型 | DeepSeek-V3（deepseek-chat） |
| 界面 | Streamlit |

## 📂 项目结构

```
campus-rag/
├── data/              # 知识库语料（9 篇安徽大学官方规章制度/指南）
├── faiss_index/       # 构建好的向量索引
├── build_index.py     # 文档切分 + 向量化 + 建索引
├── rag.py             # 命令行问答（检索 + 生成 + 溯源）
├── app.py             # Streamlit 网页版
├── evaluate.py        # 自动化评估脚本
├── evaluation.md      # 评估报告
├── requirements.txt   # 依赖清单
└── .env               # API Key（不入库）
```

## 🚀 本地运行

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. 配置 API Key（在 .env 中填写）
#    DEEPSEEK_API_KEY=...      https://platform.deepseek.com
#    SILICONFLOW_API_KEY=...   https://cloud.siliconflow.cn

# 3. 构建向量索引（首次运行）
python build_index.py

# 4a. 命令行问答
python rag.py

# 4b. 或启动网页版
streamlit run app.py
```

## 📊 评估方法

在 `evaluate.py` 中内置 20 个问题及对应的标准答案关键词，逐题执行完整 RAG 流程，回答命中预设关键词即判定正确，结果输出至 `evaluation.md`。

> 说明：该 100% 是在 9 篇文档、33 个文本块的封闭小语料下取得，问题均可在语料中直接找到答案，**反映的是流程正确性，而非泛化能力**。更严格的评估需加入改写问法、跨文档综合题与知识库外干扰题，并统计拒答准确率。

## 🔬 参数实验

`docs/实验记录-参数对比.md` 记录了两组对照实验：

| 实验 | 变量 | 现象 | 结论 |
|---|---|---|---|
| 一 | chunk_size 500 → 1000 | 块数 33→18，无关片段 5→8 | 块太大 → 噪声增多、精度下降 |
| 二 | k 5 → 1 | 召回 5→1，丢失 2 类关键信息 | k 太小 → 漏召回、答案不完整 |

最终选型：`chunk_size=500 / overlap=50 / k=5`。

## 📄 免责说明

知识库语料来自安徽大学官网公开信息，抓取时间为 2026 年 10 月，政策如有更新请以学校官方最新发布为准。
