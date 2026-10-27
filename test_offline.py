"""CI 离线冒烟测试：不调用任何外部 API，验证语料完整性、切分逻辑与评估集可用性"""
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


def test_corpus_files_exist_and_nonempty():
    """知识库语料完整：9 篇规章制度文档全部在库且非空"""
    files = [f for f in os.listdir("data") if f.endswith(".txt")]
    assert len(files) == 9, f"语料应为 9 篇，实际 {len(files)}"
    for f in files:
        with open(os.path.join("data", f), encoding="utf-8") as fh:
            assert len(fh.read().strip()) > 100, f"{f} 内容过短，疑似损坏"


def test_splitter_chunk_params():
    """切分参数符合实验结论：chunk=500 / overlap=50，长文本被切成多块且每块不超限"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text("字" * 1200)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_eval_set_integrity():
    """评估集完整：20 题且每题都有判定关键词"""
    from evaluate import TEST_SET, normalize
    assert len(TEST_SET) == 20
    assert all(len(keywords) >= 1 for _, keywords in TEST_SET)
    # normalize 应去除空白，保证关键词判定稳定
    assert normalize("自 助 补 卡") == "自助补卡"
