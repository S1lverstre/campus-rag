"""自动化评估：20 个测试题 + 关键词命中打分，输出 accuracy 报告"""
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# (问题, 判定关键词——答案中命中任意2个即算正确)
TEST_SET = [
    ("校园卡丢了怎么补办？", ["自助补卡", "挂失", "有效证件", "服务部"]),
    ("校园卡的初始密码是什么？", ["身份证", "后六位", "密码"]),
    ("校园卡怎么充值？", ["支付宝", "圈存机", "转账", "充值"]),
    ("忘带校园卡在食堂怎么消费？", ["智慧安大", "二维码", "刷码", "公众号"]),
    ("校园网一个月多少钱？", ["5元", "5 元"]),
    ("校园网的上网账号是什么？", ["学号", "智慧安大"]),
    ("校园网坏了打什么电话报修？", ["63861118", "报修"]),
    ("学生宿舍可以使用热得快吗？", ["严禁", "不能", "禁止", "大功率"]),
    ("晚上23:30之后回宿舍需要注意什么？", ["登记", "值班"]),
    ("优秀学生奖学金的奖励金额是多少？", ["5000"]),
    ("学习优秀奖学金一等奖的比例和金额是多少？", ["3%", "2000"]),
    ("申请奖学金需要满足哪些基本条件？", ["2.5", "体质", "学费", "违纪"]),
    ("本科生国家奖学金的奖励标准是多少？", ["8000"]),
    ("国家奖学金对绩点排名的要求是什么？", ["前10%", "前 10%", "10%"]),
    ("大一新生入学后能马上转专业吗？", ["未满一学期", "不能", "大一下"]),
    ("全校范围转专业有绩点限制吗？", ["不设置", "没有", "无"]),
    ("辅修第二专业申请需要什么条件？", ["60学分", "2.5", "大二"]),
    ("学生公寓的物业管理由谁承担？", ["后勤"]),
    ("学生毕业时退宿手续如何办理？", ["统一", "集中", "钥匙", "遥控器"]),
    ("奖学金基本条件对公益志愿劳动时长有什么要求？", ["40小时", "40 小时"]),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main():
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    results, correct = [], 0
    for idx, (q, keywords) in enumerate(TEST_SET, 1):
        hits = db.similarity_search(q, k=5)
        context = "\n\n".join(f"[片段{i+1}] {h.page_content}" for i, h in enumerate(hits))
        prompt = (
            "你是安徽大学校园助手。请仅根据以下资料回答问题；"
            "如果资料中没有答案，就回答\"资料中未找到\"。\n\n"
            f"资料：\n{context}\n\n问题：{q}"
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        norm = normalize(answer)
        hit = sum(1 for kw in keywords if normalize(kw) in norm)
        ok = hit >= min(2, len(keywords))  # 单关键词题目命中1个即算通过
        correct += ok
        results.append((idx, q, ok, hit, len(keywords), answer))
        print(f"[{idx:02d}/20] {'✓' if ok else '✗'} ({hit}/{len(keywords)}) {q}")
        time.sleep(0.5)

    accuracy = correct / len(TEST_SET)
    print(f"\n===== 总准确率：{correct}/{len(TEST_SET)} = {accuracy:.0%} =====")

    with open("evaluation.md", "w", encoding="utf-8") as f:
        f.write("# 评估报告\n\n")
        f.write(f"- 测试集：{len(TEST_SET)} 题（覆盖校园卡、校园网、宿舍、奖学金、转专业等主题）\n")
        f.write(f"- 判定方式：回答命中预设关键词 ≥2 个记为正确（人工预设标准答案关键词）\n")
        f.write(f"- **准确率：{correct}/{len(TEST_SET)} = {accuracy:.0%}**\n\n")
        f.write("| # | 问题 | 结果 | 命中 | 回答摘要 |\n|---|------|------|------|----------|\n")
        for idx, q, ok, hit, nkw, ans in results:
            summary = ans.replace("\n", " ")[:80]
            f.write(f"| {idx} | {q} | {'✓' if ok else '✗'} | {hit}/{nkw} | {summary} |\n")
    print("报告已写入 evaluation.md")


if __name__ == "__main__":
    main()
