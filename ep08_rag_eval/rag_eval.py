"""RAG 效果评估最小示例：检索指标 + 生成指标 + 自助法置信区间。
纯 Python 标准库，CPU 秒级跑完。设置 OPENAI_API_KEY 时可用大模型当裁判，否则用离线规则裁判。
用法：python rag_eval.py [--k 3] [--chunk 0]
"""
import argparse, json, math, os, random, re, urllib.request
from collections import Counter
from data import DOCS, QA

def grams(s):
    s = re.sub(r"[\s，。、：；！？,.:;!?（）()]", "", s)
    return [s[i:i+2] for i in range(len(s)-1)] + list(s)

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.ids = list(docs); self.tf = {i: Counter(grams(docs[i])) for i in self.ids}
        self.len = {i: sum(self.tf[i].values()) for i in self.ids}
        self.avg = sum(self.len.values()) / len(self.ids); self.k1, self.b = k1, b
        df = Counter(g for i in self.ids for g in self.tf[i]); N = len(self.ids)
        self.idf = {g: math.log(1 + (N - n + .5) / (n + .5)) for g, n in df.items()}
    def search(self, q, k):
        sc = {}
        for i in self.ids:
            s = 0.0
            for g in set(grams(q)):
                f = self.tf[i].get(g, 0)
                if f: s += self.idf[g] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            sc[i] = s
        return sorted(sc, key=lambda i: -sc[i])[:k]

class Overlap:  # 弱基线：只数重合字数，不考虑词频和稀有度
    def __init__(self, docs): self.docs = docs
    def search(self, q, k):
        qs = set(q); return sorted(self.docs, key=lambda i: -len(qs & set(self.docs[i])))[:k]

def retrieval_metrics(ranked, gold, k):
    rel = [1 if d in gold else 0 for d in ranked[:k]]
    hit = 1.0 if any(rel) else 0.0
    recall = sum(rel) / len(gold)
    mrr = next((1 / (r + 1) for r, x in enumerate(rel) if x), 0.0)
    dcg = sum(x / math.log2(r + 2) for r, x in enumerate(rel))
    idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gold), k)))
    return dict(hit=hit, recall=recall, mrr=mrr, ndcg=dcg / idcg)

def answer(q, ctx_ids):
    """离线"生成器"：从检索到的文档里挑与问题最相关的一句（抽取式），保证可复现。"""
    sents = [s for i in ctx_ids for s in re.split(r"[。]", DOCS[i]) if s]
    qg = set(grams(q))
    return max(sents, key=lambda s: len(qg & set(grams(s))) / (1 + 0.02 * len(s)))

def judge_offline(q, ans, ctx_ids, keys):
    correct = all(k in ans for k in keys)
    ctx = "".join(DOCS[i] for i in ctx_ids)
    ag = grams(ans); faithful = sum(g in ctx for g in ag) / max(1, len(ag))
    return dict(correct=float(correct), faithful=faithful)

def judge_llm(q, ans, ctx_ids, keys):
    body = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
            "messages": [{"role": "user", "content":
              f"资料：{''.join(DOCS[i] for i in ctx_ids)}\n问题：{q}\n回答：{ans}\n参考要点：{keys}\n"
              "只输出JSON：{\"correct\":0或1,\"faithful\":0到1之间的小数}。correct=回答是否包含参考要点的意思；faithful=回答内容有多大比例能在资料里找到依据。"}]}
    req = urllib.request.Request(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    return json.loads(re.search(r"\{.*\}", txt, re.S).group())

def bootstrap(xs, n=2000, seed=0):
    rnd = random.Random(seed); m = []
    for _ in range(n):
        s = [rnd.choice(xs) for _ in xs]; m.append(sum(s) / len(s))
    m.sort(); return m[int(.025 * n)], m[int(.975 * n)]

def run(k=3, retriever="bm25", use_llm=False):
    R = BM25(DOCS) if retriever == "bm25" else Overlap(DOCS)
    judge = judge_llm if use_llm else judge_offline
    rows = []
    for q, keys, gold in QA:
        ranked = R.search(q, k); a = answer(q, ranked)
        rows.append(dict(q=q, ranked=ranked, answer=a, **retrieval_metrics(ranked, gold, k), **judge(q, a, ranked, keys)))
    agg = {m: sum(r[m] for r in rows) / len(rows) for m in ["hit", "recall", "mrr", "ndcg", "correct", "faithful"]}
    agg["correct_ci95"] = bootstrap([r["correct"] for r in rows])
    # 失败归因：检索没找到 vs 找到了但答错
    agg["fail_retrieval"] = sum(1 for r in rows if not r["correct"] and r["hit"] == 0)
    agg["fail_generation"] = sum(1 for r in rows if not r["correct"] and r["hit"] == 1)
    return agg, rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=3); a = ap.parse_args()
    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    print("裁判：", "大模型" if use_llm else "离线规则（未设置 OPENAI_API_KEY）")
    out = {}
    for ret in ["overlap", "bm25"]:
        for k in [1, 3, 5]:
            agg, rows = run(k, ret, use_llm); out[f"{ret}@{k}"] = agg
            print(f"{ret:8s} k={k}  Hit={agg['hit']:.2f} Recall={agg['recall']:.2f} MRR={agg['mrr']:.2f} nDCG={agg['ndcg']:.2f} "
                  f"答对率={agg['correct']:.2f} 95%CI=[{agg['correct_ci95'][0]:.2f},{agg['correct_ci95'][1]:.2f}] 忠实度={agg['faithful']:.2f} "
                  f"失败:检索{agg['fail_retrieval']}/生成{agg['fail_generation']}")
    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open("outputs/results.json", "w"), ensure_ascii=False, indent=1)
    _, rows = run(a.k, "bm25", use_llm)
    for r in rows:
        if not r["correct"]: print("答错：", r["q"], "→", r["answer"], r["ranked"])
