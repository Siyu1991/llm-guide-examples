"""第 7 期补充实验（文章"查询改写"和"重排候选数"两节）。
  实验 4  查询改写：口语问题 vs 改写成文档措辞的问题。改写由作者手写，模拟"让大模型先改写"的效果，
          不调用任何 API；改写只换措辞，不包含答案。
  实验 5  重排候选数：混合检索后把前 N 个候选（N=5/10/20）交给交叉编码器，看命中率和耗时。
加 --no-rerank 只跑实验 4（约 0.4GB 内存）。"""
import argparse, gc, json, os, time, resource
import numpy as np
from advanced import HERE, tok, rrf, score, build
from rag import Embedder

REWRITES = {
    "乙方晚交货一天罚多少？": "乙方逾期交付的违约金如何计算",
    "合同纠纷去哪里打官司？": "合同争议解决方式，向哪个法院提起诉讼",
    "上传文件提示 E3001 是什么意思？": "错误码 E3001 的含义",
    "发票过了多久就彻底不能报销了？": "发票超过多少天不予报销",
    "报销 8000 元需要谁批？": "报销金额 8000 元由谁审批",
    "企业开专票要多久？": "企业开具增值税专用发票需要多长时间",
    "数据存在哪些城市？": "用户数据存储在哪些地域的数据中心",
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-rerank", action="store_true"); a = ap.parse_args()
    kb = os.path.join(HERE, "..", "ep06_rag", "kb")
    allq = json.load(open(os.path.join(HERE, "..", "ep06_rag", "questions.json"), encoding="utf-8")) + \
           json.load(open(os.path.join(HERE, "questions_hard.json"), encoding="utf-8"))
    emb = Embedder(); chunks, X, bm = build(kb, 300, 50, emb)
    def rank(q):
        v = list(np.argsort(-(X @ emb.query(q)))); b = list(np.argsort(-bm.get_scores(tok(q))))
        return v, b, rrf([v[:20], b[:20]])
    res = {}
    qs = [x for x in allq if x["q"] in REWRITES]
    for label, f in [("原问题", lambda x: x["q"]), ("改写后", lambda x: REWRITES[x["q"]])]:
        R = [rank(f(x)) for x in qs]
        res[label] = {m: score([r[i] for r in R], chunks, qs) for i, m in enumerate(["向量", "BM25", "混合RRF"])}
    res["改写题数"] = len(qs)
    if not a.no_rerank:
        hyb = [rank(x["q"])[2] for x in allq]
        del emb; gc.collect()
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        rr = TextCrossEncoder("BAAI/bge-reranker-base")
        list(rr.rerank("预热", ["预热"]))
        res["rerank_topn"] = []
        for n in (5, 10, 20):
            t = time.perf_counter(); out = []
            for L, x in zip(hyb, allq):
                c = L[:n]; s = np.array(list(rr.rerank(x["q"], [chunks[i]["text"] for i in c])))
                out.append([c[j] for j in np.argsort(-s)] + L[n:])
            ms = (time.perf_counter() - t) / len(allq) * 1e3
            res["rerank_topn"].append({"N": n, **score(out, chunks, allq), "ms_per_query": round(ms)})
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(HERE, "extras_result.json"), "w"), ensure_ascii=False, indent=1)
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB"); print("OK")

if __name__ == "__main__":
    main()
