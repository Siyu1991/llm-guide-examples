"""第 5 期配套：关键词检索（BM25）对比向量检索（bge-small-zh-v1.5），并实测暴力检索的延迟。
首次运行会从 Hugging Face 下载约 95MB 的 ONNX 模型（可设置 HF_ENDPOINT 镜像）。
加 --offline 只跑 BM25，不下载模型。"""
import argparse, json, time, resource
import numpy as np, jieba
from rank_bm25 import BM25Okapi
from data import PRODUCTS, QUERIES
jieba.setLogLevel(60)

def tok(s):
    return [w for w in jieba.lcut_for_search(s.lower()) if w.strip()]

def metrics(ranked_lists, k=3):
    hit, mrr = 0, 0.0
    for ranked, (_, gold) in zip(ranked_lists, QUERIES):
        hit += any(i in gold for i in ranked[:k])
        for r, i in enumerate(ranked, 1):
            if i in gold: mrr += 1 / r; break
    n = len(QUERIES)
    return {f"hit@{k}": round(hit / n, 3), "MRR": round(mrr / n, 3)}

def bm25_rank():
    bm = BM25Okapi([tok(p) for p in PRODUCTS])
    return [list(np.argsort(-bm.get_scores(tok(q)))) for q, _ in QUERIES]

def vec_rank(model_name="BAAI/bge-small-zh-v1.5"):
    from fastembed import TextEmbedding
    t0 = time.perf_counter(); m = TextEmbedding(model_name); load = time.perf_counter() - t0
    t0 = time.perf_counter(); P = np.array(list(m.embed(PRODUCTS))); t_doc = time.perf_counter() - t0
    Q = np.array(list(m.query_embed([q for q, _ in QUERIES])))
    P /= np.linalg.norm(P, axis=1, keepdims=True); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    S = Q @ P.T
    info = {"model": model_name, "dim": int(P.shape[1]), "load_s": round(load, 2),
            "embed_ms_per_doc": round(t_doc / len(PRODUCTS) * 1e3, 2)}
    return [list(np.argsort(-s)) for s in S], S, info

def brute_force_latency(dim=512, sizes=(10_000, 100_000, 300_000)):
    rng = np.random.default_rng(0); out = []
    for n in sizes:
        X = rng.standard_normal((n, dim), dtype=np.float32); q = rng.standard_normal(dim).astype(np.float32)
        X @ q
        ts = []
        for _ in range(10):
            t = time.perf_counter(); s = X @ q; np.argpartition(-s, 10)[:10]; ts.append(time.perf_counter() - t)
        out.append({"n": n, "ms": round(float(np.median(ts)) * 1e3, 2), "MB": round(X.nbytes / 2**20)})
        del X
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true"); a = ap.parse_args()
    res = {"n_products": len(PRODUCTS), "n_queries": len(QUERIES)}
    rb = bm25_rank(); res["BM25"] = metrics(rb)
    if not a.offline:
        rv, S, info = vec_rank(); res["vector"] = {**metrics(rv), **info}
        print("\n查询 → 向量检索 top3 / BM25 top3")
        for (q, gold), v, b in zip(QUERIES, rv, rb):
            mark = lambda L: " | ".join(("✓" if i in gold else "✗") + PRODUCTS[i][:12] for i in L[:3])
            print(f"[{q}]\n   向量: {mark(v)}\n   BM25: {mark(b)}")
        res["brute_force"] = brute_force_latency()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open("result.json", "w"), ensure_ascii=False, indent=1)
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print("OK")

if __name__ == "__main__":
    main()
