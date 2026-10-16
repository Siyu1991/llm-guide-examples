"""第 7 期配套：检索不准时的三板斧，逐项实测。
  实验 1  切块大小：100 / 300 / 800 字，看命中率和送进模型的字数
  实验 2  检索方式：BM25 关键词 / 向量 / 混合（RRF 融合）/ 再加交叉编码器重排
  实验 3  元数据过滤：剔除"已废止"文档后，版本冲突类问题的 top1 是否正确
知识库复用 ../ep06_rag/kb（虚构公司文档），问题 = ep06 的 24 题 + 本目录 12 道难题。
重排模型 BAAI/bge-reranker-base 约 1GB，首次运行会下载；内存紧张时加 --no-rerank。"""
import argparse, gc, json, os, sys, time, resource
import numpy as np, jieba
from rank_bm25 import BM25Okapi
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ep06_rag"))
from rag import load_docs, chunk_markdown, Embedder
jieba.setLogLevel(60)

def tok(s):
    return [w for w in jieba.lcut_for_search(s.lower()) if w.strip() and w not in "，。？！、：；（）【】｜ "]

def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion：score = Σ 1/(k + 名次)。只用名次，不用分数，BM25 和余弦分值不在同一量纲时也能融合。"""
    sc = {}
    for L in rank_lists:
        for r, i in enumerate(L):
            sc[i] = sc.get(i, 0) + 1 / (k + r + 1)
    return sorted(sc, key=lambda i: -sc[i])

def score(ranked, chunks, qs, ks=(1, 3)):
    out = {f"hit@{k}": 0.0 for k in ks}; mrr = 0.0
    for L, item in zip(ranked, qs):
        pos = next((r for r, i in enumerate(L) if item["a"] in chunks[i]["text"]), None)
        for k in ks: out[f"hit@{k}"] += pos is not None and pos < k
        mrr += 0 if pos is None else 1 / (pos + 1)
    n = len(qs)
    return {**{k: round(v / n, 3) for k, v in out.items()}, "MRR": round(mrr / n, 3)}

def naive_chunks(name, text, size, overlap):
    """对照组：不看标题，整篇按固定字数硬切，块上不带文档标题。"""
    step = max(1, size - overlap)
    return [{"doc": name, "title": "", "section": "", "text": text[i:i + size]} for i in range(0, len(text), step)]

def build(kb, size, overlap, emb, naive=False):
    cut = naive_chunks if naive else chunk_markdown
    chunks = [c for n, t in load_docs(kb) for c in cut(n, t, size, overlap)]
    X = emb.docs([c["text"] for c in chunks])
    bm = BM25Okapi([tok(c["text"]) for c in chunks])
    return chunks, X, bm

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-rerank", action="store_true"); a = ap.parse_args()
    kb = os.path.join(HERE, "..", "ep06_rag", "kb")
    qs = json.load(open(os.path.join(HERE, "..", "ep06_rag", "questions.json"), encoding="utf-8")) + \
         json.load(open(os.path.join(HERE, "questions_hard.json"), encoding="utf-8"))
    emb = Embedder(); res = {"n_questions": len(qs)}
    # 实验 1：切块大小（纯向量检索）
    res["chunk_size"] = []
    for naive, size in [(True, 100), (True, 300), (True, 800), (False, 100), (False, 300)]:
        chunks, X, _ = build(kb, size, size // 6, emb, naive)
        Q = np.stack([emb.query(x["q"]) for x in qs])
        ranked = [list(np.argsort(-s)) for s in Q @ X.T]
        avg3 = np.mean([sum(len(chunks[i]["text"]) for i in L[:3]) for L in ranked])
        res["chunk_size"].append({"方式": "固定字数盲切" if naive else "按标题切+带标题", "size": size, "n_chunks": len(chunks), **score(ranked, chunks, qs),
                                  "top3_chars": int(avg3)})
    # 实验 2：检索方式（切块 300）
    chunks, X, bm = build(kb, 300, 50, emb)
    Q = np.stack([emb.query(x["q"]) for x in qs])
    vec = [list(np.argsort(-s)) for s in Q @ X.T]
    kw = [list(np.argsort(-bm.get_scores(tok(x["q"])))) for x in qs]
    hyb = [rrf([v[:20], b[:20]]) for v, b in zip(vec, kw)]
    res["methods"] = {"BM25": score(kw, chunks, qs), "向量": score(vec, chunks, qs), "混合RRF": score(hyb, chunks, qs)}
    res["per_question_misses"] = {name: [x["q"] for L, x in zip(R, qs) if not any(x["a"] in chunks[i]["text"] for i in L[:1])]
                                  for name, R in [("BM25", kw), ("向量", vec), ("混合RRF", hyb)]}
    # 实验 3：元数据过滤（版本冲突类问题）
    ver_q = [i for i, x in enumerate(qs) if x["src"] == "01"]   # 报销类问题：库里同时有 2026 版和已废止的 2024 版
    def ver_eval(R, flt):
        sub = [[i for i in R[j] if not flt or "废止" not in chunks[i]["title"]] for j in ver_q]
        return {"hit@1": sum(qs[j]["a"] in chunks[L[0]]["text"] for j, L in zip(ver_q, sub)),
                "旧版进入top3": sum(any(chunks[i]["doc"].startswith("02") for i in L[:3]) for L in sub)}
    res["version_filter"] = {"questions": len(ver_q), "混合_不过滤": ver_eval(hyb, False), "混合_过滤已废止": ver_eval(hyb, True)}
    if not a.no_rerank:
        del emb; gc.collect()
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        t = time.perf_counter(); rr = TextCrossEncoder("BAAI/bge-reranker-base"); load = time.perf_counter() - t
        def rerank(cands, q, topn=10):
            c = cands[:topn]; s = list(rr.rerank(q, [chunks[i]["text"] for i in c]))
            return [c[j] for j in np.argsort(-np.array(s))] + cands[topn:]
        t = time.perf_counter()
        vec_rr = [rerank(L, x["q"]) for L, x in zip(vec, qs)]
        hyb_rr = [rerank(L, x["q"]) for L, x in zip(hyb, qs)]
        per_q = (time.perf_counter() - t) / (2 * len(qs))
        res["methods"]["向量+重排"] = score(vec_rr, chunks, qs)
        res["methods"]["混合+重排"] = score(hyb_rr, chunks, qs)
        res["rerank_cost"] = {"model": "BAAI/bge-reranker-base", "load_s": round(load, 1), "ms_per_query_top10": round(per_q * 1e3)}
        res["version_filter"]["混合+重排_不过滤"] = ver_eval(hyb_rr, False)
        res["version_filter"]["混合+重排_过滤已废止"] = ver_eval(hyb_rr, True)
        res["per_question_misses"]["混合+重排"] = [x["q"] for L, x in zip(hyb_rr, qs) if not any(x["a"] in chunks[i]["text"] for i in L[:1])]
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), ensure_ascii=False, indent=1)
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print("OK")

if __name__ == "__main__":
    main()
