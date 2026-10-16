"""第 5 期补充实验：三种向量索引的速度、召回率和内存（文章"作者实测：三种索引"一节）。
数据为合成聚类向量（20 万条 × 384 维），以暴力检索前 10 条为标准答案计算召回率@10。
依赖：pip install faiss-cpu numpy。单线程运行约 2 分钟，峰值内存约 1GB。--n 可调小数据量。"""
import argparse, json, os, time, resource
import numpy as np, faiss

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200_000); a = ap.parse_args()
    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(0); N, d, Q = a.n, 384, 500
    C = rng.standard_normal((2000, d)).astype("float32")
    X = (C[rng.integers(0, 2000, N)] + 0.6 * rng.standard_normal((N, d))).astype("float32"); faiss.normalize_L2(X)
    q = (C[rng.integers(0, 2000, Q)] + 0.6 * rng.standard_normal((Q, d))).astype("float32"); faiss.normalize_L2(q)
    flat = faiss.IndexFlatIP(d); flat.add(X)
    t = time.perf_counter(); _, gt = flat.search(q, 10); tf = (time.perf_counter() - t) / Q * 1e3
    rec = lambda I: float(np.mean([len(set(I[i]) & set(gt[i])) / 10 for i in range(Q)]))
    res = [dict(name="Flat 暴力", ms=tf, recall=1.0, mem_mb=N * d * 4 / 2**20)]
    t = time.perf_counter(); h = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 80; h.add(X); bh = time.perf_counter() - t
    for ef in (16, 64, 256):
        h.hnsw.efSearch = ef; t = time.perf_counter(); _, I = h.search(q, 10)
        res.append(dict(name=f"HNSW M=32 efSearch={ef}", ms=(time.perf_counter() - t) / Q * 1e3, recall=rec(I), mem_mb=N * (d * 4 + 32 * 2 * 4) / 2**20, build_s=bh))
    del h
    nlist = 1024 if N >= 100_000 else max(16, int(4 * N ** 0.5))
    t = time.perf_counter(); ivf = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
    ivf.train(X[:min(N, 60_000)]); ivf.nprobe = 32; bi = time.perf_counter() - t
    r = faiss.IndexRefineFlat(ivf); r.add(X)   # 同时存 PQ 编码和原始向量，便于对比"只用 PQ"和"PQ + 精排"
    for kf, label in [(1, "IVF-PQ 48 字节 nprobe=32"), (10, "IVF-PQ + 原始向量精排前 100"), (50, "IVF-PQ + 原始向量精排前 500")]:
        if kf == 1:
            t = time.perf_counter(); _, I = ivf.search(q, 10)
        else:
            r.k_factor = kf; t = time.perf_counter(); _, I = r.search(q, 10)
        res.append(dict(name=label, ms=(time.perf_counter() - t) / Q * 1e3, recall=rec(I), mem_mb=N * (48 + 8) / 2**20 + (0 if kf == 1 else N * d * 4 / 2**20), build_s=bi))
    for x in res: print(f"{x['name']:<32} {x['ms']:7.3f} ms/查询  召回@10 {x['recall']:.3f}  内存约 {x['mem_mb']:.1f} MB")
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ann_result.json"), "w"), ensure_ascii=False, indent=1)
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB"); print("OK")

if __name__ == "__main__":
    main()
