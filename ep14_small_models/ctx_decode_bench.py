"""延伸实测：上下文越长，生成每个新 token 越慢吗？以及 n_ctx 设大了要多付多少内存。
解码时除了读全部权重，还要读已有的 KV 缓存；上下文越长，要读的越多。
依赖 llama-cpp-python；模型默认 Qwen2.5-0.5B-Instruct q4_k_m GGUF。"""
import argparse, json, os, resource, time
from llama_cpp import Llama

def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"): return int(line.split()[1]) / 1024

def decode_speed(llm, prefix, n_new=64):
    llm.reset(); llm.eval(prefix); t = time.time()
    for _ in range(n_new):
        tok = llm.sample(top_k=1); llm.eval([tok])
    return n_new / (time.time() - t)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
    res = {"decode_vs_ctx": {}, "rss_vs_nctx": {}}
    base = rss_mb()
    for nctx in [2048, 8192, 32768]:
        llm = Llama(model_path=a.model, n_ctx=nctx, n_threads=a.threads, verbose=False)
        res["rss_vs_nctx"][nctx] = round(rss_mb() - base, 0); print(f"n_ctx={nctx}: 加载后进程内存增加 {res['rss_vs_nctx'][nctx]:.0f} MB")
        del llm
    llm = Llama(model_path=a.model, n_ctx=8192, n_threads=a.threads, verbose=False)
    text = ("退货政策说明：签收七天内可无理由退货，生鲜和定制商品除外。" * 600).encode()
    for n in [16, 1024, 3072, 6144]:
        toks = llm.tokenize(text)[:n]
        best = max(decode_speed(llm, toks) for _ in range(3))
        res["decode_vs_ctx"][n] = round(best, 1); print(f"已有上下文 {n} token：生成 {best:.1f} tok/s")
    os.makedirs("outputs", exist_ok=True); json.dump(res, open("outputs/ctx_decode_results.json", "w"), indent=1)
