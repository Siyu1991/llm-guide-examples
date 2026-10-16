"""实测：小模型在 CPU 上，线程数（≈手机/车机可用核数）和上下文长度对生成速度的影响。
依赖 pip install llama-cpp-python；模型默认 Qwen2.5-0.5B-Instruct q4_k_m GGUF。"""
import argparse, json, os, time
from llama_cpp import Llama
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf"); a = ap.parse_args()
res = {"threads": {}, "prefill": {}}
for th in [1, 2, 4, 8]:
    llm = Llama(model_path=a.model, n_ctx=2048, n_threads=th, verbose=False)
    best = 0
    for _ in range(3):  # 共享机器上测速有波动，取 3 次最好成绩
        llm.reset(); t = time.time(); o = llm.create_completion("请介绍一下杭州。", max_tokens=96, temperature=0)
        best = max(best, o["usage"]["completion_tokens"] / (time.time() - t))
    res["threads"][th] = round(best, 1); print(f"{th} 线程：生成 {res['threads'][th]} tok/s"); del llm
llm = Llama(model_path=a.model, n_ctx=2048, n_threads=4, verbose=False)
for n in [128, 512, 1536]:
    toks = llm.tokenize(("退货政策说明。" * 400).encode())[:n]; best = 0
    for _ in range(3):
        llm.reset(); t = time.time(); llm.eval(toks); best = max(best, n / (time.time() - t))
    res["prefill"][n] = round(best, 1); print(f"4 线程：读入 {n} token 的提示，预填充 {res['prefill'][n]} tok/s，首字延迟约 {n/res['prefill'][n]:.2f} 秒")
os.makedirs("outputs", exist_ok=True); json.dump(res, open("outputs/thread_results.json", "w"), indent=1)
