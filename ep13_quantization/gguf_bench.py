"""实测同一个小模型不同量化版本：文件大小、内存、速度、困惑度、简单问答正确率。
依赖：pip install llama-cpp-python（CPU 版）；模型：Qwen2.5-0.5B-Instruct 官方 GGUF（fp16/q8_0/q4_k_m/q2_k）。
用法：python gguf_bench.py --dir 模型目录   每个版本在子进程里跑，避免内存叠加。"""
import argparse, json, math, os, resource, subprocess, sys, time

TEXT = ("退货政策：自签收之日起七天内可无理由退货，商品需保持完好、吊牌未拆。生鲜和定制类商品不支持无理由退货。"
        "退货仓库验收通过后一个工作日内发起退款，原路退回到支付宝或微信一般一到三个工作日到账，银行卡最长七个工作日。"
        "订单实付满九十九元包邮，不满九十九元收取运费八元。工作日下午四点前付款的订单当天发货，四点后付款次日发货。")
QA = [("中国的首都是哪座城市？", "北京"), ("一年有几个月？只回答数字。", "12"), ("水的化学式是什么？", "H2O"),
      ("3加5等于几？只回答数字。", "8"), ("太阳从哪个方向升起？", "东"), ("《静夜思》的作者是谁？", "李白"),
      ("一周有几天？只回答数字。", "7"), ("地球绕着什么转？", "太阳"), ("英文 apple 的中文是什么？", "苹果"), ("12乘以12等于多少？只回答数字。", "144")]

def one(path, threads):
    from llama_cpp import Llama
    t0 = time.time(); llm = Llama(model_path=path, n_ctx=1024, n_threads=threads, logits_all=True, verbose=False); load = time.time() - t0
    toks = llm.tokenize(TEXT.encode()); llm.reset(); llm.eval(toks)
    import numpy as np
    lg = np.array(llm.scores[:len(toks)]); lg -= lg.max(1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    nll = -np.mean([lp[i, toks[i + 1]] for i in range(len(toks) - 1)]); ppl = float(math.exp(nll))
    ok = 0; answers = []
    for q, a in QA:
        r = llm.create_chat_completion([{"role": "user", "content": q}], max_tokens=24, temperature=0)["choices"][0]["message"]["content"]
        answers.append(r.strip()[:40]); ok += a in r
    llm.reset(); t = time.time(); out = llm.create_completion("请用三句话介绍一下杭州。", max_tokens=128, temperature=0)
    n = out["usage"]["completion_tokens"]; dec = n / (time.time() - t)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return dict(file_MB=round(os.path.getsize(path) / 2**20), load_s=round(load, 2), peak_RSS_MB=round(rss), ppl=round(ppl, 2),
                qa=f"{ok}/{len(QA)}", decode_tok_s=round(dec, 1), answers=answers)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="models"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--one"); a = ap.parse_args()
    if a.one: print(json.dumps(one(a.one, a.threads), ensure_ascii=False)); sys.exit()
    res = {}
    for q in ["fp16", "q8_0", "q4_k_m", "q2_k"]:
        p = os.path.join(a.dir, f"qwen2.5-0.5b-instruct-{q}.gguf")
        if not os.path.exists(p): print("跳过（未下载）", p); continue
        r = json.loads(subprocess.run([sys.executable, __file__, "--one", p, "--threads", str(a.threads)], capture_output=True, text=True).stdout.strip().splitlines()[-1])
        res[q] = r; print(q, {k: v for k, v in r.items() if k != "answers"})
    os.makedirs("outputs", exist_ok=True); json.dump(res, open("outputs/gguf_results.json", "w"), ensure_ascii=False, indent=1)
