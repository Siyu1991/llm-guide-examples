"""第 4 期配套：思考 token 的成本与收益。
子命令：
  cost   —— 按"思考 token 计入输出价"估算一项业务的月费
  vote   —— 自洽投票（self-consistency）：单次正确率 p 时，采样 N 次多数表决的正确率（精确计算 + 模拟）
  probe  —— 调用 OpenAI 兼容接口，对比不同推理强度下的 reasoning_tokens；
            没有设置 OPENAI_API_KEY 时自动使用离线 mock（结果标注 MOCK，仅用于跑通流程）
"""
import argparse, json, os, math, random, resource, urllib.request

def monthly_cost(calls, inp, visible_out, thinking, price_in, price_out):
    per = inp * price_in / 1e6 + (visible_out + thinking) * price_out / 1e6
    return per * calls, per

def majority_exact(p, n, k_wrong=3):
    """单次正确率 p，错误答案均匀分散到 k_wrong 个不同错答上；计算 n 次采样后正确答案得票严格最多的概率
    （平票按随机打破）。用多项分布枚举，n<=15 时很快。"""
    q = (1 - p) / k_wrong
    total = 0.0
    def rec(i, remain, counts, prob):
        nonlocal total
        if i == k_wrong:
            c = remain  # 正确答案票数
            if prob == 0: return
            pr = prob * p ** c
            m = max(counts) if counts else 0
            if c > m: total += pr
            elif c == m and c > 0: total += pr / (1 + counts.count(m))
            return
        for x in range(remain + 1):
            rec(i + 1, remain - x, counts + [x], prob * math.comb(remain, x) * q ** x)
    rec(0, n, [], 1.0)
    return total

def majority_sim(p, n, k_wrong=3, trials=20000, seed=0):
    rnd = random.Random(seed); ok = 0
    for _ in range(trials):
        votes = [0] * (k_wrong + 1)
        for _ in range(n):
            votes[0 if rnd.random() < p else rnd.randint(1, k_wrong)] += 1
        m = max(votes); top = [i for i, v in enumerate(votes) if v == m]
        ok += rnd.choice(top) == 0
    return ok / trials

QUESTIONS = [
    "一个水池有进水管和出水管，进水管 6 小时注满，出水管 9 小时放空，同时打开几小时注满？只回答数字。",
    "把 'llm-guide' 反转后是什么？只回答结果。",
    "法国的首都是哪里？只回答城市名。",
]

def call_openai_compatible(q, effort):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("REASONING_MODEL", "o4-mini")
    body = {"model": model, "messages": [{"role": "user", "content": q}], "reasoning_effort": effort}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    u = d.get("usage", {})
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {"answer": d["choices"][0]["message"]["content"], "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "reasoning_tokens": rt}

def call_mock(q, effort):
    """离线 mock：不联网，按题目长度和强度给出确定性的假用量，只为跑通流程。"""
    base = len(q) * {"low": 3, "medium": 8, "high": 20}[effort]
    return {"answer": "MOCK", "prompt_tokens": len(q), "completion_tokens": base + 10, "reasoning_tokens": base, "mock": True}

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost")
    c.add_argument("--calls", type=float, default=1e6); c.add_argument("--inp", type=int, default=800)
    c.add_argument("--out", type=int, default=200); c.add_argument("--think", type=int, nargs="+", default=[0, 1000, 4000, 16000])
    c.add_argument("--pin", type=float, default=2.0); c.add_argument("--pout", type=float, default=10.0)
    v = sub.add_parser("vote"); v.add_argument("--p", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    v.add_argument("--n", type=int, nargs="+", default=[1, 3, 5, 9, 15])
    sub.add_parser("probe")
    a = ap.parse_args()
    if a.cmd == "cost":
        for t in a.think:
            m, per = monthly_cost(a.calls, a.inp, a.out, t, a.pin, a.pout)
            print(f"思考 {t:>6} token：单次 ${per:.5f}，每月 ${m:,.0f}")
    elif a.cmd == "vote":
        rows = []
        for p in a.p:
            for n in a.n:
                ex, sim = majority_exact(p, n), majority_sim(p, n)
                assert abs(ex - sim) < 0.02, (p, n, ex, sim)
                rows.append({"p": p, "n": n, "exact": round(ex, 4), "sim": round(sim, 4)})
                print(f"单次正确率 {p:.0%}，采样 {n:>2} 次投票 → {ex:.1%}（模拟 {sim:.1%}）")
        json.dump(rows, open("vote_result.json", "w"), indent=1)
    else:
        live = bool(os.environ.get("OPENAI_API_KEY"))
        fn = call_openai_compatible if live else call_mock
        print("模式：", "在线 API" if live else "离线 MOCK（未检测到 OPENAI_API_KEY）")
        for q in QUESTIONS:
            for e in ["low", "high"]:
                r = fn(q, e)
                print(json.dumps({"q": q[:14], "effort": e, **r}, ensure_ascii=False))
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print("OK")

if __name__ == "__main__":
    main()
