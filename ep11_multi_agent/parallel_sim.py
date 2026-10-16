"""延伸模拟：主管 + 子 Agent 的两笔账——墙钟时间和主 Agent 的上下文。
K 个彼此独立的调研子任务，每个子 Agent 做 STEPS 步（搜索/阅读），每步耗时服从对数正态分布（均值约 4 秒，假设值）。
- 串行：一个 Agent 依次做完；并行：K 个子 Agent 同时做，墙钟时间取最慢的那个。
- 汇总：子 Agent 回传全部过程（每步 TOK_STEP token） vs 只回传摘要（TOK_SUMMARY token）。
价格按 Claude Sonnet 5.5 官方价 $2/$10 每百万 token；其余均为假设参数。"""
import json, math, os, random
STEPS, TOK_STEP, TOK_SUMMARY, PIN, POUT = 6, 1500, 300, 2.0, 10.0

def sim(K, trials=5000, seed=0):
    rnd = random.Random(seed); ser = par = 0.0; slow = []
    mu = math.log(4.0) - 0.5 * 0.6 ** 2
    for _ in range(trials):
        per = [sum(rnd.lognormvariate(mu, 0.6) for _ in range(STEPS)) for _ in range(K)]
        ser += sum(per); par += max(per); slow.append(max(per))
    slow.sort()
    return dict(serial_s=ser / trials, parallel_s=par / trials, parallel_p95_s=slow[int(.95 * trials)])

def lead_tokens(K, full):
    return 2000 + K * (STEPS * TOK_STEP if full else TOK_SUMMARY)   # 主 Agent 汇总时的输入 token（2000 为任务说明，假设）

def single_cost(K):  # 单个 Agent 串行做完 K 个子任务，上下文一路累加不清空
    n = K * STEPS; return sum(1500 + i * TOK_STEP for i in range(n)) * PIN / 1e6 + n * 200 * POUT / 1e6

def cost(K, full):
    sub = K * sum(1500 + i * TOK_STEP for i in range(STEPS)) * PIN / 1e6 + K * STEPS * 200 * POUT / 1e6  # 子 Agent 各自累加上下文
    lead = lead_tokens(K, full) * PIN / 1e6 + 1500 * POUT / 1e6
    return sub, lead

if __name__ == "__main__":
    out = {}
    for K in [1, 3, 5, 10]:
        t = sim(K); s, lf = cost(K, True); _, ls = cost(K, False)
        out[K] = dict(**t, lead_tokens_full=lead_tokens(K, True), lead_tokens_summary=lead_tokens(K, False),
                      sub_cost=s, lead_cost_full=lf, lead_cost_summary=ls, single_cost=single_cost(K))
        print(f"K={K:2d} 串行 {t['serial_s']:.0f}s 并行 {t['parallel_s']:.0f}s (P95 {t['parallel_p95_s']:.0f}s) | 主Agent输入 全量 {lead_tokens(K, True)} / 摘要 {lead_tokens(K, False)} token "
              f"| 子Agent费 ${s:.3f} 主Agent费 全量 ${lf:.3f} / 摘要 ${ls:.3f} | 单Agent串行累加上下文 ${single_cost(K):.3f}")
    os.makedirs("outputs", exist_ok=True); json.dump(out, open("outputs/parallel_sim.json", "w"), indent=1)
