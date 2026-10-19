"""AI 产品三种定价方式的单位经济模拟：按席位 / 按 token / 按结果。

用户群体和客户数据全部是程序按固定随机种子生成的【模拟数据】，不是任何真实公司的数据。
模型单价取自 Anthropic 官方价格表（2026-10-19 查询，美元 / 百万 token）：
  Claude Sonnet 5.5：输入 2，输出 10，缓存读取 0.10（0.05x），5 分钟缓存写入 2.50（1.25x）
  Claude Haiku 5.5（提示 ≤100K）：输入 0.10，输出 0.50，缓存读取 0.01，缓存写入 0.125
其余参数（请求量分布、每次请求的 token 数、缓存命中率、路由比例、解决率等）都是假设值，可改。
这里的"毛利率"只扣推理成本，真实毛利还要扣托管、客服、评测等成本。只依赖 numpy。
"""
import json, math, os
import numpy as np

PRICES = {  # 美元 / 百万 token
    "big":   {"in": 2.00, "out": 10.00, "cache_read": 0.10, "cache_write": 2.50},   # Sonnet 5.5
    "small": {"in": 0.10, "out": 0.50,  "cache_read": 0.01, "cache_write": 0.125},  # Haiku 5.5
}
CONFIGS = {  # 三种成本优化程度
    "base":        {"cache": False, "route": 0.0},
    "cache":       {"cache": True,  "route": 0.0},
    "cache_route": {"cache": True,  "route": 0.6},
}
CACHE_HIT = 0.85   # 可缓存前缀里，读缓存的比例（其余 15% 是写缓存），假设
MARKUP = 0.25      # 按 token 收费：在模型实际成本上加价 25%（假设）


def call_cost(tok_in, tok_out, cache_frac, model, cache):
    """一次模型调用的期望成本（美元）。cache_frac 为输入里可缓存前缀的占比。"""
    p = PRICES[model]
    if cache:
        cached = tok_in * cache_frac
        c_in = (tok_in - cached) * p["in"] + cached * (CACHE_HIT * p["cache_read"] + (1 - CACHE_HIT) * p["cache_write"])
    else:
        c_in = tok_in * p["in"]
    return (c_in + tok_out * p["out"]) / 1e6


def mixed_cost(tok_in, tok_out, cache_frac, cfg):
    """按路由比例把一部分调用发给小模型后的期望单次成本。"""
    big = call_cost(tok_in, tok_out, cache_frac, "big", cfg["cache"])
    small = call_cost(tok_in, tok_out, cache_frac, "small", cfg["cache"])
    return (1 - cfg["route"]) * big + cfg["route"] * small


# ---------------- A. 席位制 AI 助手（类似团队版聊天/编程助手） ----------------
def simulate_seats(seed=16, n=10_000, org_size=50):
    rng = np.random.default_rng(seed)
    idle = rng.random(n) < 0.15                                  # 15% 席位买了基本不用（假设）
    req = np.where(idle, rng.poisson(2, n), rng.lognormal(math.log(120), 1.5, n))
    req = np.clip(np.round(req), 0, 20_000)                     # 每席每月请求数，重尾
    tok_in = 6_000 * rng.lognormal(0, 0.6, n)                    # 每次请求平均输入 token，因人而异
    tok_out = 0.08 * tok_in
    org = np.arange(n) // org_size                               # 每 50 个席位一个客户组织
    return {"req": req, "tok_in": tok_in, "tok_out": tok_out, "org": org, "cache_frac": 0.6}


def seat_cost(pop, cfg):
    return pop["req"] * mixed_cost(pop["tok_in"], pop["tok_out"], pop["cache_frac"], cfg)


def with_agents(pop, frac, mult=8.0, seed=160):
    """让一部分活跃席位改用 Agent 模式：每次请求的 token 放大 mult 倍（假设值）。"""
    rng = np.random.default_rng(seed)
    active = pop["req"] > 10
    agent = active & (rng.random(len(active)) < frac / max(active.mean(), 1e-9))
    p = dict(pop)
    p["tok_in"] = np.where(agent, pop["tok_in"] * mult, pop["tok_in"])
    p["tok_out"] = np.where(agent, pop["tok_out"] * mult, pop["tok_out"])
    return p


def agent_sweep(pop, cfg, fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    rows = []
    for f in fracs:
        e = seat_economics(with_agents(pop, f), cfg)
        rows.append({"frac": f, "mean_cost": e["mean_cost"], "underwater": e["underwater"], **{"gm_" + k: v for k, v in e["gm"].items()}})
    return rows


def seat_economics(pop, cfg, seat_price=20.0, included=20.0):
    cost = seat_cost(pop, cfg)
    n = len(cost); s = np.sort(cost)[::-1]
    billed = cost * (1 + MARKUP)                                  # 按 token 计价的应收金额
    # 混合制：席位费含等额用量，组织内共享（池化），超出部分按 token 价另收
    org_bill = np.bincount(pop["org"], weights=billed)
    org_seats = np.bincount(pop["org"])
    overage_pool = np.maximum(0, org_bill - included * org_seats).sum()
    overage_each = np.maximum(0, billed - included).sum()            # 不池化：每个席位各算各的额度
    rev = {"seat": seat_price * n, "token": billed.sum(),
           "hybrid_pooled": seat_price * n + overage_pool, "hybrid_individual": seat_price * n + overage_each}
    gm = {k: 1 - cost.sum() / v for k, v in rev.items()}
    return {
        "mean_cost": cost.mean(), "median_cost": float(np.median(cost)), "p99_cost": float(np.quantile(cost, 0.99)),
        "share_top1": s[: n // 100].sum() / s.sum(), "share_top10": s[: n // 10].sum() / s.sum(),
        "underwater": float((cost > seat_price).mean()),
        "breakeven_seat_price": cost.mean(), "price_for_gm60": cost.mean() / 0.4,
        "gm": gm, "revenue": rev, "cost_total": cost.sum(),
    }


# ---------------- B. AI 客服 Agent（按会话 / 按解决） ----------------
SUP = {"calls_res": 5, "calls_unres": 8, "tok_in": 9_000, "cache_frac": 2 / 3, "tok_out": 500,
       "verify_in": 4_000, "verify_out": 50, "conv_per_human": 400, "outcome_price": 0.99}


def simulate_support(seed=16, n=2_000, mean_r=0.6, conc=10.0):
    rng = np.random.default_rng(seed)
    conv = np.clip(np.round(rng.lognormal(math.log(1_200), 1.3, n)), 50, None)   # 每月 AI 接待会话数
    r = rng.beta(conc * mean_r, conc * (1 - mean_r), n)                          # 各客户的 AI 解决率
    return {"conv": conv, "r": r}


def conv_costs(cfg):
    c = mixed_cost(SUP["tok_in"], SUP["tok_out"], SUP["cache_frac"], cfg)
    verify = call_cost(SUP["verify_in"], SUP["verify_out"], 0, "small", False)   # 判定"是否解决"的小模型调用
    return SUP["calls_res"] * c, SUP["calls_unres"] * c, verify


def support_economics(pop, cfg, seat_price):
    c_res, c_unres, v = conv_costs(cfg)
    conv, r = pop["conv"], pop["r"]
    ai_cost = conv * (r * c_res + (1 - r) * c_unres)
    seats = np.maximum(1, np.ceil(conv * (1 - r) / SUP["conv_per_human"]))      # 剩下的人工坐席
    out_cost = ai_cost + conv * v
    out_rev = SUP["outcome_price"] * conv * r
    rev = {"seat": (seat_price * seats).sum(), "token": (ai_cost * (1 + MARKUP)).sum(), "outcome": out_rev.sum()}
    cost = {"seat": ai_cost.sum(), "token": ai_cost.sum(), "outcome": out_cost.sum()}
    gm = {k: 1 - cost[k] / rev[k] for k in rev}
    r_star = (c_unres + v) / (SUP["outcome_price"] - c_res + c_unres)            # 按结果收费的保本解决率
    return {"gm": gm, "revenue": rev, "cost": cost, "seats": seats.sum(),
            "cost_per_conv": float(ai_cost.sum() / conv.sum()), "cost_per_resolution": float(out_cost.sum() / (conv * r).sum()),
            "breakeven_r": r_star, "acc_loss_outcome": float((out_rev < out_cost).mean()),
            "acc_loss_seat": float((seat_price * seats < ai_cost).mean())}


def calibrate_seat_price(cfg, mean_r=0.4, seed=16):
    """假设厂商在平均解决率 40% 时，把坐席价定到与按结果收费的总收入相同。"""
    pop = simulate_support(seed, mean_r=mean_r)
    seats = np.maximum(1, np.ceil(pop["conv"] * (1 - pop["r"]) / SUP["conv_per_human"]))
    return float(SUP["outcome_price"] * (pop["conv"] * pop["r"]).sum() / seats.sum())


def resolution_sweep(cfg, seat_price, rates=None, seed=16):
    rates = rates if rates is not None else [round(x, 2) for x in np.arange(0.3, 0.81, 0.05)]
    rows = []
    for m in rates:
        e = support_economics(simulate_support(seed, mean_r=m), cfg, seat_price)
        rows.append({"mean_r": m, "rev_seat": e["revenue"]["seat"], "rev_outcome": e["revenue"]["outcome"],
                     "rev_token": e["revenue"]["token"], "cost": e["cost"]["seat"], "cost_outcome": e["cost"]["outcome"],
                     "gm_seat": e["gm"]["seat"], "gm_outcome": e["gm"]["outcome"], "seats": e["seats"]})
    return rows


def run(seed=16):
    pop = simulate_seats(seed)
    A = {k: seat_economics(pop, c) for k, c in CONFIGS.items()}
    base_cost = np.sort(seat_cost(pop, CONFIGS["base"]))
    A["base_cost_quantiles"] = [float(np.quantile(base_cost, q)) for q in np.linspace(0, 1, 201)]
    A["base_cost_lorenz"] = (np.cumsum(base_cost) / base_cost.sum())[:: 50].tolist()
    A["agent_sweep"] = agent_sweep(pop, CONFIGS["cache"])
    sp = calibrate_seat_price(CONFIGS["base"], seed=seed)
    sup = simulate_support(seed)
    B = {k: support_economics(sup, c, sp) for k, c in CONFIGS.items()}
    B["seat_price"] = sp
    B["sweep"] = resolution_sweep(CONFIGS["base"], sp, seed=seed)
    return {"seats": A, "support": B}


def _fmt(x):
    return f"{x:.1%}"


def main():
    res = run()
    A, B = res["seats"], res["support"]
    print("== A. 席位制助手：1 万个模拟席位，席位价 $20/月 ==")
    for k in CONFIGS:
        a = A[k]
        print(f"[{k}] 每席月均成本 ${a['mean_cost']:.2f}，中位数 ${a['median_cost']:.2f}，P99 ${a['p99_cost']:.1f}；"
              f"前 1% 席位占成本 {_fmt(a['share_top1'])}，前 10% 占 {_fmt(a['share_top10'])}；成本超过 $20 的席位 {_fmt(a['underwater'])}")
        print(f"      毛利率（只算推理）：按席位 {_fmt(a['gm']['seat'])} / 按 token {_fmt(a['gm']['token'])} / "
              f"席位含额度·池化 {_fmt(a['gm']['hybrid_pooled'])} / 席位含额度·不池化 {_fmt(a['gm']['hybrid_individual'])}；"
              f"保本席位价 ${a['breakeven_seat_price']:.2f}，60% 毛利席位价 ${a['price_for_gm60']:.2f}")
    print("\nAgent 模式渗透率（已开缓存）  每席成本  成本超$20席位  按席位  池化混合  不池化混合  按token")
    for row in A["agent_sweep"]:
        print(f"  {row['frac']:.0%}  ${row['mean_cost']:6.2f}  {_fmt(row['underwater']):>6}  {_fmt(row['gm_seat']):>7}  "
              f"{_fmt(row['gm_hybrid_pooled']):>7}  {_fmt(row['gm_hybrid_individual']):>7}  {_fmt(row['gm_token']):>6}")
    print(f"\n== B. AI 客服 Agent：2,000 家模拟客户，平均解决率 60%，按结果 ${SUP['outcome_price']}/次，坐席价 ${B['seat_price']:.0f}/席 ==")
    for k in CONFIGS:
        b = B[k]
        print(f"[{k}] 每会话成本 ${b['cost_per_conv']:.3f}，每次解决成本 ${b['cost_per_resolution']:.3f}，保本解决率 {_fmt(b['breakeven_r'])}；"
              f"毛利率 坐席 {_fmt(b['gm']['seat'])} / token {_fmt(b['gm']['token'])} / 按结果 {_fmt(b['gm']['outcome'])}；"
              f"亏损客户占比 坐席 {_fmt(b['acc_loss_seat'])} / 按结果 {_fmt(b['acc_loss_outcome'])}")
    print("\n平均解决率  坐席收入   按结果收入  推理成本   坐席毛利  按结果毛利  坐席数")
    for row in B["sweep"]:
        print(f"  {row['mean_r']:.2f}   {row['rev_seat']/1e3:8.0f}k {row['rev_outcome']/1e3:8.0f}k {row['cost']/1e3:8.0f}k"
              f"   {_fmt(row['gm_seat']):>7}  {_fmt(row['gm_outcome']):>7}  {row['seats']:7.0f}")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ep16_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\n结果已写入 outputs/ep16_results.json")


if __name__ == "__main__":
    main()
