"""ep17 开源模型公司的成本账（作者模拟，只依赖 numpy）。

A. 推理托管的单位经济：用 DeepSeek 2025-02 公布的一天线上数据做标定
   （开源周 Day 6《DeepSeek-V3/R1 Inference System Overview》），
   复算日成本、理论收入、成本利润率，并给出"付费流量占比"敏感性。
B. 放出权重后训练成本怎么收回：给定训练 + 研究总投入、每百万 token 推理毛利、
   模型商业寿命，算需要多少日付费 token 才能回本。除标定数据外，其余都是假设参数。
C. 买方视角：自建（按 GPU 小时租卡）和按 token 调 API 的保本利用率。

所有"假设值"都写在本文件顶部，可以直接改。
"""
import json
import os

import numpy as np

# ---------- A. 标定数据（DeepSeek 官方公布，2025-02-27 12:00 至 02-28 12:00，UTC+8）----------
DS_DAY = {
    "avg_nodes": 226.75,          # 平均占用节点数
    "peak_nodes": 278,            # 峰值节点数
    "gpus_per_node": 8,           # 每节点 8 张 H800
    "gpu_hour": 2.0,              # DeepSeek 假设的 H800 租金，美元/卡·时
    "in_tokens": 608e9,           # 总输入 token
    "in_cache_hit": 342e9,        # 其中命中磁盘 KV 缓存
    "out_tokens": 168e9,          # 总输出 token
    "prefill_tps_node": 73.7e3,   # 每节点预填充吞吐（含缓存命中），token/s
    "decode_tps_node": 14.8e3,    # 每节点解码吞吐，token/s
    "reported_cost": 87072.0,     # 官方报告的日成本
    "reported_revenue": 562027.0, # 官方报告的理论日收入（全部按 R1 价计）
    "reported_margin": 5.45,      # 官方报告的成本利润率 545%
}
R1_PRICE = {"cache_hit": 0.14, "cache_miss": 0.55, "output": 2.19}   # 美元/百万 token（R1 当时标价）

# ---------- B. 回本模拟的假设参数 ----------
SPEND_GRID_M = np.logspace(np.log10(5), np.log10(2000), 60)   # 训练 + 研究总投入，百万美元（假设区间）
LIFE_MONTHS = [3, 6, 12]                                      # 一代模型的商业寿命（假设）
MARGINS = [0.1, 0.3, 0.612, 1.0]                              # 每百万 token 推理毛利，美元（0.612 为 A 部分标定值）
RND_MULTIPLIER = 3.0                                          # 研究与试验投入 = 正式训练的倍数（假设）
V3_FINAL_RUN_M = 5.576                                        # DeepSeek-V3 正式训练成本（官方，$2/卡·时口径）


def daily_gpu_cost(nodes, gpus_per_node=8, gpu_hour=2.0, hours=24.0):
    return nodes * gpus_per_node * gpu_hour * hours


def theoretical_revenue(in_tokens, in_cache_hit, out_tokens, price=R1_PRICE):
    miss = in_tokens - in_cache_hit
    parts = {
        "cache_hit": in_cache_hit / 1e6 * price["cache_hit"],
        "cache_miss": miss / 1e6 * price["cache_miss"],
        "output": out_tokens / 1e6 * price["output"],
    }
    parts["total"] = sum(parts.values())
    return parts


def node_hours_from_throughput(in_tokens, out_tokens, prefill_tps, decode_tps):
    """按每节点吞吐反推需要的节点·时（预填充与解码分开部署）。"""
    pre = in_tokens / prefill_tps / 3600
    dec = out_tokens / decode_tps / 3600
    return {"prefill": pre, "decode": dec, "total": pre + dec}


def unit_costs(gpu_hour, gpus_per_node, prefill_tps, decode_tps):
    """每百万输入 / 输出 token 的 GPU 成本（满负荷）。"""
    node_hour = gpu_hour * gpus_per_node
    return {
        "per_m_input": node_hour / (prefill_tps * 3600 / 1e6),
        "per_m_output": node_hour / (decode_tps * 3600 / 1e6),
    }


def cost_profit_margin(revenue, cost):
    """成本利润率 = (收入 - 成本) / 成本，DeepSeek 报告的 545% 用的是这个口径。"""
    return (revenue - cost) / cost


def margin_vs_paid_share(revenue_full, cost, shares):
    shares = np.asarray(shares, dtype=float)
    return (shares * revenue_full - cost) / cost


def breakeven_paid_share(revenue_full, cost):
    return cost / revenue_full


def calibrate(d=DS_DAY, price=R1_PRICE):
    cost = daily_gpu_cost(d["avg_nodes"], d["gpus_per_node"], d["gpu_hour"])
    cost_peak = daily_gpu_cost(d["peak_nodes"], d["gpus_per_node"], d["gpu_hour"])
    rev = theoretical_revenue(d["in_tokens"], d["in_cache_hit"], d["out_tokens"], price)
    nh = node_hours_from_throughput(d["in_tokens"], d["out_tokens"], d["prefill_tps_node"], d["decode_tps_node"])
    uc = unit_costs(d["gpu_hour"], d["gpus_per_node"], d["prefill_tps_node"], d["decode_tps_node"])
    traffic = d["in_tokens"] + d["out_tokens"]
    return {
        "daily_cost": cost,
        "daily_cost_if_peak_all_day": cost_peak,
        "revenue": rev,
        "cost_profit_margin": cost_profit_margin(rev["total"], cost),
        "cost_profit_margin_reported_revenue": cost_profit_margin(d["reported_revenue"], cost),
        "node_hours_from_throughput": nh,
        "node_hours_actual": d["avg_nodes"] * 24,
        "unit_cost": uc,
        "blended_price_per_m": rev["total"] / (traffic / 1e6),
        "blended_cost_per_m": cost / (traffic / 1e6),
        "blended_margin_per_m": (rev["total"] - cost) / (traffic / 1e6),
        "breakeven_paid_share": breakeven_paid_share(d["reported_revenue"], cost),
        "cache_hit_ratio": d["in_cache_hit"] / d["in_tokens"],
        "output_share": d["out_tokens"] / traffic,
    }


# ---------- B. 回本 ----------
def required_daily_tokens(total_cost, margin_per_m, life_days):
    """在 life_days 内收回 total_cost 所需的日付费 token 数。"""
    return total_cost / (margin_per_m * life_days) * 1e6


def payback_days(total_cost, daily_tokens, margin_per_m):
    return total_cost / (daily_tokens / 1e6 * margin_per_m)


def payback_table(margin_per_m, spends_m=(5.576, 16.728, 50, 100, 300, 1000), life_months=(3, 6, 12)):
    rows = []
    for s in spends_m:
        row = {"spend_m": s}
        for L in life_months:
            row[f"tokens_b_per_day_{L}m"] = required_daily_tokens(s * 1e6, margin_per_m, L * 30) / 1e9
        rows.append(row)
    return rows


# ---------- C. 买方：自建 vs API ----------
def selfhost_cost_per_m_traffic(gpu_hour, gpus_per_node, prefill_tps, decode_tps, out_share, util):
    """每百万"流量 token"（输入+输出）的自建成本；util 为平均利用率（0~1）。"""
    uc = unit_costs(gpu_hour, gpus_per_node, prefill_tps, decode_tps)
    full = (1 - out_share) * uc["per_m_input"] + out_share * uc["per_m_output"]
    return full / util


def breakeven_utilization(api_price_per_m_traffic, gpu_hour, gpus_per_node, prefill_tps, decode_tps, out_share):
    full = selfhost_cost_per_m_traffic(gpu_hour, gpus_per_node, prefill_tps, decode_tps, out_share, 1.0)
    return full / api_price_per_m_traffic


def run(out_dir="outputs"):
    cal = calibrate()
    m = cal["blended_margin_per_m"]
    res = {
        "calibration": cal,
        "payback_table_margin_calibrated": payback_table(m),
        "v3_points": {
            "final_run_only_m": V3_FINAL_RUN_M,
            "with_rnd_multiplier_m": V3_FINAL_RUN_M * RND_MULTIPLIER,
            "payback_days_at_200B_per_day": payback_days(V3_FINAL_RUN_M * RND_MULTIPLIER * 1e6, 200e9, m),
        },
        "selfhost_breakeven_util": {
            str(g): breakeven_utilization(cal["blended_price_per_m"], g, 8, DS_DAY["prefill_tps_node"],
                                          DS_DAY["decode_tps_node"], cal["output_share"])
            for g in (1.99, 2.0, 3.19, 3.99)
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ep17_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    return res


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    r = run(os.path.join(here, "outputs"))
    c = r["calibration"]
    print(f"日 GPU 成本 ${c['daily_cost']:,.0f}（若整天按峰值节点 ${c['daily_cost_if_peak_all_day']:,.0f}）")
    print(f"理论日收入 ${c['revenue']['total']:,.0f}（缓存命中 {c['revenue']['cache_hit']:,.0f} / 未命中 {c['revenue']['cache_miss']:,.0f} / 输出 {c['revenue']['output']:,.0f}）")
    print(f"成本利润率 {c['cost_profit_margin']:.1%}（按官方收入 {c['cost_profit_margin_reported_revenue']:.1%}）")
    nh = c["node_hours_from_throughput"]
    print(f"按吞吐反推节点·时 {nh['total']:,.0f}（预填充 {nh['prefill']:,.0f} + 解码 {nh['decode']:,.0f}），实际 {c['node_hours_actual']:,.0f}")
    print(f"单位成本：输入 ${c['unit_cost']['per_m_input']:.4f}/M，输出 ${c['unit_cost']['per_m_output']:.4f}/M")
    print(f"混合单价 ${c['blended_price_per_m']:.3f}/M，混合成本 ${c['blended_cost_per_m']:.3f}/M，毛利 ${c['blended_margin_per_m']:.3f}/M")
    print(f"保本付费占比 {c['breakeven_paid_share']:.1%}")
    print("回本所需日付费 token（十亿/天），毛利按标定值：")
    for row in r["payback_table_margin_calibrated"]:
        print("  ", {k: round(v, 1) for k, v in row.items()})
    print("V3 点：", {k: round(v, 2) for k, v in r["v3_points"].items()})
    print("自建保本利用率：", {k: f"{v:.1%}" for k, v in r["selfhost_breakeven_util"].items()})
