import numpy as np

import open_model_econ as om


def test_calibration_reproduces_deepseek_day():
    c = om.calibrate()
    d = om.DS_DAY
    assert abs(c["daily_cost"] - d["reported_cost"]) < 1e-6                       # 226.75 节点 × 8 卡 × 24 时 × $2
    assert abs(c["revenue"]["total"] / d["reported_revenue"] - 1) < 5e-4         # 公布的 token 数是取整后的，误差 < 0.05%
    assert abs(c["cost_profit_margin_reported_revenue"] - d["reported_margin"]) < 0.01   # 545%
    nh = c["node_hours_from_throughput"]["total"]
    assert abs(nh / c["node_hours_actual"] - 1) < 2e-3                           # 吞吐反推的节点·时与实际一致
    assert c["daily_cost_if_peak_all_day"] > c["daily_cost"]


def test_breakeven_share():
    d = om.DS_DAY
    s = om.breakeven_paid_share(d["reported_revenue"], d["reported_cost"])
    m = om.margin_vs_paid_share(d["reported_revenue"], d["reported_cost"], [s, 1.0])
    assert abs(m[0]) < 1e-12 and abs(m[1] - (d["reported_revenue"] / d["reported_cost"] - 1)) < 1e-12
    assert 0.15 < s < 0.16


def test_payback_identity():
    for spend in (5.576e6, 1e8):
        for life in (90, 180, 360):
            q = om.required_daily_tokens(spend, 0.612, life)
            assert abs(q / 1e6 * 0.612 * life - spend) < 1e-3
            assert abs(om.payback_days(spend, q, 0.612) - life) < 1e-9
    t = om.payback_table(0.612)
    assert all(r["tokens_b_per_day_3m"] > r["tokens_b_per_day_6m"] > r["tokens_b_per_day_12m"] for r in t)


def test_selfhost_breakeven():
    c = om.calibrate()
    d = om.DS_DAY
    args = (8, d["prefill_tps_node"], d["decode_tps_node"], c["output_share"])
    u = om.breakeven_utilization(c["blended_price_per_m"], 3.99, *args)
    cost_at_u = om.selfhost_cost_per_m_traffic(3.99, *args, util=u)
    assert abs(cost_at_u - c["blended_price_per_m"]) < 1e-9
    us = [om.breakeven_utilization(c["blended_price_per_m"], g, *args) for g in (1.0, 2.0, 4.0)]
    assert us[0] < us[1] < us[2]
    # 满负荷、$2/卡·时 时，自建混合成本就是 DeepSeek 那一天的混合成本
    full = om.selfhost_cost_per_m_traffic(2.0, *args, util=1.0)
    assert abs(full / c["blended_cost_per_m"] - 1) < 2e-3


def test_run_writes_json(tmp_path=None):
    import json, os, tempfile
    out = tempfile.mkdtemp()
    r = om.run(out)
    with open(os.path.join(out, "ep17_results.json"), encoding="utf-8") as f:
        j = json.load(f)
    assert abs(j["calibration"]["daily_cost"] - 87072) < 1e-6
    assert np.isfinite(r["v3_points"]["payback_days_at_200B_per_day"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all tests passed")
