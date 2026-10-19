import numpy as np
import pricing_sim as ps


def test_seats():
    pop = ps.simulate_seats(16)
    E = {k: ps.seat_economics(pop, c) for k, c in ps.CONFIGS.items()}
    for e in E.values():   # 按 token 加价 25%，毛利率恒为 0.25/1.25 = 20%
        assert abs(e["gm"]["token"] - ps.MARKUP / (1 + ps.MARKUP)) < 1e-9
        assert abs(1 - e["cost_total"] / (e["breakeven_seat_price"] * len(pop["req"]))) < 1e-9   # 保本价下毛利为 0
        assert e["gm"]["hybrid_individual"] >= e["gm"]["hybrid_pooled"] >= e["gm"]["seat"] - 1e-12   # 池化让利
    assert E["base"]["mean_cost"] > E["cache"]["mean_cost"] > E["cache_route"]["mean_cost"]   # 缓存、路由都降成本
    assert E["base"]["share_top10"] > 0.5        # 重尾：前 10% 席位吃掉一半以上成本
    sw = ps.agent_sweep(pop, ps.CONFIGS["cache"])
    assert all(a["gm_seat"] > b["gm_seat"] for a, b in zip(sw, sw[1:]))   # Agent 越多，席位制毛利越低


def test_support():
    cfg = ps.CONFIGS["base"]
    sp = ps.calibrate_seat_price(cfg)
    e40 = ps.support_economics(ps.simulate_support(16, mean_r=0.4), cfg, sp)
    assert abs(e40["revenue"]["seat"] / e40["revenue"]["outcome"] - 1) < 1e-9   # 校准点收入相等
    c_res, c_unres, v = ps.conv_costs(cfg); r = e40["breakeven_r"]
    assert abs(ps.SUP["outcome_price"] * r - (r * c_res + (1 - r) * c_unres + v)) < 1e-12   # 保本解决率
    sw = ps.resolution_sweep(cfg, sp)
    assert all(a["rev_seat"] > b["rev_seat"] for a, b in zip(sw, sw[1:]))        # 解决率越高，坐席收入越少
    assert all(a["rev_outcome"] < b["rev_outcome"] for a, b in zip(sw, sw[1:]))  # 按结果收入越多


def test_reproducible():
    a, b = ps.run(16), ps.run(16)
    assert a["seats"]["base"]["mean_cost"] == b["seats"]["base"]["mean_cost"]
    assert np.isclose(a["support"]["seat_price"], b["support"]["seat_price"])


if __name__ == "__main__":
    test_seats(); test_support(); test_reproducible(); print("ep16 tests passed")
