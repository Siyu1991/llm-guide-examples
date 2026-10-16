from shapley_vs_ig import main
o = main()
for name, v in o["bases"].items():
    target = o["p"] - v["p_base"]
    assert abs(v["shap_sum"] - target) < 1e-6, name   # Shapley 满足效率性
    assert abs(v["ig_sum"] - target) < 1e-3, name     # 积分梯度满足完备性
print("ok")
