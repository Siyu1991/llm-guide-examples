# ep02 MoE：最小混合专家层

对应专栏第 2 期《MoE》。只依赖 numpy，CPU 运行，峰值内存约 0.5GB。

```bash
pip install -r requirements.txt
python moe_toy.py --quick   # 小尺寸自检，1 秒内
python moe_toy.py           # 正式实测，约 20 秒
```

输出三部分：
1. MoE（16 专家选 2）与"同总参数""同激活参数"两种稠密 FFN 的参数量与耗时；
2. 路由不均衡与 DeepSeek-V3 式偏置均衡前后的专家负载；
3. 按 Muse Spark Open 公开参数（816B 总参数 / 42B 激活、1.2TB/s 带宽）估算不同精度下的权重内存和单流解码速度上限（理论值）。

结果写入 `result.json`。
