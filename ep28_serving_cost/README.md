# ep28 KV 显存与自建/API 保本计算

按层数×KV 头×head_dim 估算 KV cache 占用；用 Runpod Community Cloud 公开小时价与 gpt-6.1-sol 官方 API 价估算保本输出 token/小时。

价格快照 2026-10-30。吞吐数字为脚本假设，不是本机压测。

```bash
python3 calc.py
python3 -m unittest test_calc -v
```
