# ep03 长上下文：三笔账

对应专栏第 3 期《长上下文》。只依赖 numpy。

```bash
pip install -r requirements.txt
python long_context.py --quick
python long_context.py        # 约 15 秒
```

- 实测单层 8 头因果注意力的 prefill 耗时随长度增长（单线程，取 3 次最小值）；
- 计算 Llama 2 7B（MHA）、Llama 3 70B（GQA）、DeepSeek-V3（MLA）每 token 的 KV 缓存；
- 计算合同审阅场景六种方案的单份成本（价格为 2026-10-16 官方页面数字，使用前请核对）。
