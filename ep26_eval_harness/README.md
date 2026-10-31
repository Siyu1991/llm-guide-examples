# ep26 回归评测脚手架

固定黄金集，对比两个 mock 模型版本的 contains 命中率，并用一个带「偏爱长文」偏见的 mock judge 作对照。换模型门禁只看 pass_rate 是否下降。

```bash
python3 harness.py
python3 -m unittest test_harness -v
```
