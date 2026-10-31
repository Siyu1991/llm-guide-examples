# ep23 提示缓存成本模拟

多轮 Agent 会话下，对比 OpenAI / Anthropic / Gemini / DeepSeek 在有无提示缓存时的输入费用。

价格取自 2026-10-30 各厂商官方文档（见脚本顶部注释）。不调用 API，不需要密钥。

## 运行

```bash
python3 cache_cost.py
```

结果写入 `outputs/ep23_results.json`。

## 场景参数

* 稳定前缀（系统提示 + 工具定义）：8000 token
* 每轮新增输入：1200 token
* 每轮输出：400 token
* 共 12 轮；前缀按精确匹配累积写入/命中
