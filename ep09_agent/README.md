# ep09 工具调用与最小 Agent

对应专栏第 9 期。约 150 行标准库代码实现一个能查订单、查航班、订票（带用户确认）、出汇总报表的 Agent 循环。

```bash
python3 agent.py            # 6 个演示任务
python3 test_agent.py       # 含"死循环兜底""未知工具""坏 JSON 参数"测试
python3 measure_tokens.py   # 估算每一步的输入 token（可选 pip install tiktoken）
```

- 有 `OPENAI_API_KEY` 时走 OpenAI 兼容的 `chat/completions` + `tools` 接口（可用 `OPENAI_BASE_URL`、`OPENAI_MODEL` 指向其他兼容服务）。
- 没有密钥时用 `MockLLM`：它用规则生成与真实接口相同格式的 `tool_calls`，只用于演示循环结构，不代表模型能力。
