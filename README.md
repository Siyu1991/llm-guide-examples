# llm-guide-examples

知乎专栏「代码与模型之间」的配套示例代码。每期一个目录，均可在普通笔记本 CPU 上运行（内存 < 2GB，无需 GPU）。需要调用大模型 API 的示例从环境变量读取密钥，没有密钥时自动使用离线模拟。

许可证：MIT

## 目录

### 第 8-15 期

| 目录 | 期数 | 内容 | 依赖 |
| --- | --- | --- | --- |
| `ep08_rag_eval/` | 08 RAG 效果评估 | 检索指标、答对率/忠实度、失败归因、bootstrap 置信区间 | 标准库 |
| `ep09_agent/` | 09 工具调用与最小 Agent | 查单/订票/报表 Agent 循环，确认机制与死循环兜底 | 标准库（可选 tiktoken） |
| `ep10_mcp/` | 10 MCP 协议 | 零依赖 JSON-RPC 实现 + 官方 SDK 服务器/客户端，按 2026-07-28 版规范 | 标准库 / mcp |
| `ep11_multi_agent/` | 11 多 Agent 协作 | 单 Agent、单 Agent+自动测试、多 Agent 三种方式的通过率与调用次数 | 标准库 |
| `ep12_agent_reliability/` | 12 Agent 的可靠性与成本 | 权限/确认/幂等/重试/预算护栏，成功率与成本模拟 | 标准库 |
| `ep13_quantization/` | 13 量化与本地部署 | numpy 量化误差实验；GGUF 四种量化实测 | numpy / llama-cpp-python |
| `ep14_small_models/` | 14 小模型与端侧 | 内存带宽实测与速度上限估算；线程数与预填充实测 | numpy / llama-cpp-python |
| `ep15_interpretability/` | 15 可解释性入门 | 系数、积分梯度、置换重要性、线性探针 | numpy |
