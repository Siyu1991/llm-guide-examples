# llm-guide-examples

知乎专栏「代码与模型之间」的配套示例代码。每期一个目录，均可在普通笔记本 CPU 上运行（内存 < 2GB，无需 GPU）。需要调用大模型 API 的示例从环境变量读取密钥，没有密钥时自动使用离线模拟。

许可证：MIT

## 目录

### 第 2-7 期

| 目录 | 对应专栏 | 内容 | 依赖 |
| --- | --- | --- | --- |
| `ep02_moe/` | 第 2 期 MoE | numpy 实现的最小 MoE 层，实测与稠密 FFN 的耗时，路由偏置均衡 | numpy |
| `ep03_long_context/` | 第 3 期 长上下文 | 注意力 prefill 耗时随长度增长实测、KV 缓存计算、合同审阅成本 | numpy |
| `ep04_reasoning/` | 第 4 期 推理模型 | 思考 token 成本、自洽投票正确率、推理强度探针（API 或离线 mock） | 标准库 |
| `ep05_embedding/` | 第 5 期 Embedding | BM25 与 bge-small-zh 向量检索对比、暴力检索延迟 | fastembed、jieba |
| `ep06_rag/` | 第 6 期 RAG | 最小 RAG：切块、向量检索、带出处的生成（API 或离线抽取） | fastembed |
| `ep07_rag_advanced/` | 第 7 期 RAG 进阶 | 切块方式、混合检索 RRF、交叉编码器重排、元数据过滤 | fastembed、jieba |

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

### 第 16 期起（辅线·商业化）

| 目录 | 期数 | 内容 | 依赖 |
| --- | --- | --- | --- |
| `ep16_pricing/` | 16 AI 产品怎么定价 | 按席位 / 按 token / 按结果的单位经济模拟：重尾用量、Agent 渗透、保本解决率 | numpy |
| `ep17_open_model_econ/` | 17 开源公司怎么赚钱 | 用 DeepSeek 公开的一天推理数据复算成本利润率；保本付费占比、训练回本所需日付费 token、自建保本利用率 | numpy |
