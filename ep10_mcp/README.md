# ep10 MCP 协议

对应专栏第 10 期。两套实现，按 MCP 规范 2026-07-28 版（无握手、`server/discover`、每个请求在 `_meta` 里带协议版本）：

| 文件 | 说明 | 依赖 |
| --- | --- | --- |
| `server_raw.py` / `client_raw.py` | 零依赖实现，逐条打印 JSON-RPC 原文，看清线上传了什么 | 无 |
| `server_sdk.py` / `client_sdk.py` | 官方 Python SDK（`MCPServer`），含 2 个工具和 1 个资源 | `mcp>=2.3` |
| `sdk_to_raw.py` | 互通测试：官方 SDK 客户端连零依赖服务器 | `mcp>=2.3` |

```bash
python3 client_raw.py                      # 离线、零依赖
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python client_sdk.py
.venv/bin/python sdk_to_raw.py
```

`server_raw.py` 只实现了 tools 的最小子集，用于教学；生产环境请使用官方 SDK。

## 延伸实验（第二版文章新增）

```bash
python3 tool_audit.py   # 工具数量与固定前缀 token；第三方工具描述的简单审查
```
