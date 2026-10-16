"""不依赖任何第三方库的 MCP 服务器（stdio，JSON-RPC 2.0，每行一条消息）。
用来看清线上到底传了什么。同时兼容 2026-07-28（无握手，server/discover）和 2025-11-25（initialize 握手）两种客户端。
只实现了 tools 的最小子集，生产环境请用官方 SDK。"""
import json, sys
VERSIONS = ["2026-07-28", "2025-11-25"]
TOOLS = [{"name": "query_order", "description": "按订单号查询物流状态",
          "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}]
ORDERS = {"A1001": "已发货，顺丰，预计10月18日送达"}
INFO = {"name": "shop-raw", "version": "0.1.0"}

def handle(msg):
    m, p = msg.get("method"), msg.get("params") or {}
    if m == "server/discover":
        return {"supportedVersions": VERSIONS, "capabilities": {"tools": {}}, "serverInfo": INFO, "resultType": "complete"}
    if m == "initialize":  # 旧版握手
        return {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": INFO}
    if m == "tools/list":
        return {"tools": TOOLS, "ttlMs": 60000, "cacheScope": "public", "resultType": "complete"}
    if m == "tools/call":
        if p.get("name") != "query_order": raise ValueError("unknown tool")
        oid = p.get("arguments", {}).get("order_id", "")
        text = ORDERS.get(oid, f"订单 {oid} 不存在")
        return {"content": [{"type": "text", "text": text}], "isError": oid not in ORDERS, "resultType": "complete"}
    raise LookupError(m)

for line in sys.stdin:
    if not line.strip(): continue
    sys.stderr.write("[server 收到] " + line)  # stdio 模式下日志只能写 stderr，写 stdout 会破坏协议
    msg = json.loads(line)
    if "id" not in msg: continue  # 通知（如 notifications/initialized）不需要回复
    try: out = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
    except LookupError as e: out = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"Method not found: {e}"}}
    except ValueError as e: out = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32602, "message": str(e)}}
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n"); sys.stdout.flush()
