"""零依赖客户端：启动 server_raw.py，逐条打印收发的 JSON-RPC 原文。"""
import json, subprocess, sys
p = subprocess.Popen([sys.executable, "server_raw.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def rpc(i, method, params=None):
    req = {"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}
    print("→", json.dumps(req, ensure_ascii=False)); p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n"); p.stdin.flush()
    resp = json.loads(p.stdout.readline()); print("←", json.dumps(resp, ensure_ascii=False)); return resp
meta = {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}}
assert "2026-07-28" in rpc(1, "server/discover", meta)["result"]["supportedVersions"]
assert rpc(2, "tools/list", meta)["result"]["tools"][0]["name"] == "query_order"
assert "顺丰" in rpc(3, "tools/call", {**meta, "name": "query_order", "arguments": {"order_id": "A1001"}})["result"]["content"][0]["text"]
assert rpc(4, "tools/call", {**meta, "name": "query_order", "arguments": {"order_id": "B2"}})["result"]["isError"]
assert rpc(5, "resources/list", meta)["error"]["code"] == -32601
p.stdin.close(); p.wait(); print("零依赖客户端测试通过")
