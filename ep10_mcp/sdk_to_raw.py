"""互通测试：官方 SDK 客户端 → 零依赖服务器。服务器在 stderr 打印收到的原始消息。"""
import asyncio, sys
from mcp import Client, StdioServerParameters
async def main():
    async with Client(StdioServerParameters(command=sys.executable, args=["server_raw.py"])) as c:
        r = await c.call_tool("query_order", {"order_id": "A1001"}); print("SDK 拿到：", r.content[0].text)
asyncio.run(main())
