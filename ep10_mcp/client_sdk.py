"""用官方 SDK 客户端以 stdio 启动 server_sdk.py，列出工具并调用。"""
import asyncio, sys
from mcp import Client, StdioServerParameters

async def main():
    params = StdioServerParameters(command=sys.executable, args=["server_sdk.py"])
    async with Client(params) as c:
        tools = await c.list_tools()
        for t in tools.tools: print("工具：", t.name, "|", t.description, "| 参数：", list(t.input_schema["properties"]))
        r = await c.call_tool("query_order", {"order_id": "A1001"})
        print("调用 query_order(A1001) →", r.content[0].text)
        r = await c.call_tool("refund_days", {"pay_method": "银行卡"})
        print("调用 refund_days(银行卡) →", r.content[0].text)
        res = await c.read_resource("policy://refund")
        print("读取资源 policy://refund →", res.contents[0].text)
        return [t.name for t in tools.tools]

if __name__ == "__main__":
    names = asyncio.run(main()); assert set(names) == {"query_order", "refund_days"}; print("SDK 客户端测试通过")
