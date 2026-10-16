"""用官方 Python SDK（mcp>=2.3，协议 2026-07-28）写的 MCP 服务器：暴露订单查询工具和一份退货政策资源。
运行方式由客户端以子进程启动（stdio），不需要手动运行。"""
from mcp.server.mcpserver import MCPServer

ORDERS = {"A1001": "已发货，顺丰，预计10月18日送达", "A1002": "待发货，预计10月19日发出"}
app = MCPServer("shop-demo")

@app.tool()
def query_order(order_id: str) -> str:
    """按订单号查询物流状态，例如 A1001"""
    return ORDERS.get(order_id, f"订单 {order_id} 不存在")

@app.tool()
def refund_days(pay_method: str) -> str:
    """查询退款到账时间。pay_method 取 支付宝/微信/银行卡"""
    return "7个工作日" if pay_method == "银行卡" else "1到3个工作日"

@app.resource("policy://refund")
def refund_policy() -> str:
    """退货政策全文"""
    return "签收7天内可无理由退货，生鲜与定制商品除外。"

if __name__ == "__main__":
    app.run("stdio")
