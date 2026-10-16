"""工具与模拟业务数据（全部虚构）。每个工具 = 函数 + JSON Schema 描述，描述会原样发给模型。"""
import json
ORDERS = {"A1001": {"status": "已发货", "carrier": "顺丰", "eta": "10月18日", "amount": 268.0},
          "A1002": {"status": "待发货", "carrier": None, "eta": "10月19日", "amount": 99.0},
          "A1003": {"status": "已签收", "carrier": "中通", "eta": None, "amount": 1450.5}}
FLIGHTS = [{"no": "MU5101", "from": "上海", "to": "北京", "date": "2026-10-20", "dep": "08:00", "price": 1180},
           {"no": "CA1858", "from": "上海", "to": "北京", "date": "2026-10-20", "dep": "13:30", "price": 860},
           {"no": "HO1253", "from": "上海", "to": "北京", "date": "2026-10-20", "dep": "19:45", "price": 720}]
BOOKINGS = []

def query_order(order_id: str):
    o = ORDERS.get(order_id)
    return {"error": f"订单 {order_id} 不存在"} if o is None else {"order_id": order_id, **o}

def search_flights(origin: str, dest: str, date: str):
    return [f for f in FLIGHTS if f["from"] == origin and f["to"] == dest and f["date"] == date]

def book_flight(flight_no: str, passenger: str, confirmed: bool = False):
    if not confirmed:  # 有副作用的操作：必须带上用户确认
        return {"need_confirmation": True, "msg": f"将为 {passenger} 预订 {flight_no}，请用户确认"}
    BOOKINGS.append({"flight_no": flight_no, "passenger": passenger})
    return {"ok": True, "booking_id": f"BK{len(BOOKINGS):04d}"}

def sales_report(order_ids: list):
    rows = [ORDERS[i] for i in order_ids if i in ORDERS]
    return {"count": len(rows), "total": round(sum(r["amount"] for r in rows), 2)}

REGISTRY = {"query_order": query_order, "search_flights": search_flights,
            "book_flight": book_flight, "sales_report": sales_report}
SCHEMAS = [
 {"type": "function", "function": {"name": "query_order", "description": "按订单号查询物流状态和金额",
   "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "如 A1001"}}, "required": ["order_id"]}}},
 {"type": "function", "function": {"name": "search_flights", "description": "查询某日两城之间的航班",
   "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "dest": {"type": "string"},
     "date": {"type": "string", "description": "YYYY-MM-DD"}}, "required": ["origin", "dest", "date"]}}},
 {"type": "function", "function": {"name": "book_flight", "description": "预订航班。必须先向用户确认，确认后 confirmed=true",
   "parameters": {"type": "object", "properties": {"flight_no": {"type": "string"}, "passenger": {"type": "string"},
     "confirmed": {"type": "boolean"}}, "required": ["flight_no", "passenger"]}}},
 {"type": "function", "function": {"name": "sales_report", "description": "汇总若干订单的数量和总金额",
   "parameters": {"type": "object", "properties": {"order_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["order_ids"]}}},
]

def call_tool(name, args_json):
    if name not in REGISTRY: return {"error": f"未知工具 {name}"}
    try: args = json.loads(args_json) if isinstance(args_json, str) else args_json
    except json.JSONDecodeError as e: return {"error": f"参数不是合法 JSON: {e}"}
    try: return REGISTRY[name](**args)
    except TypeError as e: return {"error": f"参数错误: {e}"}
