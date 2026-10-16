"""延伸实测：工具返回值精简前后，Agent 每一步的输入 token 差多少。
模拟一个真实风格的航班查询接口：返回 20 个航班，每个航班带 14 个字段（很多是模型做决定用不上的）。
精简版只保留 5 个字段、按价格取前 5 个。计数用 tiktoken o200k_base；没有则按字符数/1.5 粗估。"""
import json, random
try:
    import tiktoken; enc = tiktoken.get_encoding("o200k_base"); count = lambda s: len(enc.encode(s)); how = "tiktoken o200k_base"
except Exception:
    count = lambda s: int(len(s) / 1.5); how = "字符数/1.5 粗估"

def raw_flights(n=20, seed=0):
    r = random.Random(seed); air = ["HO", "MU", "CA", "CZ", "FM", "9C"]
    out = []
    for i in range(n):
        h = 6 + i * 0.8
        out.append({"flight_no": f"{r.choice(air)}{1000 + r.randint(0, 8999)}", "origin": "上海", "origin_airport": "PVG" if i % 2 else "SHA",
                    "origin_terminal": f"T{1 + i % 2}", "dest": "北京", "dest_airport": "PEK" if i % 3 else "PKX",
                    "dep_time": f"2026-10-20T{int(h):02d}:{int((h % 1) * 60):02d}:00+08:00", "arr_time": f"2026-10-20T{int(h) + 2:02d}:{int((h % 1) * 60):02d}:00+08:00",
                    "aircraft": r.choice(["A320", "A321neo", "B737-800", "B787-9", "C919"]), "cabin": "Y",
                    "price_cny": 600 + r.randint(0, 1400), "tax_cny": 50, "fuel_surcharge_cny": 0,
                    "fare_rules": "退改：起飞前7天以上收取10%手续费，起飞前7天内收取30%，起飞后不得退票。行李：免费托运20kg。",
                    "seats_left": r.randint(1, 9)})
    return out

def trimmed(fl, top=5):
    keep = sorted(fl, key=lambda f: f["price_cny"])[:top]
    return [{"flight_no": f["flight_no"], "dep": f["dep_time"][11:16], "arr": f["arr_time"][11:16], "price": f["price_cny"], "seats": f["seats_left"]} for f in keep]

if __name__ == "__main__":
    fl = raw_flights(); a = count(json.dumps(fl, ensure_ascii=False)); b = count(json.dumps(trimmed(fl), ensure_ascii=False))
    print(f"计数方式：{how}\n原始返回 20 个航班 × 14 字段：{a} token；精简后 5 个航班 × 5 字段：{b} token（{b / a:.1%}）")
    # 这条结果进入对话后，后面每一步都要重发。设查询后还有 S 步
    for S in [2, 4, 8]:
        print(f"查询之后还有 {S} 次模型调用：原始结果累计多付 {a * S} token，精简后 {b * S} token")
