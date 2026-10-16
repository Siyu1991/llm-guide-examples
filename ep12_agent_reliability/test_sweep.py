from sweep import run
a, b = run(10, 0.95, 0.9, trials=5000), run(10, 0.99, 0.9, trials=5000)
assert b["success"] > a["success"] and b["total_per_task"] < a["total_per_task"]
print("ok")
