from gptq_mini import main
r = main()
assert all(r[b]["gptq"] < r[b]["rtn"] for b in r), r
print("ok")
