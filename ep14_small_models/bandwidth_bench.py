"""端侧推理为什么"看内存带宽"：解码每生成 1 个 token，都要把全部权重从内存读一遍。
本脚本实测本机：1) 内存拷贝带宽；2) 用 numpy 做矩阵×向量（解码的核心运算），测有效带宽；
3) 用公式 速度上限 ≈ 有效带宽 / 每 token 读取的权重字节数，估算不同大小模型的 token/s 上限。
默认占用约 600MB 内存，可用 --mb 调小。"""
import argparse, json, os, platform, time
import numpy as np

def copy_bw(mb):
    a = np.ones(mb * 1024 * 1024 // 8); b = np.empty_like(a); best = 1e9
    for _ in range(5):
        t = time.perf_counter(); np.copyto(b, a); best = min(best, time.perf_counter() - t)
    return 2 * a.nbytes / best / 1e9  # 读 + 写

def gemv_bw(mb, dtype):
    n = int((mb * 1024 * 1024 / np.dtype(dtype).itemsize) ** 0.5)
    W = np.random.default_rng(0).standard_normal((n, n)).astype(dtype); x = np.ones(n, dtype); best = 1e9
    for _ in range(5):
        t = time.perf_counter(); W @ x; best = min(best, time.perf_counter() - t)
    return W.nbytes / best / 1e9

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mb", type=int, default=256); a = ap.parse_args()
    cb = copy_bw(a.mb); g32 = gemv_bw(a.mb, np.float32)
    print(f"机器：{platform.machine()}，{os.cpu_count()} 个逻辑核")
    print(f"内存拷贝带宽 ≈ {cb:.1f} GB/s；fp32 矩阵×向量有效带宽 ≈ {g32:.1f} GB/s")
    models = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3, "7B": 7, "14B": 14}
    est = {}
    for name, pb in models.items():
        est[name] = {}
        for bits in [16, 8, 4.5]:
            gb = pb * bits / 8; est[name][f"{bits}bit"] = round(g32 / gb, 1)
        print(f"{name}: " + "  ".join(f"{k} 权重{models[name]*float(k[:-3])/8:.1f}GB→上限约 {v} tok/s" for k, v in est[name].items()))
    os.makedirs("outputs", exist_ok=True)
    json.dump({"copy_GBps": cb, "gemv_GBps": g32, "est_tok_s": est}, open("outputs/results.json", "w"), indent=1)
