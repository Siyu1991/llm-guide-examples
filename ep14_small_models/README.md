# ep14 小模型与端侧

对应专栏第 14 期。

```bash
python3 bandwidth_bench.py              # 只需 numpy：实测内存带宽，估算速度上限（峰值内存约 1.6GB，可用 --mb 调整）
pip install llama-cpp-python            # CPU 版
python3 thread_bench.py --model 模型路径/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

模型从 Hugging Face 官方仓库 `Qwen/Qwen2.5-0.5B-Instruct-GGUF` 下载（约 470MB，不随仓库分发）。`thread_bench.py` 测 1/2/4/8 线程的生成速度和不同提示长度的预填充速度，每项取 3 次最好成绩。

## 延伸实验（第二版文章新增）

```bash
python3 ctx_decode_bench.py --model 模型路径/qwen2.5-0.5b-instruct-q4_k_m.gguf   # 上下文长度对生成速度、n_ctx 对内存的影响，约 3 分钟
```
