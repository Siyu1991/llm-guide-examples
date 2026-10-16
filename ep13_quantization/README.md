# ep13 量化与本地部署

对应专栏第 13 期。纯 numpy，约 2 秒跑完，内存占用 < 300MB。

```bash
pip install numpy
python3 quantize.py
python3 test_quantize.py
```

实验内容：
1. 1024×1024 权重矩阵（人为加入 8 个离群通道）在 8/4/3/2 bit 下，per-tensor、per-channel、分组量化（group=128/64）的相对误差；
2. 训练一个 20→256→2 的小 MLP，量化权重后测准确率；
3. 按"参数量 × 位宽 ÷ 8 × 1.1"估算权重内存（不含 KV 缓存）。

这里的量化是最朴素的对称线性量化（round-to-nearest），GPTQ、AWQ 等方法在此基础上用校准数据进一步降低误差。

## 实测真实模型：gguf_bench.py

```bash
pip install llama-cpp-python numpy
# 从 Hugging Face 的 Qwen/Qwen2.5-0.5B-Instruct-GGUF 下载 fp16 / q8_0 / q4_k_m / q2_k 四个文件到 models/
python3 gguf_bench.py --dir models --threads 4
```

输出每个版本的文件大小、峰值内存、困惑度（固定一段中文文本）、10 道常识题正确数和生成速度。fp16 版峰值内存约 1.7GB，其余 < 1.2GB。

## 延伸实验（第二版文章新增）

```bash
python3 gptq_mini.py && python3 test_gptq.py   # 用 numpy 实现 GPTQ 误差补偿，对比就近取整
```
