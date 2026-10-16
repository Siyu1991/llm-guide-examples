# ep07 RAG 进阶：检索不准怎么办

对应专栏第 7 期。复用 `../ep06_rag/kb` 和 `../ep06_rag/questions.json`，另加 12 道难题。

```bash
pip install -r requirements.txt
python advanced.py --no-rerank   # 不下载重排模型，峰值内存约 0.5GB
python advanced.py               # 加交叉编码器重排 BAAI/bge-reranker-base（约 1GB），峰值内存约 1.9GB
```

三组实验：切块方式与大小、BM25 / 向量 / RRF 混合 / 重排、按元数据过滤已废止文档。结果写入 `result.json`。
