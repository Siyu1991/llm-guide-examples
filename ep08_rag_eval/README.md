# ep08 RAG 效果评估

对应专栏第 8 期。用一个虚构网店的 22 篇客服文档（其中 10 篇是干扰/过期文档）和 24 道评测题，演示：

- 检索指标：Hit@k、Recall@k、MRR、nDCG@k
- 生成指标：答对率（关键要点命中）、忠实度（回答内容能在检索资料里找到的比例）
- 失败归因：每道错题判定是"没检索到"还是"检索到了但答错"
- 自助法（bootstrap）95% 置信区间：评测集太小时数字有多不可靠

```bash
python3 rag_eval.py        # 纯标准库，约 0.3 秒
python3 test_rag_eval.py
```

可选：设置 `OPENAI_API_KEY`（以及兼容接口的 `OPENAI_BASE_URL`、`OPENAI_MODEL`）后用大模型当裁判；不设置时使用离线规则裁判，结果可复现。
