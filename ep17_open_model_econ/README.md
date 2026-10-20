# ep17 开源公司怎么赚钱：推理成本账与训练回本

对应专栏第 17 期（辅线·商业化第 2 篇）。只依赖 numpy，不到 1 秒跑完。

数据来源：

- DeepSeek 在 2025 年 2 月开源周公开的 V3/R1 推理系统一天数据（节点数、H800 按 $2/卡·时、输入/缓存命中/输出 token 量、单节点吞吐、R1 价格下的理论收入）：https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- DeepSeek-V3 技术报告中的正式训练 GPU 小时（278.8 万 H800 小时，合 557.6 万美元）：https://arxiv.org/abs/2412.19437

其余参数（商业寿命、毛利档位、研究试验倍数等）是写在代码顶部的假设值，可以直接改。

```bash
python3 open_model_econ.py        # 打印结果，并写出 outputs/ep17_results.json
python3 test_open_model_econ.py
```

三部分：

- A. 标定：复算 DeepSeek 一天的 GPU 成本、理论收入和 545% 成本利润率；按单节点吞吐反推节点·时，与实际节点数核对；算每百万输入/输出 token 的单位成本、混合毛利，以及“付费占比”要过多少才不亏。
- B. 回本：给定一代模型的总投入、每百万 token 毛利和商业寿命，算每天需要多少付费 token 才能回本。
- C. 买方：给定 GPU 小时价，以 DeepSeek 公开吞吐为上限，算自建每百万 token 成本和与 API 打平所需的平均利用率。

注意：

- 成本只算 GPU 租金，不含研发人员、带宽、机房和销售；“毛利”“成本利润率”都不是公司财务口径。
- 自建保本利用率假设单节点能达到 DeepSeek 的专家并行吞吐，是乐观下限；小规模部署的实际吞吐通常低得多。
- 免费用户的推理成本没有计入回本计算，视为获客费用。
