# ep24 结构化输出：校验 + 重试

用 Pydantic schema 约束工单抽取字段，对 mock 模型输出做 JSON 剥离、校验与纠错重试。离线运行，不需要 API key。

## 运行

```bash
pip install -r requirements.txt
python3 extract.py
python3 -m unittest test_extract -v
```

结果写入 `outputs/ep24_results.json`。

## 说明

* `TicketExtract`：枚举 category/priority、邮箱与 order_id 形态校验
* `MockModel`：按预设坏→好序列返回，模拟被纠错后收敛
* 生产环境可把 `MockModel.complete` 换成真实 Chat Completions / Structured Outputs 调用
