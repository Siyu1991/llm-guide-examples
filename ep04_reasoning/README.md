# ep04 推理模型：思考 token 的成本与收益

对应专栏第 4 期。纯 Python 标准库，无需安装依赖。

```bash
python reasoning.py cost                       # 思考 token 对月费的影响
python reasoning.py cost --think 0 2000 --pin 0.3 --pout 1.2   # 换成自己的单价
python reasoning.py vote                       # 自洽投票：N 次采样多数表决的正确率
python reasoning.py probe                      # 对比不同推理强度的 reasoning_tokens
```

`probe` 需要 OpenAI 兼容接口：

```bash
export OPENAI_API_KEY=你的密钥            # 不要写进代码或提交到仓库
export OPENAI_BASE_URL=https://api.openai.com/v1   # 可换成其他兼容服务
export REASONING_MODEL=你要测的推理模型名
```

没有设置 `OPENAI_API_KEY` 时自动进入离线 MOCK 模式，输出带 `"mock": true`，数字是假的，只用于跑通流程。
