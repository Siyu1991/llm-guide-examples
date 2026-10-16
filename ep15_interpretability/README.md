# ep15 可解释性入门

对应专栏第 15 期。纯 numpy，约 6 秒。数据是程序生成的虚构信贷申请（2 万训练、5 千测试）。

```bash
python3 credit_explain.py
python3 test_explain.py
```

演示：逻辑回归系数；小神经网络上的梯度×输入、积分梯度（验证"归因之和等于输出差"）、置换重要性；用线性探针检查隐藏层里是否编码了"地区"这个没直接输入的敏感属性，以及去掉代理变量"邮编区段"后的变化。

## 延伸实验（第二版文章新增）

```bash
python3 shapley_vs_ig.py && python3 test_shapley.py   # 精确 Shapley 与积分梯度、基准选择、步数、反事实
```
