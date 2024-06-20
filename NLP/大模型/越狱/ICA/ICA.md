# paper

[Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations](https://arxiv.org/pdf/2310.06387)

# method

利用模型的In-Context Learning(上下文学习)能力进行越狱攻击

![alt text](image.png)

## 算法流程

收集一个有害数据集$C=\{(x_1, y_1), ..., (x_n, y_n)\}$，其中$x_i$是输入(越狱攻击)，$y_i$是输出(有害回应)。

每次越狱攻击从$C$中采样$k$条越狱攻击对以构造新的ICA攻击prompt: $P=[x_1, y_1, ..., x_k, y_k, x_{new}]$，并通过该prompt得到新的回复$y_{new}$

## 总结

ICA是一种完全基于prompt的越狱攻击算法。

个人猜测能够成功的工作原理在于：自回归模型会把自己之前的输出当做**确定**的输入，因此越狱攻击可以利用这一点。这种现象同样在计算困惑度时使用。因此，通过上下文学习并收集若干越狱攻击对话，可以伪造**对话历史**，破坏大模型的对齐。
