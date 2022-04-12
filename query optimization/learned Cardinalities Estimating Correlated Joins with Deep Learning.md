# 标题
[Learned Cardinalities:Estimating Correlated Joins with Deep Learning](https://arxiv.org/pdf/1809.00677.pdf)
# 摘要
提出MSCN(multi-set convolutional network)用于表示关系查询计划，通过集合语义捕捉查询特征和真实基数。
MSCN基于采样评估，解决了当没有采样元组限定谓词时的缺点，以及捕捉连接交叉相关性。
# 结论
MSCN是实现可靠的基于ML的基数估计的第一步，可以扩展到多个维度，包括**复杂谓词、不确定性估计和可更新性。**

另一个方向：MSCN可以预测列或列组合中唯一值的数量（即，估计GROUP BY的结果大小），虽然结果不太理想，当该方向有望通过ML解决。
# 简介
基数估计的最大问题：交叉连接相关性。

机器学习是解决基数估计问题的一种非常有前途的技术。

基数估计可以表述为一个有监督的学习问题，输入为查询特征，输出为估计基数。

机器学习模型产生的估计值可以直接被现有的、复杂的枚举算法和成本模型利用，而无需对数据库系统进行任何其他更改。

MSCN将查询表达式表示成集合：$(A \bowtie B)\bowtie C$和$A\bowtie(B\bowtie C)$都可以被表示成$\left \{ A,B,C\right \}$

连接枚举和成本模型由查询优化器解决。

在**IMDb**数据集表明MSCN技术比基于采样的技术更稳健，甚至在这些技术的最佳点（即，当有许多合格样本时）具有竞争力。
# 学习基数
问题：
- 如何表示查询； 
- 如何选择有监督学习算法。 
- 冷启动问题

符号：
$$
q: 查询\\
T_q: 表\\
J_q: 连接\\
P_q: 谓词集合
$$
每个表通过one-hot的$v_i$和一个基表或位图进行表示。每个连接通过one-hot向量表示。对于(col,op,val)形式的的谓词集合，将columns(col)和operators(op)通过one-hot向量表示，val进行min-max标准化。如图1所示：
![图1](/image/MSCN001.png)
下图所示的是**模型结构**：
![图2](/image/MSCN002.png)

模型原理：深度集合模型上的函数f(S)对集合有置换不变性，所以$f(S)=\rho[\sum_{x\in S}\phi(x)]$。因此MSCN模型结构做出以下选择：对于任意一个集合S学到一个特定的神经网络$MLP_S(v_s)$，所以最终输出是各个几个的均值：$w_S=\frac{1}{|S|}\sum_{s\in S}MLP_S(v_s)$。
查询表达式最终被组成:
$$
\begin{aligned}
表组件 &: w_T=\frac{1}{|T_q|}\sum_{t\in T_q}MLP_T(v_t)\\
连接组件 &: w_J=\frac{1}{|J_q|}\sum_{j\in J_q}MLP_J(v_j)\\
谓词组件 &: w_P=\frac{1}{|P_q|}\sum_{p\in P_q}MLP_P(v_p)\\
合并和预测 &: w_{out}=MLP_{out}([w_T,w_J,w_P])
\end{aligned}
$$
# 实验
随机抽样根据合格样本数推断输出基数。因此，它不能简单地从这个数字推断，必须在RS实现中使用单个连接的选择性的乘积，或者使用具有最选择性谓词的列的不同值的数量。相比之下，MSCN可以使用单个查询特征（在本例中是特定的表和谓词特征）的信号来提供更精确的估计。