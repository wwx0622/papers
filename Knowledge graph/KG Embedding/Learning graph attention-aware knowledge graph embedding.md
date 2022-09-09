# title
[Learning graph attention-aware knowledge graph embedding](https://www.sciencedirect.com/science/article/pii/S0925231221010961)

# Abstract
具体而言，最近的模型独立处理每个单个三元组或不加区分地考虑上下文，这是片面的，因为每个知识单元（即三元组）可以从其部分周围三元组中导出。本文中提出了一种基于图注意力的实体编码模型，该模型将知识图表示为不规则图，并通过多个独立通道整合图结构信息，探索了许多具体和可解释的知识组合。为了从不同角度（即实体对、关系和结构）测量实体之间的相关性，分别开发了三种注意力度量。通过使用增强实体嵌入，进一步引入了几个改进的因子分解函数，用于更新关系嵌入和评估候选三元组。

# Conclusion
本文将图多头注意机制引入到关系数据建模中，并通过大量实验证明了其在知识嵌入中的有效性。特别是，使用三种不同的注意力来源（实体对、关系和结构）来实现图中顶点特征的学习。实验结果验证了我们关于实体类型和关系的假设。
在未来，我们将探索特征学习中图注意力和图卷积之间的本质相似性。

# Introduction
由TransE及其变体表示的翻译模型将每个三元组视为训练样本，并通过投影实体和附加原则在各种约束下获得相应的知识表示。其他知识图嵌入研究集中于构建顶点为实体、边为关系的图，然后结合图结构和顶点的信息对知识进行编码。这些策略的一个明显限制是，它们不适合处理复杂的多关系数据，因为它们要么直接进行推理计算，而不考虑训练样本的相邻三元组，要么不加区分地组合所有环境因素。特别是，当单个给定实体对之间存在大量类型的候选实体或多个关系时，现有方法的性能将出现负增益。
本文引入了图注意机制，为知识图嵌入捕获更丰富和更细粒度的信息。我们模型的核心是通过选择性地合并来自相邻实体的信息并利用因子分解函数来确定候选三元组的概率，从而获得各种信道中实体的隐藏表示。这种新颖的基于图的多头注意机制带来了几个优点：
- 通过多通道处理输入信息，大大提高了现有知识图嵌入的可解释性；
- 通过迭代或叠加图关注层，可以轻松实现多跳信息获取；
- 由于我们算法的并行性和移动性，甚至可以保证对任意大规模知识图数据集的泛化。

此外，本文还提出了基于不同来源（即实体对、关系和结构）计算注意力系数的三个度量，以及获得高质量实体嵌入的实现。通过使用特定实体嵌入注入策略，我们改进了因子分解函数，以训练关系嵌入并对候选三元组进行评分。
工作:
1. 开发了一种新的图注意机制，充分考虑结构和实体信息来更新实体嵌入。
2. 通过考虑不同的来源和计算策略，提出了三种基于注意力的方法，并对它们捕获多通道信息的能力进行了比较研究。
3. 在实体分类和实体类型方面实现了新的最新性能，并改进了因子分解模型在链路预测中的有效性。

# Methodology
## Graph attention in knowledge graph
在图注意力模型中，基于通常用于表示学习的假设，即具有相似特征的顶点具有相似的邻居，注意力函数被类似地定义并应用于学习嵌入G中的实体。具体而言，图注意力层的输入是一组实体特征：$h=\{\overrightarrow{h_1}, ..., \overrightarrow(h_{N_e})\}$，图注意力层产生一组新的实体嵌入作为其输出，$h'=\{\overrightarrow{h'_1}, ..., \overrightarrow(h'_{N_e})\}$，二者维度可以变化。
由于相邻实体或实体对之间的关系可以作为信息获取的度量，我们将图注意的实现分为两类：实体注意和关系注意。特别是，为了消除三元组对实体之间关系的约束，我们还引入了结构注意来评估顶点之间相邻分布的相关性。

<center><image src="../../image/other 01.png"><br><text>注意力机制</text></center>
### Entity attention
实体对注意力：$$c_{ij}=a\left(\mathrm{~W} \vec{h}_{i}, \mathrm{~W} \vec{h}_{j}\right)=\operatorname{ReLU}\left(\vec{a}^{\top}\left[\mathrm{W} \vec{h}_{i}|| \mathrm{W} \vec{h}_{j}\right]\right)$$
实体对注意力归一化: $$\alpha_{i j}^{e}=\operatorname{softmax}\left(c_{i j}\right)=\frac{\exp \left(c_{i j}\right)}{\sum_{l=0}^{n_{i}} \exp \left(c_{i l}\right)},$$ $\color{red}{节点i对领域内所有节点注意力之和为1}$

### Relation attention
仅使用关系计算关注系数。本文采用了多头注意框架来实现类似于实体注意的多通道特征学习，并且可以通过知识图、任务要求或随机初始化中关系的相关性来确定注意对象头head。注意力系数可以通过点积注意力直接获得：$c_{ij}=ReLU(r_{ij}\cdot head)$, $head\in R^d$是各种注意力头的可训练的注意力参数。
类似的，可以得到归一化的关系注意力系数:$$\alpha_{i j}^{r}=\operatorname{softmax}\left(c_{i j}\right)=\frac{\exp \left(c_{i j}\right)}{\sum_{l=0}^{n_{i}} \exp \left(c_{i l}\right)},$$

### Structure attention
邻居对于实体相关性也有影响，提出基于结构指纹*structural fingerprint*计算两个实体邻居之间的相关性。首先，我们生成每个目标实体的结构指纹，即基于邻域中的实体与目标实体之间的结构关系确定重要性，并根据目标实体自适应地划分其邻域。然后，在评估两个实体之间的相关性时，我们通过分析两个实体的结构指纹之间的关系来计算结果。

直观地说，邻域中实体的重要性将随着与目标实体的距离增加而衰减，但由于邻域内部结构的连通性/密度，这种衰减将略有变化。为了根据局部图结构（连通形状或密度）调整相邻实体的权重，我们使用重新启动的随机游走策略（RWR）来计算权重并完成相应的自适应邻域划分。RWR通过模拟粒子在相邻节点之间的迭代运动来探索网络的全局拓扑。它量化了网络中节点之间的接近度，广泛应用于信息检索和其他领域。

为了获得实体的结构指纹，我们考虑其相邻实体集和相应的邻接矩阵。粒子从目标实体开始，并以与关系权重成比例的概率随机游走到中的相邻实体。在每个步骤中，它也有一定的概率返回到目标实体。迭代过程可以写成$$w_{i}^{(t+1)}=c \cdot \tilde{A}_{i} w_{i}^{t}+(1-c) \cdot v_{i}$$随机游走的概率为1，因此邻居矩阵列要进行标准化，即$\tilde{A}$，c为这种参数，$v_i$是对应实体i的ont-hot向量，以封闭形式求此收敛解为: $$w_{i}=\left(I-c \cdot \tilde{A}_{i}\right)^{-1} v_{i}$$

结构注意力参数: $$c_{i j}=\frac{\sum_{p \in\left(E_{i} \cup E_{j}\right)} \min \left(w_{i p}, w_{j p}\right)}{\sum_{p \in\left(E_{i} \cup E_{j}\right)} \max \left(w_{i p}, w_{j p}\right)}$$


结构注意力参数归一化: $$\alpha_{i j}^{s}=\operatorname{softmax}\left(c_{i j}\right)=\frac{\exp \left(c_{i j}\right)}{\sum_{l=0}^{n_{i}} \exp \left(c_{i l}\right)}$$

### Learning entity embedding
$$\vec{h}_{i}^{\prime}=\sigma\left(\sum_{j=0}^{n_{i}} \alpha_{i j} \mathrm{~W} \vec{h}_{j}\right)，其中的\sigma(x)=\frac{1}{1+\exp (-x)}$$

$$\alpha_{i j}=\frac{\lambda\left(\alpha_{i j}^{e}\right) \alpha_{i j}^{e}+\mu\left(\alpha_{i j}^{r}\right) \alpha_{i j}^{r}+\eta\left(\alpha_{i j}^{s}\right) \alpha_{i j}^{s}}{\lambda\left(\alpha_{i j}^{e}\right)+\mu\left(\alpha_{i j}^{r}\right)+\eta\left(\alpha_{i j}^{s}\right)}$$
因为超参数的调整不会显著影响性能，因此本文使用平均池：
$$\alpha_{i j}=\left(\right.  normalize  \left.\left(\alpha_{i j}^{e}\right)+\operatorname{normalize}\left(\alpha_{i j}^{r}\right)+\operatorname{normalize}\left(\alpha_{i j}^{s}\right)\right) / 3，\\ 其中的normalize  (\cdot)=\frac{\alpha_{i j}-\min (\alpha)}{\max (\alpha)-\min (\alpha)}$$K通道注意力学习：$$\vec{h}_{i}^{\prime}=\|_{k=1}^{K} \sigma\left(\sum_{j=0}^{n_{i}} \alpha_{i j}^{k} W^{k} \vec{h}_{j}\right) $$
可以根据下游任务的要求预先设置多头部注意框架，即使用域特征预训练变换权重矩阵W和权重向量a，或随机初始化。然后，我们可以根据任务要求将部分特征输入到特定的注意头中，或者将非线性综合特征表示为:
$$\vec{h}_{i}^{\prime}=\sigma\left(\frac{1}{K} \sum_{k=1}^{k} \sum_{j=0}^{n_{i}} \alpha_{i j}^{k} W^{k} \vec{h}_{j}\right) .$$

## 多关系数据建模
使用负采样来训练模型。对于对应于正知识的每个三元组，随机选择$n_s$个实体替换正三元组的实体与关系。我们优化了交叉熵损失，以鼓励我们的模型在实际三元组中得分高于负样本：$$\mathscr{L}=-\frac{1}{n_{s}+1} \sum_{i=0, t_{i} \in \mathscr{F}_{n s} \bigcup_{t_{r}}^{n_{s}+1}}\left(\hat{y}_{i} \log y_{i}+\left(1-\hat{y}_{i}\right) \log \left(1-y_{i}\right)\right)$$

# 总结
基于结构指纹的注意力是一个比较新奇的点，注意力的池化方式也是一个可以选择的点。