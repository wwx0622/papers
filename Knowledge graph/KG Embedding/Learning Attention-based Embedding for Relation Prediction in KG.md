# title

[Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://arxiv.org/abs/1906.01195)
《在KG中针对关系预测学习基于注意力的嵌入》
[code](https://github.com/deepakn97/relationPrediction)

# Abstract

最近的几项研究表明，基于卷积神经网络（CNN）的模型可以生成更丰富、更具表现力的特征嵌入，因此在关系预测方面也表现良好。
然而，我们观察到这些KG嵌入独立地处理三元组，因此无法覆盖三元组周围的局部邻域中固有的复杂和隐藏信息。为此，本文提出了一种新的基于注意力的特征嵌入方法，该方法可以捕获任意给定实体邻域中的实体和关系特征。此外，我们还在模型中封装了关系簇和多跳关系。

# Conclusion and Future Work

本文提出了一种新的关系预测方法。此外，我们推广并扩展了图关注机制，以捕获给定实体的多跳邻域中的实体和关系特征。建议的模型可以扩展到使用幼儿园学习各种任务的嵌入。在未来，我们打算扩展我们的方法，以更好地执行分层图，并在我们的图注意力模型中捕捉实体（如基序）之间的高阶关系。

# Introduction

我们提出了一种用于关系预测的基于注意力的广义图嵌入。对于节点分类，图注意力网络（GAT）已被证明专注于图中最相关的部分，即1跳邻域中的节点特征。给定KG和关系预测任务，我们的模型通过将注意力引导到给定实体/节点的多跳邻域中的实体（节点）和关系（边缘）特征来概括和扩展注意力机制。

我们的想法是：

1. 为了捕获给定节点周围的多跳关系
2. 为了封装实体在各种关系中扮演的角色的多样性
3. 整合语义相似关系簇中存在的现有知识

我们的模型通过将不同的权重质量（注意力）分配给邻域中的节点并通过迭代方式通过层传播注意力来实现这些目标。然而，随着模型深度的增加，远距离实体的贡献呈指数级下降。为了解决这个问题，我们使用Lin提出的关系组合来在n跳邻居之间引入辅助边，从而容易地允许实体之间的知识流动。我们的架构是一个编码器-解码器模型，其中我们的广义图注意力模型和ConvKB分别扮演编码器和解码器的角色。

我们的贡献如下。据我们所知，我们是第一个学习新的基于图形注意力的嵌入技术的人，专门针对KG的关系预测。其次，我们推广和扩展了图关注机制，以捕获给定实体的多跳邻域中的实体和关系特征。最后，我们评估了我们的模型在各种现实数据集的关系预测任务上的挑战性。我们的实验结果表明，与现有的关系预测方法相比，我们有了明显的实质性改进。例如，我们基于注意力的嵌入比最先进的方法提高了104%Hits@1在流行的Freebase（FB15K-237）数据集上的度量。

# Method

## GAT
输入是一组节点的特征集: $\mathbf{x}=\{\overrightarrow{x_1}, ..., \overrightarrow{x_N}\}$
每一层的输出是: $\mathbf{x'}=\{\overrightarrow{x'_1}, ..., \overrightarrow{x'_N}\}$
对注意力参数进行softmax归一化后，在节点邻域上进行消息聚合：$\overrightarrow{x_{i}^{\prime}}=\sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j} \mathbf{W} \overrightarrow{x_{j}}\right)$

## Relations are Important

针对三元组$t_{ij}^k=(e_i, r_k, e_j)$，有$\overrightarrow{c_{i j k} }=\mathbf{W}_{1}\left[\vec{h}_{i}\left\|\vec{h}_{j}\right\| \vec{g}_{k}\right]$, 这生成了一个表示$t_{ij}^k$的向量。
然后通过$b_{i j k}=\operatorname{LeakyReLU}\left(\mathbf{W}_{2} c_{i j k}\right)$得到对于对应三元组的重要性。
归一化：$$\alpha_{i j k} =\operatorname{softmax}_{j k}\left(b_{i j k}\right)=\frac{\exp \left(b_{i j k}\right)}{\sum_{n \in \mathcal{N}_{i}} \sum_{r \in \mathcal{R}_{i n}} \exp \left(b_{i n r}\right)}$$
聚合: $\overrightarrow{h_{i}^{\prime}}=\sigma\left(\sum_{j \in \mathcal{N}_{i}} \sum_{k \in \mathcal{R}_{i j}} \alpha_{i j k} c_{i j k}^{\overrightarrow{ }}\right)$
多头注意力: $\overrightarrow{h_{i}^{\prime}}=\|_{m=1}^{M} \sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j k}^{m} c_{i j k}^{m}\right)$
为了获取原始嵌入，需要将输出加上输入进行输出，这其中要进行维度变换: $H_{out}=H_{attention\_layer}+WH_{input}$
