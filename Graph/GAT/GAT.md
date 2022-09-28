[toc]

# title

[Graph Attention Networks](https://arxiv.org/abs/1710.10903)

# Abstract

本文提出了图形注意网络（GAT），这是一种新型的神经网络体系结构，用于处理图形结构数据，利用掩蔽的自我注意层来解决以前基于图形卷积或其近似的方法的缺点。通过堆叠层，其中节点能够关注其邻域的特征，我们可以（隐式地）为邻域中的不同节点指定不同的权重，而无需任何昂贵的矩阵运算（例如求逆）或依赖于预先了解图形结构。

# Conclusion

本文提出了GAT，这是一种新型的卷积型神经网络，它利用隐藏的自我注意层，对图形结构数据进行操作。这些网络中使用的图形注意层计算效率高（不需要昂贵的矩阵运算，并且可以在图形中的所有节点之间并行），允许（隐含地）在处理不同大小的邻里时，将不同的重要性分配给邻里中的不同节点，并且不依赖于预先了解整个图形结构，从而使用以前的基于光谱的方法解决许多理论问题。本文利用注意力的模型在四个成熟的节点分类基准（包括传递性和归纳性）中成功实现或匹配了最先进的性能（尤其是使用完全看不见的图形进行测试）。

GAT有几个潜在的改进和扩展，可以作为未来的工作来解决，例如处理较大的批量。一个特别有趣的研究方向是利用注意机制对模型的可解释性进行彻底分析。此外，从应用程序的角度来看，扩展该方法以执行图分类而不是节点分类也很重要。最后，扩展模型以包含边缘特征（可能指示节点之间的关系）将允许我们处理更多的问题。

# Introduction

一方面，谱方法使用图的谱表示，并已成功应用于节点分类上下文中。在谱方法中，学习的滤波器依赖于拉普拉斯特征基，而拉普拉斯特征基取决于图形结构。因此，根据特定结构训练的模型不能直接应用于具有不同结构的图。

另一方面，我们有非谱方法直接在图上定义卷积。
在许多基于序列的任务中，注意机制几乎已成为事实上的标准。注意力机制的一个好处是，它可以处理各种大小的输入，专注于输入中最相关的部分，从而做出决策。当注意力机制用于计算单个序列的表示时，通常称为自我注意*self-attention*或内部注意*intra-attention*。与递归神经网络（RNN）或卷积一起，自我注意已被证明对诸如机器阅读和学习句子表征等任务有用。然而，self-attention不仅可以改进基于RNN或卷积的方法，而且足以构建一个强大的模型，在机器翻译任务中获得最先进的性能。
受最近这项工作的启发，本文引入了一种基于注意的体系结构来执行图结构数据的节点分类。其思想是通过关注其邻居，遵循自我关注策略，计算图中每个节点的隐藏表示。注意力架构有几个有趣的特性：

1. 操作是高效的，因为它可以跨节点邻居对并行；
2. 它可以通过为相邻节点指定任意权重来应用于具有不同程度的图节点；
3. 该模型直接适用于归纳学习问题，包括模型必须推广到完全不可见图的任务。

# GAT Architecture

## Graph Attention Layer

我们将首先描述一个图形注意层 **(graph attentional layer)**。

输入: $\mathbf{h}=\left\{\vec{h}_{1}, \vec{h}_{2}, \ldots, \vec{h}_{N}\right\}, \vec{h}_{i} \in \mathbb{R}^{F}$
每一层的输出: $\mathbf{h}^{\prime}=\left\{\vec{h}_{1}^{\prime}, \vec{h}_{2}^{\prime}, \ldots, \vec{h}_{N}^{\prime}\right\}, \vec{h}_{i}^{\prime} \in \mathbb{R}^{F}$
在每个节点对上求得注意力系数: $e_{i j}=a\left(\mathbf{W} \vec{h}_{i}, \mathbf{W} \vec{h}_{j}\right)$
注意力归一化: $\alpha_{i j}=\operatorname{softmax}_{j}\left(e_{i j}\right)=\frac{\exp \left(e_{i j}\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(e_{i k}\right)}$
综上，注意力系数: $\alpha_{i j}=\frac{\exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{j}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{k}\right]\right)\right)}$
信息传播: $\vec{h}_{i}^{\prime}=\sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j} \mathbf{W} \vec{h}_{j}\right)$
为了捕获更多特征，使用多头注意力: $\vec{h}_{i}^{\prime}=\sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \vec{h}_{j}\right)$

# 总结

利用图注意力网络进行消息传递。
