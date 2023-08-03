# title

[Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476)
《2D卷积KGE》
代码仓库: <https://github.com/TimDettmers/ConvE>

# Abstract

在这项工作中，我们介绍了**ConvE**，一种用于链路预测的多层卷积网络模型，并报告了几个已建立数据集的最新结果。我们还表明，该模型具有很高的参数效率，与DistMult和R-GCN具有相同的性能，参数减少了8倍和17倍。

# Conclusion

我们引入了ConvE，这是一种链接预测模型，它在嵌入和多层非线性特征上使用2D卷积来建模知识图。ConvE使用较少的参数；通过1-N打分速度快；它通过多层非线性特征表现出来；它对因批次归一化和脱落而导致的过拟合具有鲁棒性；并在多个数据集上获得最先进的结果，同时仍可以扩展到大型知识图。在我们的分析中，我们表明ConvE相对于普通链路预测器DistMult的性能可以部分解释为它能够以高（递归）度建模节点。

我们通过引入一个简单的基于规则的模型来调查常用数据集的这一问题的严重性，发现它可以在WN18RR与FB15k上获得最先进的结果。为了确保所有调查数据集的健壮版本存在，我们推导了WN18RR。

与计算机视觉中的卷积结构相比，我们的模型仍然是肤浅的，未来的工作可能会处理越来越深的卷积模型。进一步的工作可能还会研究2D卷积的解释，或者如何在嵌入空间中加强大规模结构，从而增加嵌入之间的交互数量。

# Introduction

增加浅层模型中功能的数量，从而提高其表现力的唯一方法是增加嵌入大小。然而，这样做并不能扩展到更大的知识图，因为嵌入参数的总数与图中的实体和关系的数量成正比。例如，像DistMult这样嵌入大小为200的浅层模型应用于Freebase，其参数需要33GB内存。要独立于嵌入大小增加特征的数量，需要使用多层特征。然而，以前以全连接层为特征的多层知识图嵌入体系结构容易溢出。解决浅层架构的缩放问题和完全连接的深层架构的过拟合问题的一种方法是使用可以组合成深层网络的参数高效、快速的运算符。

计算机视觉中常用的卷积算子正好具有这些特性：由于GPU实现的高度优化，它的参数效率高，计算速度快。此外，由于其广泛使用，在训练多层卷积网络时，已经建立了稳健的方法来控制过拟合.

在本文中，我们介绍了ConvE，这是一种在嵌入上使用2D卷积预测知识图中缺失链接的模型。ConvE是用于链路预测的最简单的多层卷积结构：它由单个卷积层、嵌入维度的投影层和内积层定义。

- 介绍了一种简单、有竞争力的2D卷积链路预测模型ConvE。
- 制定1-N评分程序，将训练速度提高三倍，评估速度提高300倍。
- 证明我们的模型具有高参数效率，在FB15k-237上比DistMult和R-GCN获得更好的分数，参数减少了8倍和17倍。
- 这表明，对于越来越复杂的知识图，我们的模型和浅层模型之间的性能差异与图的复杂程度成比例增加。
- 系统地调查报告的反向关系测试集在常用链接预测数据集中的泄漏情况，必要时引入数据集的健壮版本，以便使用简单的基于规则的模型无法解决这些问题。
- 在这些稳健的数据集上评估Conv和几个先前提出的模型：我们的模型在大多数数据集上实现了最先进的MRR。

# 2D卷积

$\psi_{r}\left(\mathbf{e}_{s}, \mathbf{e}_{o}\right)=f\left(\operatorname{vec}\left(f\left(\left[\overline{\mathbf{e}_{s}} ; \overline{\mathbf{r}_{r}}\right] * \omega\right)\right) \mathbf{W}\right) \mathbf{e}_{o}$