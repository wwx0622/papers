# title

[MRGAT: Multi-Relational Graph Attention Network for knowledge graph completion](https://www.sciencedirect.com/science/article/abs/pii/S0893608022002714)

# Abstract

当前基于GNN的知识图完成方法假设节点的所有邻居具有同等重要性。

本文指出，这种不能为邻居分配不同权重的假设是不合理的。

此外，由于知识图是一种具有多个关系的异构图，节点和邻居之间的多个复杂交互可能会给GNN的有效消息传递带来挑战。

然后，设计了一个多关系图注意力网络（multi-relational graph attention network：MRGAT），该网络可以适应异构多关系连接的不同情况，然后通过自注意力层计算不同相邻节点的重要性。在具有不同节点权重的网络中加入自注意力机制可以优化网络结构，从而显著提高性能。

# Conclusion

现有的基于GNN的知识图嵌入模型在聚合关键邻域没有考虑异构多关系连接。
我们提出了一种新的图神经网络MRGAT，它可以适应异构多关系连接的不同情况，然后通过自注意力层计算不同相邻节点的重要性。在知识图完成任务中，自注意力层显式地建模了中心节点对其邻居的注意力程度，从而提高了预测三元组的可解释性。我们在知识图完成数据集上对MRGAT进行了综合评估，并证明与其他模型相比，MRGAT实现了SOTA性能。

由于GNN中众所周知的过平滑问题，深度编码器层将影响MRGAT在链路预测任务中的性能。因此，我们希望在未来加深MRGAT编码器的层次，以实现进一步的性能改进。此外，我们希望研究过拟问题，以提高MRGAT在高度节点上的性能。

# Intruduction

预测确实三元组通常表述为：给定一个查询(h,r,?)或(?,r,t)，其中?是KG中缺失的需要补全的值。预测缺失三元组通常也被称为**link prediction**。

随着表示学习的兴起，研究人员希望通过学习KG的数值表示，即知识图嵌入来完成KG。具体而言，知识图嵌入模型首先将实体和关系嵌入到隐藏空间中，以学习它们的低维密集向量表示，然后通过设计评分函数预测三元组中的缺失值。然而，传统的知识图嵌入模型，如TransE、DistMult和Conv，在学习实体和关系的表示时倾向于独立处理每个三元组。

几年来GNN在处理图书局势获得了良好的表现。但是现存的给予GNN的嵌入模型以下三种异构多关系连接：

* 头尾相同有多条边
* 头边相同有多个尾
* 边尾均不同

异构多关系连接是KG的关键属性，它使实体能够获得复杂拓扑和高阶结构的上下文意义。
因此，以前基于GNN的模型通过简单的静态注意机制聚集关键邻域信息，无法有效捕获局部邻域中固有的信息。
为了解决这个问题，本文提出了多关系图注意网络（MRGAT），它可以在复杂和不同的异构多关系连接情况下有效地选择关键邻域信息。换句话说，我们假设中心节点对其邻居给予不同的关注，这不仅取决于邻居本身，还取决于关系。

本文的工作：

* 提出了一种新的图神经网络MRGAT，它可以适应异构多关系连接的不同情况，然后通过自关注层计算不同相邻节点的重要性。在知识图完成任务中，自注意层显式地建模了中心节点对其邻居的注意程度，从而提高了预测三元组的可解释性。
* 在链路预测(link prediction)数据集上对MRGAT进行了全面评估，并证明与现有模型相比，MRGAT实现了最先进的性能。

# Preliminary

## Graph convolutional networks

作为一种GNN，图卷积网络（GCN）旨在通过聚集邻域信息来学习图的结构特征，从而使节点表示可用于节点分类或链路预测。

以前大多数基于GNN的知识图嵌入模型本质上是GCN的变体，以不同的方式将KG的拓扑信息合并到实体嵌入中。本文提出的模型MRGAT也可以看作是GCNs的一种变体。

大多数现有的GCNs模型遵循消息传递神经网络（MPNN）的框架，该框架将图上的卷积运算视为一个消息传播过程，通过连接边将一个节点的表示传递给另一个节点，然后对该运算进行K轮迭代。消息传递函数定义如下，

$h_v^{(k)}=U_k(h_v^{(k-1)},\sum_{u\in N(v)} M_k(h_v^{(k-1)}, h_u^{(k-1)}, x_{vu}^e))$
其中k表示图上的消息传播轮，$h_v^{(k)}$是第k轮中节点v的表示向量，$N（v）$是节点v的邻居集。$M_k$和$U_k$分别是消息函数和具有可学习参数的更新函数。$x_{vu}^e$表示源节点v和目标节点u之间的边缘e的特征。消息函数聚合来自邻居和边缘的信息，而更新函数基于聚合的信息生成新的节点表示。
## attention
### self-attention
$self-attention(Q, K, V) = softmax(\frac{K^TQ}{\sqrt{D_k}})V$
且$Q=W_qH, K=W_kH,V=W_vH$，H是输入
### multi-head attention
$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$
# Method
MRGAT可被视为编码器–解码器框架。编码器生成包含邻域信息和全局结构的实体表示。MRGAT可以通过堆叠多个编码器层来实现多跳邻居信息的聚合。编码器内部分为注意模块和聚合模块。
- 首先，注意力模块计算中心节点与其邻居之间的注意力得分。
- 然后，聚合模块基于注意力得分将邻居的信息聚合到中心节点，以便使用融合的邻居信息生成节点表示，其中注意力得分在控制该邻居的信息被聚合的程度方面发挥作用。
- 最后，编码器将生成的节点表示向量提供给解码器，解码器可以由任何现有的知识图嵌入模型组成。本文采用SACN中提出的Conv-TransE作为MRGAT的解码器。

## Encoder
### Attention module
中心节点对其邻居的关注不同：这种关注不仅取决于邻居，还取决于它们之间的关系（边）。受Transformer的启发，设计了一种应用于KGs的自注意机制，该机制根据节点之间的交互计算邻居的注意分数。
首先对每个R定义一个Q,K,给定中心节点v和他的邻居N(v),Q将v的向量投影到一个query向量中，K将N(v)中每个节点的表示向量投影到一个key向量中。通过query与key的点积运算得到节点对的分数。不同于Transformer，本文针对每一类边都定义了一个Q和K。
每一层的输入是所有节点的向量表示$H\in R^{N*D}$，其中N和D分别表示节点数量和维数。然后输出一个新的表示矩阵$H'\in R^{N*D'}$，节点的表示向量有高斯分布采样进行初始化。对任意节点对$(h, t) = (h_i, h_j)$有$q_i^r = h_iQ_r$, $k_j^r=h_jK_r$,得到这二者后进行点积运算并归一化得到分数$e_{ij}^r = \frac{q_i^{rT} k_j^r}{\sqrt{F}}$，F是q,k的维度。然后使用softmax函数标准化$a_{ij}=softmax_j(e_{ij})=\frac{exp(e_{ij})}{\sum_{k\in N_i}exp(e_{ik})}$,$a_{ij}$表示节点对$(h_i, h_j)$的注意力分数，两个节点是邻居。
### Aggregation module
聚合模块根据上述计算的注意力得分对邻居的表示向量进行线性组合，然后在中心节点上聚合它们以获得其新的表示向量。
$\widetilde{h_i}=\sum_{j\in N_i}a_{ij}h_jW$,$W\in R^{D*D'}$,是可学习的转移矩阵。上述聚合函数可以扩展到多头的：$\hat{h_i}=Concat(\widetilde{h_i}^{(1)}, ...)W_m$,$W\in R^{kD*D'}，其中k表示头数$。
考虑到深度对模型的影响，添加一个残差网络:$h_i'=\sigma(\hat{h_i}+h_iW)$
## Decoder
解码器可以采用任何知识图嵌入模型，如TransE、DistMult、ConvE等。本文使用Conv-TransE作为MRGAT的解码器。Conv TransE的灵感来自Conv，并使用卷积核来捕获实体和关系之间的交互。然而，Conv-TransE和Conv的主要区别在于，在叠加节点和关系表示后，Conv-TransE不会像ConvE那样执行整形操作，因此它可以具有与TransE相同的平移特性：h+r=t。
Conv-TransE：上述encoder的输出H作为实体嵌入，还有关系嵌入$E\in R^{T*\hat{D}}$,评分函数是：$f(h,r,t)=ReLu(vec(M(h,r))W)t$,M(h,r)组合两个向量为2*D的矩阵，然后映射到D维向量，vec将M()映射到1*CD的向量，W为CD*D的矩阵，t为D维向量，最后进行点积运算。最后计算一个概率：$p{h,r,t}=sigmoid(f(h,r,t))$
## Loss function
$L=\sum_{(h,r,t)\in T_i}-\frac{1}{N}\sum_{i=1}^N(y(h,r,t_i)*log(p(h,r,t))+(1-y(h,r,t_i)*log(1-p(h,r,t))))$，其中y是一个标签，值为0或1，N是所有尾结点总数。