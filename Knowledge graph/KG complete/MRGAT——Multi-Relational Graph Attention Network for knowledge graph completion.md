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
