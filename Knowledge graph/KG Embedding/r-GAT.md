[toc]

# title

[r-GAT: Relational Graph Attention Network for Multi-Relational Graphs](https://arxiv.org/abs/2109.05922)

# Abstract

图形注意网络（**Graph Attention Networks, GAT**）专注于仅建模简单的无向和单关系图数据。这限制了它处理更一般和复杂的多关系图的能力，这些图包含具有不同标签的定向链接的实体（例如，知识图）。因此，在多关系图上直接应用GAT会导致次优解。为了解决这个问题，我们提出了r-GAT，一种关系图注意网络，用于学习多通道实体表示。具体来说，每个通道对应于实体的潜在语义方面。这使我们能够使用关系特征来聚合当前方面的邻域信息。我们进一步为后续任务提出了一种查询感知注意机制，以选择有用的方面。

# Conclusion

本文提出了一种新的多关系图的关系图注意网络（r-GAT），它学习不纠缠的实体特征，并在邻域聚合中利用关系特征。r-GAT的每个通道都特定于实体的语义方面。我们进一步提出了一种查询感知注意机制，以利用不同方面的信息。我们通过大量实验证明了不同模块的效率，并展示了我们提出的方法的可解释性。未来的工作将考虑进一步探索关系中的语义信息，并利用分离的实体特征。

# Introduction

本文提出了r-GAT来处理多关系图，它将图的实体和关系投射到多个独立的语义通道中。r-GAT中的每个通道都包含关系特征，以度量邻居的重要性，并同时聚合与相应方面相关的邻居信息。这使我们能够将实体的语义方面分解为多个组件。为了进一步利用分离的实体表示，我们提出了一种查询感知注意机制，为后续任务选择更多相关方面，从而获得自适应的实体表示。

- 本文提出了一种新的网络r-GAT，它利用关系特征并学习分离的实体表示来处理多关系图
- 我们提出了一种查询感知注意机制，以便更好地为后续任务利用不同的语义方面。
- 实体分类和链接预测任务的大量实验证明了r-GAT的有效性，并且我们表明，与其他方法相比，该方法更易于解释。

# Method
$$
\operatorname{att}_{v i u}^{k}=f^{k}\left[e_{v}^{k}\left\|r_{i}^{k}\right\| e_{u}^{k}\right]
$$
$$\begin{aligned}
\alpha_{v i u}^{k} &=\operatorname{softmax}_{u i}\left(\operatorname{att}_{v i u}^{k}\right) 
&=\frac{\exp \left(\operatorname{att}_{v i u}^{k}\right)}{\sum_{z \in \mathcal{N}_{v}} \sum_{j \in \mathcal{R}_{v z}} \exp \left(\operatorname{att}_{v j z}^{k}\right)},
\end{aligned}$$
$$e_{v}^{k(l)}=\sigma_{1}\left(\sum_{u \in \mathcal{N}_{v}} \sum_{i \in \mathcal{R}_{v u}} \alpha_{v i u}^{k}\left[e_{u}^{k} * r_{i}^{k}\right]\right)$$
$$e_{v}^{(l)}=\|_{k=1}^{K} \sigma_{1}\left(\sum_{u \in \mathcal{N}_{v}} \sum_{i \in \mathcal{R}_{v u}} \alpha_{v i u}^{k}\left[e_{u}^{k} * r_{i}^{k}\right]\right),$$
