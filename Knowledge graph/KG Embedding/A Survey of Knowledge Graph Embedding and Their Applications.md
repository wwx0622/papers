# title
[A Survey of Knowledge Graph Embedding and Their Applications](https://arxiv.org/abs/2107.07842)
知识图谱嵌入与其应用的调研
# Abstract
KGE提供了一种表示知识的通用技术。
这些技术可用于各种应用，如完成知识图以预测缺失信息、推荐系统、问答、查询扩展等。
知识图中嵌入的信息虽然是结构化的，但在实际应用中很难使用。
KGE使真实世界的应用程序能够使用信息以提高性能。KGE是一个活跃的研究领域。
大多数嵌入方法关注基于结构的信息。最近的研究已经将边界扩展到实体嵌入中的基于文本的信息和基于图像的信息。已经做出努力来增强上下文信息的表示。本文介绍了KGE领域的发展，从简单的基于翻译(Translation)的模型到基于丰富的模型。本文包括知识图在实际应用中的应用。

# Introduction
KG经历了三个阶段:
1. Knowledge Representation(知识表示)被提升到Web标准的水平。
2. 核心重点转移到数据管理、链接数据和应用等。
3. 重点转向了现实世界的应用。
现实世界的应用范围包括语义分析、推荐系统、问答、命名实体消歧、信息提取等。
KGE是一种将知识图中的知识(实体与关系等)整合到现实应用中的解决方案。
KGE背后的动机是保留结构信息，即实体之间的关系，并将其表示在某些向量空间中。这使得操作信息更容易。
KGE的大部分工作集中在为实体和关系生成连续向量表示，并在嵌入上应用关系推理。关系推理应该优化一些评分函数，以学习嵌入。
- 基于嵌入路径的学习：Compositional learning of embeddings for relation paths in knowledge bases and text^[https://aclanthology.org/P16-1136/]
- 基于实体的学习
- 基于文本的学习
- 等

大量工作集中在翻译模型和基于语义的模型。三元组的表示会导致大量信息丢失，因为它没有考虑到“文本信息”。随着图形注意力网络(Graph attention network^[https://arxiv.org/abs/1710.10903])的提出，实体的表示变得更加上下文化。近年来，multi-model KG的提出将频谱扩展到了一个新的水平。在多模态知识图中，知识图可以具有图像和文本等多模态信息^[https://dl.acm.org/doi/10.1145/3340531.3411947]。
之前的调查工作主要集中在KGE^[https://dl.acm.org/doi/10.1145/3292500.3330989], KGE和应用^[https://arxiv.org/abs/2002.00388], 使用文本数据的KGE^[https://ieeexplore.ieee.org/document/9094215], 以及基于深度学习的KGE^[https://link.springer.com/article/10.1007/s00607-019-00768-7]。

# Knowledge Graph Embedding
这些方法大致分为两类：***翻译模型和语义匹配模型。***
## Translation Models: 翻译模型
基于翻译的模型使用基于距离的度量来生成一对实体及其关系的相似性得分。基于翻译的模型旨在找到与实体翻译相关的实体的向量表示。它将实体映射到低维向量空间。
### TransE^[https://link.springer.com/article/10.1007/s10994-013-5363-6]
度量：$h+r\approx t$
最小化损失函数:$$\mathcal{L}=\sum_{\substack{h,r,t\in S\\\hat{h},r,\hat{t}\in \hat{S}}}[\gamma+d(h+r, t)-d(\hat{h}+r,\hat{t})]$$
缺点：不能应对一对多、多对多、自环关系
### TransH^[https://ojs.aaai.org/index.php/AAAI/article/view/8870]
$h_{\perp}, t_{\perp}$
度量和损失同TransE
### TransR^[https://dl.acm.org/doi/10.5555/2886521.2886624]
度量:$hW_{h2r}+r\approx tW_{t2r}$
其余同上
### RotatE^[https://arxiv.org/abs/1902.10197]
关系是有一定对称/反对称、翻转、合成属性的，如"Marriage(结婚)"是具有对称属性的。"我的侄/甥女是我的姐妹的女儿"具有合成属性。本文提出的模型基于这样一种直觉：欧拉公式:$e^{l\theta}=\mathrm{sin}\,\theta\pm l\,\mathrm{cos}\,\theta$
度量：$t=h\circ r$
$\circ$表示对应元素乘。此外还需要将$r$单位化
因此有:$$\begin{align}
关系对称 &\Longleftrightarrow& \mathbf{r_i}\circ \mathbf{r_j}  = \mathbf{\pm1}\\
关系反对称&\Longleftrightarrow& r_i = \overline{r_j}\\
关系合成&\Longleftrightarrow&r_k=r_i\circ r_j
\end{align}
$$
### HakE^[http://arxiv.org/abs/1911.09419]
将知识图建模在极坐标圆上，需要两个量$(r;\theta)$,
- 令$h_m, r_m, t_m$表示模长，有度量：
  $d_{r,m}(h_{m},\,t_{m})\,=\,\left|\right|h_{m}\ \circ\,\,r_{m}\,-\,t_{m}\,||_2$
- 令$(h_{p}+r_{p})mod2\pi=t_p$表示角度，有度量：
  $d_{r,p}(h_{p},t_{p})\,=\,||\mathrm{sin}((h_{p}\circ r_{p}-t_{p})/\ 2)||_{1}\!$

## 语义匹配模型
语义匹配是自然语言处理的核心任务之一。语义匹配模型使用基于相似性的评分函数。该模型下有几种KGE算法。
### RESCAL^[https://dl.acm.org/doi/10.5555/3104482.3104584]
RESCAL遵循基于张量分解模型的统计关系学习方法，该模型考虑了关系数据的固有结构。张量因式分解是将张量表示为一系列其他张量（通常是更简单的张量）的初等运算。统计关系学习继承了概率论和统计学，以解决关系结构的不确定性和复杂性。
RESCAL使用三维张量$\mathcal{X}$表示知识图。
$$\mathcal{X_{ijk}}=\left\{\begin{matrix}
 1, & E_i 和 E_j 存在 R_k 关系\\
 0, & 其他
\end{matrix}\right.$$
对每个$r$的实体邻接矩阵$\mathcal{X_{ijr}}$进行以下的双线性分解：$$f_{r}(h, t)=\mathbf{h}^{T} \mathbf{M}_{r} \mathbf{t}=\sum_{i=0}^{d-1} \sum_{j=0}^{d-1}\left[\mathbf{M}_{r}\right]_{i j} \cdot[\mathbf{h}]_{i} \cdot[\mathbf{t}]_{j}$$其中的$M_r$符合:$$\mathcal{[M_r]_{ij}}=\left\{\begin{matrix}
 1, & E_i 和 E_j 存在 r 关系\\
 0, & 其他
\end{matrix}\right.$$
缺点：对稀疏的矩阵(关系数较少)的表现较差，导致过拟合
### TATEC^[https://link.springer.com/chapter/10.1007/978-3-662-44848-9_28]
评分函数有两部分组成:
1. 双路交互：$f_r^1(h,t)=\mathbf{h}^T\mathbf{r}+\mathbf{t}^T\mathbf{r}+\mathbf{h}^T\mathbf{D}\mathbf{t}$,D是一个共享的对角矩阵，不依赖输入的三元组
2. 三路交互：$f_r^2(h,t)=\mathbf{h}^T\mathbf{M}_r\mathbf{t}$

最终评分： $f_r(h,t)=f_r^1(h,t)+f_r^2(h,t)$
### DistMult^[https://arxiv.org/abs/1412.6575]
RESCAL的双线性公式与不同形式的正则化相结合，以形成不同的模型。在DistMult中，作者考虑了一种更简单的方法，通过将$M_r$限制为对角矩阵来减少参数的数量。这导致了一个更简单的模型，该模型具有与TransE相同的可伸缩性，并且比TransE实现了更好的性能。因此，最终得分函数如下所示：$$f_{r}(h, t)=\mathbf{h}^{T} \operatorname{diag}(\mathbf{r}) \mathbf{t}=\sum_{i=0}^{d-1}[\mathbf{r}]_{i} \cdot[\mathbf{h}]_{i} \cdot[\mathbf{t}]_{i}$$
缺点：过于简洁，虽然复杂度较与RESCAL和TATEC都低，但是因为过于简洁的模型在一般知识图上的表现较差，而且只适用于对称关系。
### HolE^[https://arxiv.org/abs/1510.04935]
HolE代表全息嵌入。HolE试图通过使用循环相关来克服RESCAL中使用张量积的问题。张量积使用特征向量之间的成对乘性交互，这导致表示的维数增加，也即到了维度平方的复杂度

优点：HolE的另一个优点是循环相关不是交换的（$h∗ t\ne t∗ h$）因此，HolE能够用成分表示对不对称关系（有向图）建模，这在RESCAL中是不可能的。
### ComplEx^[https://arxiv.org/abs/1606.06357]
KG的一个应用即预测缺失信息(实体或关系)
KG三元组向量嵌入的点积已成功用于对称、自反、反自反甚至传递关系，但它不能用于反对称关系。例如，关系capitalOf是不对称的，因为不能在这种关系中交换主语和宾语实体，因此需要对实体进行不同的嵌入，以增加参数的数量。复杂嵌入有助于主体和对象实体的联合学习，同时保持关系的不对称性。它使用主体实体和对象实体嵌入的Hermitian积。特征向量分解用于识别低秩对角矩阵W，从而存在$X=Re(EWE^T)$，使得X具有与Y相同的符号模式。然后，低秩对角线矩阵W通过应用$Re(e_s^TW\overline{e}_o)$来预测缺失关系。

<font color='red'>
笔者补充：W即特征值对角阵
</font>
### ANALOGY^[https://arxiv.org/abs/1705.02426]
$$\phi(s,r,o)=\langle v_{s}^{\top}W_{r},v_{o}\rangle=v_{s}^{\top}W_{r}v_{o}$$
## Enrichment based embedding
近年来，新出现的研究领域集中于上下文嵌入。在此情况下，所考虑的实体从邻域信息中丰富信息。
一些值得注意的方法是基于图形注意力网络（GAT）^[https://arxiv.org/abs/1710.10903] 的信息丰富。基于KGAT^[https://dl.acm.org/doi/10.1145/3292500.3330989] 和MMGAT^[https://dl.acm.org/doi/10.1145/3340531.3411947]的两种方法提出了嵌入实体上下文信息的模型。在MMGAT下，他们提出了一种模型，将多模态数据的嵌入与注意力框架相结合，采用GAT的注意力框架。两个框架都在使用翻译模型来学习丰富后的表示。新出现的研究领域是尝试学习结构信息以及基于路径的信息、多模态数据。嵌入中还有其他研究领域：文本增强嵌入、逻辑增强嵌入、图像增强嵌入等。

# Summary
知识图提供了一种展示真实世界关系的有效方法。
因此，知识图有一个固有的优势，即服务于信息需求。KG本身就是一个不断增长的研究领域。KGE是一种以向量形式表示KG的所有分量的技术。这些向量表示图的分量的潜在属性。
嵌入方法的各种模型基于向量代数的不同组合，这是一个有趣的研究领域。
在这项工作中，我们调查了开始这一活跃研究领域的嵌入方法、最新模型和KGE中正在探索的新前沿。
KGE方法从基于向量相加的基于翻译的模型发展而来。
在这项工作中，我们介绍了基于翻译的模型如何随着时间的推移而改进，以克服早期模型的缺点。
基于翻译的模型使用向量加法，而语义模型可以作为乘法模型组合在一起。我们包括了从基本语义模型到更高级语义模型的过渡，这些模型可以用来解释不同类型的真实世界关系，如对称、反对称、逆或复合。