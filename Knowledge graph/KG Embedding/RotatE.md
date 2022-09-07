# title
[RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197)
《RotatE：在复空间中通过关系旋转进行KGE》

# Abstract
LP在很大程度上取决于建模和推断关系模式的能力。
本文提出了RotatE，它能够建模和推断各种关系模式，包括：对称/反对称、反转和合成。具体而言，旋转模型将每个关系定义为在复杂向量空间中从源实体到目标实体的旋转。此外，我们提出了一种新的自对抗负采样技术，用于有效地训练旋转模型。

# Conclusion
- 本文提出了一种新的知识图嵌入方法，称为RotatE，它将实体表示为复杂向量，将关系表示为复杂矢量空间中的旋转。
- 本文提出了一种新的自对抗负采样技术，用于有效地训练旋转模型。
- 实验结果表明，在四个大规模基准测试中，旋转模型优于所有现有的状态模型。
- RotatE还在一个为组合模式推理和建模而明确设计的基准上实现了最先进的结果。
- 对旋转关系嵌入的深入研究表明，这三种关系模式都隐含在关系嵌入中。
- 未来，计划在更多数据集上评估旋转模型，并利用概率框架对实体和关系的不确定性进行建模。

# Introduction
KGE方法的一般直觉是根据观察到的知识事实来建模和推断知识图中的连通模式。例如，一些关系是对称的（如marriage），而另一些关系是反对称的（如filiation）；某些关系与其他关系相反（例如hypernym和hyponym， <font color='red'>这里可以理解为抽象与实例，或者概括与具体，比如动物是上位词(hypernym)，猫、狗为下位词(hyponym)</font>）；有些关系可能由其他人组成（例如，我母亲的丈夫是我父亲）。从观察到的事实中找到建模和推断这些模式（即**symmetry/anti-symmetry、inversion、composition**）的方法，以LP，这一点至关重要。
本文提出了一种称为**RotatE**的KGE方法。灵感来自来自欧拉恒等式$e^{iθ}=cos θ+i\cdot sin θ$，这表明同一单位的复数可以被视为复平面中的旋转。具体而言，旋转模型将实体和关系映射到复杂向量空间，并将每个关系定义为从源实体到目标实体的旋转。给定三元组（h；r；t），我们期望$t=h\circ r$、 h, r, t都是嵌入且$|r|=1$，并且$\circ$是Hadamard积，也即元素对应相乘。
- 对称关系: $\forall r_i=e^{\frac{0}{r}\pi}=\pm1$
- 共轭: $\mathbf{r_1}=\hat{\mathbf{r_2}}$
- 组合关系: $\mathbf{r_3}=\mathbf{r_2}\circ\mathbf{r_1} 或者说 \theta_3 = \theta_1+\theta_2$

此外，因其线性的时间空间复杂度，在大型KG上同样适用。

新的自对抗负采样技术: 该技术根据当前实体和关系嵌入生成负采样。所提出的技术非常通用，可以应用于许多现有的知识图嵌入模型。

# RotatE: Relational Rotation In Complex Vector Space
RotatE模型能够建模和推断所有三种关系模式。

## Modeling and Inferring Relation Patterns
**Definition 1**. A relation  $\mathrm{r}$  is **symmetric (antisymmetric)** if  $\forall \mathrm{x}, \mathrm{y} $

$$r(\mathrm{x}, \mathrm{y}) \Rightarrow \mathrm{r}(\mathrm{y}, \mathrm{x})(\mathrm{r}(\mathrm{x}, \mathrm{y}) \Rightarrow \neg \mathrm{r}(\mathrm{y}, \mathrm{x}))$$

A clause with such form is a **symmetry (antisymmetry)** pattern.
**Definition 2**. Relation  $r_{1}$  is **inverse** to relation  $r_{2}$  if  $\forall \mathrm{x}, \mathrm{y} $

$$\mathrm{r}_{2}(\mathrm{x}, \mathrm{y}) \Rightarrow \mathrm{r}_{1}(\mathrm{y}, \mathrm{x})$$

A clause with such form is a **inversion** pattern.
**Definition 3**. Relation  $r_{1}$  is **composed** of relation  $r_{2}  $and relation  $r_{3}  if  \forall \mathrm{x}, \mathrm{y}, \mathrm{z} $

$$\mathrm{r}_{2}(\mathrm{x}, \mathrm{y}) \wedge \mathrm{r}_{3}(\mathrm{y}, \mathrm{z}) \Rightarrow \mathrm{r}_{1}(\mathrm{x}, \mathrm{z})$$

A clause with such form is a **composition** pattern.
![比对表](../../image/RotatE%2001.png "几种模型的比较")

## Modeling Relations as Rotations in Complex Vector Space
核心思想: $\mathbf{t}=\mathbf{h}\circ\mathbf{r}, \forall  1 <= i <= len(\mathbf{r})，|r_i|=1$
距离度量: $d_r(\mathbf{h}, \mathbf{t})=||\mathbf{h}\circ\mathbf{r}-\mathbf{t}||$
TransE除了对称关系外都能建模，因为对称关系会被计算到0，导致头和尾无法区分。

## Optimization
损失函数:$\gamma是固定间隔，\sigma是sigmoid函数，以及(h', r, t')是负三元组$ $$L=-\log \sigma\left(\gamma-d_{r}(\mathbf{h}, \mathbf{t})\right)-\sum_{i=1}^{n} \frac{1}{k} \log \sigma\left(d_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)-\gamma\right)$$
提取负样本的新方法：本文提出了一种称为自对抗负采样的方法，该方法根据当前的嵌入模型对负三元组进行采样。具体而言，我们从以下分布中采样负三元组：$$p\left(h_{j}^{\prime}, r, t_{j}^{\prime} \mid\left\{\left(h_{i}, r_{i}, t_{i}\right)\right\}\right)=\frac{\exp \alpha f_{r}\left(\mathbf{h}_{j}^{\prime}, \mathbf{t}_{j}^{\prime}\right)}{\sum_{i} \exp \alpha f_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)}$$
$\alpha$是采样参数，此外，由于采样过程可能成本高昂，将上述概率视为负样本的权重。因此，具有自对抗训练的最终负采样损失采用以下形式：$$L=-\log \sigma\left(\gamma-d_{r}(\mathbf{h}, \mathbf{t})\right)-\sum_{i=1}^{n} p\left(h_{i}^{\prime}, r, t_{i}^{\prime}\right) \log \sigma\left(d_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)-\gamma\right)$$

# Experiments
- FB15k: **symmetry/antisymmetry 和 inversion**
- WN18: **symmetry/antisymmetry 和 inversion**
- FB15k-237: 子集, **symmetry/antisymmetry 和 composition**
- WN18RR: 子集， **symmetry/antisymmetry 和 composition**

**Baseline**：pRotatE:令$|h_i|=|t_i|=C$，距离度量为:$2 C\left\|\sin \frac{\boldsymbol{\theta}_{h}+\boldsymbol{\theta}_{r}-\boldsymbol{\theta}_{t}}{2}\right\|$

# 总结
优：
1. 能够很好的建模对称/反对称关系
2. 新的自对抗负采样技术(借鉴？)

缺点：
1. 主要针对关系建模，忽视了实体的结构信息
2. 拥有大多类似模式的关系的数据集效果良好，否则表现效果不佳。