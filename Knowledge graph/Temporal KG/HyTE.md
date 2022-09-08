# title
[HyTE: Hyperplane-based Temporally aware Knowledge Graph
Embedding](https://aclanthology.org/D18-1225/)
《HyTE: 基于超平面的时间感知知识图嵌入》
[Github项目地址](https://github.com/malllabiisc/HyTE)

# Abstract
KG中的关系事实通常随时间变化。大多数现有KG嵌入方法在学习KG元素嵌入时忽略了时间维度。本文提出了一种时间感知KGE方法HyTE，该方法通过将每个时间戳与对应的超平面相关联，显式地将时间合并到实体关系空间中。HyTE不仅使用时间引导执行KG推理，还预测缺少时间注释的关系事实的时间范围。

# Conclusion
- 本文提出了一种基于超平面的学习时间感知知识图嵌入的方法HyTE。
- HyTE利用KG的时间范围事实来LP以及未标注时间事实的时间范围预测。
- 在真实数据集上的大量实验结果显示，HyTE相对于传统和时间感知嵌入方法的有效性。
- 未来，我们希望结合类型一致性信息，进一步改进我们的模型，并将HyTE与开放世界知识图完成集成。

# Introduction
具有时间有效性标记的KG信念称为时间范围。这些时间范围越来越多地出现在几个大型KG上。主流KGE在学习KG中节点和关系的嵌入时忽略了这种时间范围的可用性或重要性。这些方法将KG视为静态图。这显然是不够的，可以想象，在表示学习期间合并时间范围可能会产生更好的KG嵌入。
HyTE将时间范围的输入KG分割成多个静态子图，每个子图对应于时间戳。然后，HyTE将每个子图的实体和关系投影到特定于时间戳的超平面上。学习超平面（法向）向量和KG元素随时间联合分布的表示。
本文工作如下:
- 提前注意到时间序列有关的KG，提出了HyTE，一种学习知识图（KG）嵌入的时间感知方法。
- HyTE直接在学习的嵌入中编码时间信息。这能够预测先前没有时间范围的KG的时间范围。
- 通过对多个真实数据集的大量实验，证明了HyTE的有效性。

# Background: KGE
- TransE：$e_h+e_r\approx e_t$, $f(h,r,t)=||e_h+e_r-e_t||_{l_1\ or\ l_2}$
- TransH: 将关系r建模为关系特定超平面上的向量，并将与其关联的实体投影到该特定超平面，以学习实体的分布式表示。$W_he_h+r\approx W_te_t$

实体的角色随着时间而变化，而且它们之间的关系也在变化。本文打算捕捉实体和关系的这种时间行为，并尝试相应地学习它们的嵌入。受TransH目标的启发，·提出了一种基于超平面的方法来学习时间分布的KG表示。

# Proposed Method：HyTE
HyTE(图1)，它不仅利用了实体之间的关系属性，还使用了与实体相关的时间元数据。
<center><image src='../../image/HyTE 01.png'></image><br>图1</center>

## Temporal Knowledge Graph
将单独的时间维度添加到三元组使KG动态。考虑四元组$(h，r，t，[\tau_s, \tau_e])$，$[\tau_s, \tau_e]$其中表示三元组（h，r，t）有效的开始和结束时间。HyTE将这此时间元事实直接纳入学习算法，以学习KG元素的时间嵌入。给定时间戳，可将图分解为若干静态图，包括在相应时间步长内有效的三元组。

**时间分量图**: $\mathbb{G}_\tau$, 该图中包含的三元组都是正三元组，并且有$\tau_s \le \tau \le \tau_e$
对应于时间$τ$的正三元组集合表示为$\mathcal{D}_\tau^+$。

## Projected-Time Translation

(h；r)对可以在不同的时间点与不同的尾部实体t相关联。在我们的时间引导模型中，我们希望实体具有与不同时间点关联的分布式表示。
时间表示为一个超平面，上面划分的每个时间分量图中有一个映射向量$\mathcal{w}_{t_1},...$，分别对应到T个子图上。现在每个子图中的三元组对于对应的时间$\tau$是有效的。
投影表示: 将投影向量标准化，即$\mathcal{w}_\tau$ $$\begin{array}{c}
P_{\tau}\left(e_{h}\right)=e_{h}-\left(w_{\tau}^{\top} e_{h}\right) w_{\tau} \\
P_{\tau}\left(e_{t}\right)=e_{t}-\left(w_{\tau}^{\top} e_{t}\right) w_{\tau} \\
P_{\tau}\left(e_{r}\right)=e_{r}-\left(w_{\tau}^{\top} e_{r}\right) w_{\tau}
\end{array}$$
评分函数：$f_{\tau}(h, r, t)=\left\|P_{\tau}\left(e_{h}\right)+P_{\tau}\left(e_{r}\right)-P_{\tau}\left(e_{t}\right)\right\|_{l_{1} / l_{2}}$
优化函数:$\mathcal{D}_\tau^-是负采样$ $$\mathcal{L}=\sum_{\tau \in[T]} \sum_{x \in D_{\tau}^{+}} \sum_{y \in D_{\tau}^{-}} \max \left(0, f_{\tau}(x)-f_{\tau}(y)+\gamma\right)$$
两种负采样方式:
***
- **时间不可知负采样(Time agnostic negative sampling, TANS)**: 考虑不属于KG的所有三元组的集合，而不考虑时间戳。$$\begin{aligned}
\{D_{\tau}^{-}\}_{TANS}=&\left\{\left(h^{\prime}, r, t, \tau\right) \mid h^{\prime} \in \mathcal{E},\left(h^{\prime}, r, t\right) \notin D^{+}\right.
\} \cup\left\{\left(h, r, t^{\prime}, \tau\right) \mid t^{\prime} \in \mathcal{E},\right.
&\left.\left(h, r, t^{\prime}\right) \notin D^{+}\right\} .
\end{aligned}$$ $\color{red}{在静态子图中随机换头或者换尾}$
- **时间相关负采样（Time dependent negative sampling, TDNS）**: 强调时间。除了与时间无关的负样本，还添加了额外的负样本。这些样本存在于KG中，但不存在于特定时间戳的子图中。$$\begin{array}{c}
\mathscr{D}_{\tau}^{-}=\left\{\left(h^{\prime}, r, t, \tau\right) \mid h^{\prime} \in \mathcal{E},\left(h^{\prime}, r, t\right) \in D^{+}\right.
\left.,\left(h^{\prime}, r, t, \tau\right) \notin \mathscr{D}_{\tau}^{+}\right\} \cup \\
\left\{\left(h, r, t^{\prime}, \tau\right) \mid t^{\prime} \in \mathcal{E},\left(h, r, t^{\prime}\right) \in D^{+},\right.
\left.\left(h, r, t^{\prime}, \tau\right) \notin \mathscr{D}_{\tau}^{+}\right\} .
\end{array}$$
$\color{red}{时间戳不同的三元组}$
***
约束: $\left\|e_{p}\right\|_{2} \leq 1, \forall p \in \mathcal{E}, \quad\left\|w_{\tau}\right\|_{2}=1, \forall \tau \in[T]$

# Experiments

- Datasets:
  1. YAGO11k
  2. Wikidata12k

- Method Compared:
  1. t-TransE: 该方法使用关系的时间顺序来在时间维度上建模知识演化。他们利用观察到的关于头部实体的关系排序来正则化传统的嵌入分数函数。
  2. HolE: 静态图学习方法(2016)
  3. TransE
  4. TransH
  5. HyTE: 本文方法

- Experiments:
  1. Entity Prediction
  2. Relation Prediction
  3. Temporal Scope Prediction