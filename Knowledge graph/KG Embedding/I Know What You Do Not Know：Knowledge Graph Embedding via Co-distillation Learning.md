# title
[I Know What You Do Not Know: Knowledge Graph Embedding via Co-distillation Learning](https://arxiv.org/abs/2208.09828)
# Abstract
传统模型对图结构进行推理，但它们存在图不完全性(graph incompleteness)和长尾实体(long-tail entities)的问题。
最近的研究使用预先训练的语言模型来学习基于实体和关系的文本信息的嵌入，但它们不能利用图结构。
在本文中，我们通过经验证明这两种特征对于KG嵌入是互补的。为此，我们提出了CoLE，一种用于KG嵌入的**共蒸馏学习(Co-distillation Learning)**方法，该方法利用了图结构和文本信息的互补性。其图嵌入模型采用Transformer重构邻域子图实体的表示。它的文本嵌入模型使用预训练语言模型，根据名称、描述和关系邻居的软提示生成实体表示。
为了让这两个模型相互促进，我们提出了共蒸馏学习，允许它们从彼此的预测逻辑中提取选择性知识。

# Conclusion
CoLE在基于结构和基于PLM的KG嵌入模型之间寻求有效的知识转移和相互增强。
- 对于基于结构的KG嵌入，我们提出了N-Former，它基于不完整三元组的关系邻居来重构和预测缺失的实体。
- 对于基于PLM的KG嵌入，我们提出了一种N-BERT，它通过探测具有实体名称、描述和邻居提示的BERT来生成缺失的实体表示。

CoLE首先将两个模型的预测概率解耦，然后通过概率蒸馏(?logit distillation)进行双向知识转移，让它们相互传授有用的知识。
在FB15K-237和WN18RR上的实验表明，与现有工作相比，N-Former和N-BERT实现了竞争性甚至最佳的结果。
CoLE取得了最优结果。
在未来的工作中，我们计划研究多源KG之间的知识转移，并使用额外的KG嵌入任务（如实体对齐）进行实验。
# Introduction
TODO:阅读 (Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs)^[https://arxiv.org/abs/1905.04914]
TODO:阅读(Modeling Relation Paths for Representation Learning of Knowledge Bases)^[https://aclanthology.org/D15-1082/]
TODO:阅读(RDF2Vec: RDF Graph Embeddings for Data Mining.)^[https://link.springer.com/chapter/10.1007/978-3-319-46523-4_30]
TODO:阅读(HittER: Hierarchical Transformers for Knowledge Graph Embeddings)^[https://aclanthology.org/2021.emnlp-main.812.pdf]
TODO:阅读(Modeling Relational Data with Graph Convolutional Networks)^[https://arxiv.org/abs/1703.06103]
TODO:阅读(Composition-based Multi-Relational Graph Convolutional Networks)^[https://arxiv.org/abs/1911.03082]
现有的KG嵌入模型主要侧重于探索图结构，包括对边缘似然性[TransE, ConvE, RotatE]进行评分，对路径进行推理，以及对邻域子图进行卷积或聚合。我们将这些研究称为**基于结构的模型**。从图结构中学习与实体或关系的名称无关，但存在不完整性和稀疏性问题，因此很难预测具有很少边的长尾实体的三元组。
预训练语言模型（pre-trained language model简称PLM，如BERT）具有存储从大量文本语料库中获得的一些事实和常识知识的能力，可以帮助完成KGE。基于PLM的KG嵌入模型通过拼接实体和关系的名称将三元组转换为自然语言风格的序列。由PLM编码自然语言序列。输出的表示可以用于预测或生成掩码实体。基于PLM的KG嵌入模型不存在不完整性问题，因为PLM是使用外部开放域语料库训练的。
<font color='red'>
粒度：将<h, r, t>表示成str(CONCAT(h, r, t))的序列，进行后续实验。
</font>
初步试验：在FB15K-237数据集上，基于每个实体的度(所连边数)进行分组。实验结果如下图:
![实验结果](../../image/ColE%2001.png)
可以看到，
- 在长尾实体的连接预测(LP)中，kNN-KGE比HittER表现得更好（参见[1,6]组的结果），因为它可以从PLM的外部知识中受益。
- HittER对于具有丰富边缘的实体的性能优于kNN-K GE。这表明它们具有强互补性。

此外，在[6,+∞]的组中，它们的表现逐渐变差. HittER性能下降的原因在于实体的富边通常涉及多映射（例如，一对多和多对一）关系。而在PLM模型也有类似问题的挑战，如噪声、同音异义等多映射关系。

使用T-REx和部分Google RE中的实体名称作为Google搜索的查询关键字，并根据检索到的网页数量将实体分成若干组。下图是LP结果：![实验二](../../image/CoLE%2002.png)
该实验揭示了PLM在富文本时的局限性，可以看到，LAMA在流行实体的链接预测中表现不佳。基于PLM的模型在遇到一些名称相似甚至相同的流行实体时，会出现自然语言歧义。另一个原因是，尽管使用大量文本语料库对PLM进行训练，但很难从PLM中检索有用的直接知识来帮助完成特定的KG。
本文为了克服以上的局限性提出了CoLE，通过基于结构的模型N-Former和基于PLM的模型N-BERT之间的共蒸馏学习来嵌入KG。
关键思想是让两个模型有选择地相互学习和教导。具体而言，CoLE由三个部分组成：
1. 基于结构的模型N-Former采用Transformer通过实体的邻域子图(邻居)来重构不完整三元组的缺失实体。具体来说，给定一个不完整的三元组(ℎ,𝑟, ?), N-Former首先通过邻居重建ℎ的表示。然后将此表示与h(?这里应该是r)的原始结合重构缺失实体t。
2. 基于PLM的模型N-BERT建立在BERT的基础上，并试图从软提示中生成缺失的实体表示，该软提示包括所见实体和关系的描述、邻居和名称。提示中的描述和邻居信息可以帮助检索相关特定实体的PLM中隐藏的知识。
3. CoLE不假设一个模型是教师，另一个是学生。而是认为这两种模式在大多数情况下是互补的。为了使它们相互受益，CoLE基于两个模型的解耦预测逻辑设计了两个知识蒸馏（KD）目标。第一个目标是将N-Former关于高置信度预测的知识转移到N-BERT，而第二个目标是从N-BERT转移到N-Former。将模型的预测逻辑分为两个不相交的部分，分别计算两个KD目标，用于选择性知识转移和避免负转移。

结果表明，与现有相关工作相比，N-Former和N-BERT这两种模型实现了相当甚至更好的性能，而具有共蒸馏学习的集成方法CoLE提高了KG嵌入的最新水平。
# Related Work
仅介绍PLM KGE。
LAMA^[https://arxiv.org/abs/1909.01066]是一种知识探测模型，它首先揭示了PLM可以捕获训练数据中存在的事实知识，而结构化为完形填空语句的自然语言查询能够在不微调PLM的情况下获取此类知识。然而，一些强大的限制，如手动构造的提示和仅预测具有单个token名称的实体，阻碍了LAMA进行KG嵌入任务。
KG-BERT^[https://arxiv.org/abs/1909.03193]是第一个将PLM应用于KG嵌入的模型，通过简单地连接实体名称和关系名称，将三元组转化为自然语言句子，然后对序列分类任务的BERT进行微调。
在KG-BERT之后，PKGC^[https://aclanthology.org/2022.findings-acl.282/]利用手动模板构建充分利用PLM的连贯句子。它还添加了实体定义和属性作为支持信息。
KG-BERT和PKGC都是三重分类模型。然而，使用三重分类模型进行LP非常耗时。他们通常假设KG中出现的所有实体都是候选实体，因此对于一个不完整的三元组，他们需要许多推理步骤，当KG很大时，这是不切实际的。
为了在一个推理步骤中预测缺失的实体，MLMLM^[https://arxiv.org/abs/2009.07058]在提示中添加多个[MASK]标记，以预测具有多标记名称的实体.
kNN KGE^[https://arxiv.org/pdf/2201.05575.pdf]利用PLM从其描述中学习每个实体的初始表示。
# Approach
## Notations
在KG中，实体或关系通常用统一资源标识符（URI）表示。例如，科比·布莱恩特在Freebase中的URI是/m/01kmd4，CoLE假设每个实体$e\in \mathcal{E}$与关系$𝑟 ∈ \mathcal{R}$具有人类可读的文字名称，如“Kobe Bryant”，记为$𝑁_𝑒$和$𝑁_𝑟$,。$D_e$表示e的描述.$\mathbf{E}$表示嵌入,如$\mathbf{E}_e$
## Framework Overview
![总体框架](../../image/CoLE%2003.png "CoLE总体框架")
目标是预测不完整三元组中缺失的实体，例如FB15K-237中的（Kobe Bryant，location，?）。用于支持此预测的可用信息包括科比的邻域及其文本描述。
N-Former首先将子图作为输入来重构科比的表示。该表示连同科比和$R_{location}$的初始嵌入一起被馈送到N-Former中，以重构输入中记录为特殊占位符“$[MASK]$”的缺失实体的表示。$[MASK]$的输出表示用于计算预测概率。此外，基于PLM的模型N-BERT根据实体的描述、邻居和名称构造提示，作为BERT的输入。丢失的实体也将替换为占位符$[MASK]$。$[MASK]$的BERT输出表示用于预测丢失的实体。
CoLE假设这两个模型是互补的，它们应该互相传授各自擅长的东西。它首先将概率分为两部分，一部分涉及N-Former的“高置信度”预测，另一部分涉及N-BERT。然后，它通过最小化两个模型的部分概率的KL散度来实现双向知识转移。
## Neighborhood-aware Transformer
![N-Former](../../image/CoLE%2004.png "N-Former整体架构")
上图显示了Transformer为主干的N-Former的架构。N-Former基于递归实体重建的思想，使模型能够看到更多的预测信息，同时保持对图不完整性和稀疏性的鲁棒性。
### Entity Reconstruction from a Triplet
给定$(ℎ,𝑟, [MASK])$，重构t的表示，将其随机初始化，送入Transformer，并获得$[MASK]$的输出表示：$$(1)：\mathbf{E}_{[\mathrm{MASK}]}^{n}=\mathrm{Transformer}(\mathbf{E}_{h}^{0},\mathbf{E}_{r}^{0},\mathbf{E}_{[\mathrm{MASK}]}^{0})$$其中𝑛表示自注意力层(self-attention, SA)的数量。在SA的帮助下，$[MASK]$的输出表示可以从实体捕获信息ℎ 和关系𝑟.
重构：令$\mathbf{E}_t=\mathbf{E}_{[MASK]}^n$，预测概率$P_t$为：$$(2)：{\bf P}_{t}={\bf S o f t m a x}\big({\bf E_{E N T}\cdot M L P(E_{t})}\big),$$其中的$E_{ENT}$是全部实体的嵌入矩阵。
计算预测损失:${\mathcal{L}}_{\mathrm{triplet}}(t)={\mathsf{C r o s}}{\mathsf{s}}{\mathsf{E}}{\mathsf{ntr}}{\mathsf{o p y}}({\mathsf{P}}_{t},{\mathsf{L}}_{t}),$
交叉熵损失函数为：${\mathsf{C r o s s E n t r o p y}}(\mathbf{x},\mathbf{y})=-\sum_{j}\mathbf{x}_{j}\log\mathbf{y}_{j},$
<font color='red'>
$L_t$没有声明，或许是对应t的one-hot向量？
</font>

### Entity Reconstruction from Neighborhood
递归调用上述过程，得到邻域信息。
ℎ还可以从其关系邻居重建的表示中受益，这也使模型能够看到更多信息，以帮助预测长尾实体。
h的邻居记为$Neighbor(h)=\left\{\left(h^{\prime}{}_{,}{r}^{\prime}\right)\mid\left(h^{\prime}{}_{,}{r}^{\prime}{}_{,}h\right)\in\mathcal{T}\right\}$
**请注意**，CoLE添加了一个反向三元组$(𝑡,𝑟^−, ℎ)$对于每个三元组$(ℎ,𝑟, 𝑡)$，因此邻域中只需考虑入边。
通过对邻域的嵌入求和得到：$$\mathbf{E}_{\text {Neighbor }}^{S}=\sum_{\left(h^{\prime}, r^{\prime}\right) \in \mathrm{Neighbor}(h)} \text { Transformer }\left(\mathbf{E}_{h^{\prime}}^{0}, \mathbf{E}_{r^{\prime}}^{0}, \mathbf{E}_{[\mathrm{MASK}]}^{0}\right)$$

结合上式对式(1)进行扩展：$$\mathbf{E}_{[\mathrm{MASK}]}^{S}=\operatorname{Transformer}\left(\operatorname{mean}\left(\mathbf{E}_{h}^{0}, \mathbf{E}_{\text {Neighbor }}^{S}\right), \mathbf{E}_{r}^{0}, \mathbf{E}_{[\mathrm{MASK}]}^{0}\right)$$
<font color="red">
这里的mean()函数是求均值。感觉有残差网络的想法。但是又考虑到h的初始嵌入不可能与[MASK]有太大的相关性，这里可以用更多的方法进行结合如:
$CNN(E_h^0;E_{Neighbor}^S)$、$\beta E_h^0+(1-\beta)E_{Neighbor}^S$等。
</font>
如式(2)求解$P_t^S$，计算损失：$$(6)：\mathcal{L}_{\mathrm{structure}}=\sum_{(h,r,t)\in\mathcal{T}}\left(\mathcal{L}_{\mathrm{triplet}}(t)+\mathsf{C r o s s E n t r o p y}(\mathsf{P}_{t}^{S},\mathsf{L}_{t})\right).$$
## Neighborhood-aware BERT
BERT捕获了一些结构化知识，这些知识可用于LP。一个解决$(h,r,[MASK])$的典型方案如下图: ![triplet prompt](../../image/CoLE%2005.png) 其中[CLS]、[SEP]是用于分隔提示不同部分的特殊标记，缺失的实体用占位符[MASK]替换。
**需要注意**因为得到关系r的反关系$r^-$是一个很困难的任务，所以不用对关系取逆。最终，通过提示得到:$\mathbf{E}_{[M\mathrm{ASK}]}=\operatorname{BERT}({\mathrm{prompt}}_{T}(h,r)),$
为了在一个推理步骤中获得预测的实体，我们需要将实体的名称作为新的token添加到PLM的词汇表中。但这些新令牌的随机初始嵌入表现不佳，因为它们不拥有来自开源领域语料库的任何外部知识。CoLE使用从描述提示中学习的嵌入作为实体的初始表示：
$\mathbf{E}_{\mathrm{h}}^{i n i t}=\mathrm{BERT}("T h e\,d e s c r i p t i o n\,o f\,[MA S K]\,\,i s\,D_{h}"{\mathrm{~)}}$
### Soft Prompts
更连贯的提示将更好地利用PLM中的知识，而[SEP]连接的提示显然没有表现力。
使用一些可调整的软提示，使提示更具表现力。受PKGC的启发，我们将其他token之间的特殊[SEP]令牌替换为关系感知软提示，以获得一种新的提示——***关系提示***，其中$[SP]^𝑟_𝑖 (𝑖 = 1、2、3、4）$表示软提示𝑟的第$i$个软提示,并且$𝑖$指示要插入相应软提示的位置。在实施中，$[SP]^𝑟_𝑖$ 是添加到词汇表中并与关系$r$相关的特殊标记, 并且其表示被随机初始化。

### Neighborhood Prompts
为了充分利用KG中的相邻三元组，引入了实体的上下文嵌入作为附加支持信息，也可以将其重新分级为entity-aware软提示。记$InNeighbor(ℎ) = {(ℎ′,𝑟′) | (ℎ′,𝑟′, ℎ) ∈
T}$和$OutNeighbor(ℎ) = {(𝑟′, 𝑡′) | (ℎ,𝑟′, 𝑡′) ∈ T}$，顾名思义。
类似的:$$\mathrm{\mathrm{E}}_{N e i g h b o r}^{T}=\sum_{(h^{\prime},r^{\prime})\in\mathrm{InNeighbor}(h)}\mathrm{\small{BERT}}(\mathrm{\small{Prompt}}_{R}(h^{\prime},r^{\prime}))+\sum_{\left(r^{\prime}, t^{\prime}\right) \in \text { OutNeighbor }(h)} \operatorname{BERT}\left(\text { prompt }_{R}\left(r^{\prime}, t^{\prime}\right)\right)$$
然后令:$\left[\mathrm{Neighbor}\right]\,=\,\mathbf{E}_{N e i g h b o r}$
得到：$\mathrm{E}_{h}^{T}=\mathrm{BERT}(\mathrm{prompt}_{N}(r,t))$；$\mathrm{E}_{t}^{T}=\mathrm{BERT}(\mathrm{prompt}_{N}(h,r)),$
通过式(2)得$P_h^T与P_t^T$。
计算损失：$$(11)：\mathcal{L}_{\mathrm{{text}}}=\sum_{(h,r,t)\in\mathcal{T}}\left(\mathrm{CrossEntropy}(\mathrm{P}_{h}^{T},\mathrm{L}_{h})+\mathrm{CrossEntropy}(\mathrm{P}_{t}^{T},\mathrm{L}_{t})\right),$$

## Co-distillation Learning
考虑到N-Former和N-BERT之间的互补性，我们建议通过知识蒸馏（也称为共蒸馏）来相互传递知识。在固定教师模型和待优化学生模型的情况下，传统知识蒸馏的损失是分类损失和Kullback-Leibler（KL）散度损失的组合，定义如下：
$$\mathcal{L}_{S}=\mathrm{CrossEntropy}(\mathrm{P}_{s},\mathrm{L})+\mathrm{K}\mathrm{l}(\mathrm{P}_{t}\mid\Vert\mathrm{P}_{s}),$$
$L_s$表示学生模型的知识蒸馏损失。Pt 和Ps 分别表示教师模型和学生模型预测的所有类的概率。L表示给定实例的标签向量。
对于N-Former和N-BERT，目标实体的得分在训练过程中都变得更高。
这是合理的，因为我们的两个模型都经过训练，将目标实体预测为top-1。然而，非目标实体的得分明显不同。**鉴于N-Former是基于结构的模型，而N-BERT是基于PLM的模型，我们认为非目标实体的得分对于量化模型知识也很重要**。因此，共蒸馏的关键是相互传递非目标实体的预测概率，而不仅仅是传递目标实体的概率。
最近，非目标类的重要性越来越受到关注。DKD将经典知识蒸馏损失分解为两部分，以释放非目标类中包含的知识潜力。基于此，我们还强调非目标实体对于KG嵌入的重要性。与具有固定教师的经典知识蒸馏场景不同，由于模型可以同时是教师和学生，因此不能保证知识进行共蒸馏。因此，我们设计了一种用于选择性知识转移的启发式方法。
对于不完整的三元组$(ℎ,𝑟, [MASK])$，我们得到预测Logit $P_𝑡^𝑆$ 和$P^𝑇_𝑡$ 分别来自N-Former和N-BERT。当N-Former是教师模型时，我们根据$P_𝑡^𝑆$按降序排列所有实体 并选择得分较高的半个实体。然后，从$P_𝑡^𝑆$ 和$P^𝑇_𝑡$选择实体概率$P_𝑡^{𝑆_1}$ 和$P^{𝑇_1}_𝑡$，解耦损失为：$$\mathcal{L}_{K D}\left(\mathbf{P}_{t}^{S_{1}}, \mathbf{P}_{t}^{T_{1}}\right)=\mathrm{KL}\left(\mathbf{b}_{t}^{S_{1}} \| \mathbf{b}_{t}^{T_{1}}\right)+\mathrm{KL}\left(\hat{\mathbf{P}}_{t}^{S_{1}} \| \hat{\mathbf{P}}_{t}^{T_{1}}\right)$$
其中的:${\bf b}_{t}^{S_{1}}=[p_{1,}1-p_{1}]\in\mathbb{R}^{1\times2},{\bf b}_{t}^{T_{1}}=[p_{2},1-p_{2}]\in\mathbb{R}^{1\times2}$表示目标实体的二元概率，可以认定p1和p2是来自N-Former和N-BERT的目标实体概率。$\hat{\mathbf{P}}_{t}^{S_{1}}$表示从$\mathbf{P}_{t}^{S_{1}}$除去目标实体的概率。同理，当N-BERT作为教师实体时，用同样的方式得到$\mathbf{P}_{t}^{S_{2}}$和$\mathbf{P}_{t}^{T_{2}}$。
## Put It All Together
使用互蒸馏学习后的损失为：
$$\begin{aligned}
(14)：
\mathcal{L}_{\mathrm{N} \text {-BERT }} &=\alpha \mathcal{L}_{K D}\left(\mathbf{P}_{t}^{S_{1}}, \mathbf{P}_{t}^{T_{1}}\right)+(1-\alpha) \text { CrossEntropy }\left(\mathbf{P}_{t}^{T}, \mathbf{L}_{t}\right), \\
\mathcal{L}_{\text {N-Former }} &=\beta \mathcal{L}_{K D}\left(\mathbf{P}_{t}^{T_{2}}, \mathbf{P}_{t}^{S_{2}}\right)+(1-\beta) \text { CrossEntropy }\left(\mathbf{P}_{t}^{S}, \mathbf{L}_{t}\right),
\end{aligned}$$

对于每个小批量，根据相同的数据计算损失，并根据损失分别更新两个模型。算法1给出了训练过程。在推理过程中，给定不完全三元组，我们通过加权平均将N-Former和N-BERT的预测概率组合为最终输出概率，对候选进行排序。
![算法](../../image/CoLE%2006.png)

# 知识补充
TODO:
**KL散度**: