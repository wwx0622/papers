# title
《使用BERT做KGC》
[KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/abs/1909.03193)
[项目Github地址](https://github.com/yao8839836/kg-bert)
# Abstract
KG-BERT建议使用预训练的语言模型(pre-trained language model, PLM)来完成知识图(KGC)。
将知识图中的三元组视为文本序列，并提出了一种新的框架，称为知识图双向编码器表示变换器（KG Bidirectional Encoder Representations from Transformer, KG-BERT），以对这些三元组进行建模。该方法以三元组的实体和关系描述为输入，并使用KG-BERT语言模型计算三元组得分函数。

# Conclusion and Future Work
KG-BERT将实体和关系表示为它们的名称/描述文本序列，并将知识图完成问题转化为序列分类问题。KG-BERT可以利用大量自由文本中丰富的语言信息，突出显示与三元组相关的最重要单词。该方法在多个基准KG数据集上的表现优于最新的结果。
未来的一些方向包括通过使用**KG结构**联合建模文本信息，或使用具有更多文本数据（如XLNet）的预训练模型来改进结果。
将KG-BERT作为知识增强的语言模型应用于语言理解任务是我们将要探索的一项有趣的未来工作。

# Introduction
KGC: 旨在评估知识图中不存在的三元组的合理性。
最近的一些研究结合了文本信息来丰富知识表示：但他们学习了不同三元组中相同实体/关系的唯一文本嵌入，这忽略了上下文信息。例如，对"Steve Jobs"描述中的不同词语应具有不同的重要性权重，与“founded”和“isCitizenOf”这两个关系相关，“wroteMusicFor”可以有两种不同的含义“作词”和“作曲”赋予不同的实体。<font color='red'>这里想要表达的是实体或关系嵌入时的异义问题。</font>另一方面，大规模文本数据中的句法和语义信息没有得到充分利用，因为它们仅使用实体描述、关系提及或与实体的词共现。
最近，PLM如ELMo、GPT，BERT和XLNet在NLP方面取得了巨大成功，这些模型可以学习大量自由文本数据的上下文化单词嵌入，并在许多语言理解任务中实现最先进的性能。其中，通过掩蔽语言(ML)建模和下一句预测(next sentence prediction, NSP)对双向变换器编码器进行预训练的BERT取得最好的表现。它可以在预训练的模型权重中捕获丰富的语言知识。

KG-BERT: 具体来说，首先将实体、关系和三元组视为文本序列，并将KGC转化为序列分类问题。然后，我们对这些序列上的BERT模型进行微调，以预测三元或关系的似然性。该方法可以在多个KG完成任务中实现强大的性能，本文的贡献如下:
1. 我们提出了一种新的用于知识图完成的语言建模方法。据我们所知，这是第一次用PLM对三元组的似然性进行建模的研究。<font color='red'>一定程度上，算是基于PLM做KGC的开山之作</font>
2. 在几个基准数据集上的结果表明，我们的方法可以在triple classification、关系预测(relation prediction, RP)和链接预测(link prediction, LP)任务中实现最先进的结果。

# Method
**BERT**: BERT是一种基于多层双向Transformer Encoder的最优PLM, 变压器编码器基于自注意机制。BERT框架中有两个步骤：
预训练和微调。在预训练期间，BERT在大规模未标记的通用领域语料库上接受两项自我监督任务的训练：ML和NSP。在ML中，BERT预测随机[MASK]的输入标记。在NSP中，BERT预测两个输入句子是否连续。对于微调，用预训练的参数权重初始化BERT，并使用来自下游任务(如问答等)的标记数据对所有参数进行微调。
**KG-BERT**: 我们将实体和关系表示为它们的名称(Name)或描述(Description)，然后将名称/描述词序列作为BERT模型的输入语句进行微调。作为原始BERT，“句子”可以是连续文本或单词序列的任意跨度，而不是实际的语言句子。为了模拟三元组的似然性，我们将（h；r；t）的句子打包为单个序列。序列是指BERT的输入标记序列，可以是两个实体名称/描述语句或三个（h；r；t）组合在一起的语句。
整体结构: ![整体架构](../../image/KG-BERT%2001.png)
顾名思义: 该模型被称作为KG-BERT(a), 输入是以[CLS]开头，head和tail和relation都以name或者description的形式输入，部分与部分用[SEP]分割。每个token由corresponding token、segment token、position token组成。
- segment: 实体使用e嵌入，关系使用r嵌入，二者不同。
- position: 该嵌入与位置有关。
- corresponding: 单词的嵌入

最终输入$E_i$为上述三者加和。嵌入馈送至BERT中。最终输出为C, T。$C, T\in R^H$，H是隐藏层大小。C可用作计算最终的三元组分数。
**针对三元组分类**: 引入$W\in R^{2*H}$，引入评分函数: $\tau=(h,r,t)$，$s_\tau=f(\tau)=f(h,r,t)=sigmoid(CW^T)$,其中的$sum(s)=1$，分别表示正类或者负类的概率，因此可以得到损失函数:$$\mathcal{L}=-\sum_{\tau\in D+\cup D^-}(y_\tau log(s_{\tau0}+(1-y_\tau)log(s_{\tau1})))$$其中的$y_\tau$是表示三元组正负的标签。并且$D^-$是一个负三元组集合:$$\mathrm{D}^{-}=\{(h^{\prime},r,t)|h^{\prime}\in\mathbb{E}\wedge h^{\prime}\ne h\wedge(h^{\prime},r,t)\not\in\mathbb{D}^{+}\}\left.\cup\{(h,r,t^{\prime})|t^{\prime}\in\mathbb{E}\wedge t^{\prime}\not\to t\wedge(h,r,t^{\prime})\not\in\mathbb{D}^{+}\}\right.$$简单来说就是通过替换头尾使得相应的三元组不在正集合内即可。

**RP**:
整体架构:![整体架构](../../image/KG-BERT%2002.png)
容易理解，输入是head和tail的嵌入，用C预测关系。类似的打分函数$s_\tau'=softmax(CW'^T)$，这里的$W'\in \mathbb{R}^{|R|\times H}$，损失为:$$\mathcal{L}^{\prime}=-\sum_{\tau\in\mathbb{D}^{+}}\sum_{i=1}^{|R|}y_{\tau i}^{\prime}\log(s_{\tau i}^{\prime})$$

# 笔者总结
- 优点：
    1. 提出了应用PLM进行KGC的研究。
    2. 进行实体嵌入时输入的是name或description，更好的利用上下文信息。
- 缺点：
    1. 没有利用知识图的结构信息
    2. 上下文信息利用不足(?)，能否延长路径进行相关预测。
