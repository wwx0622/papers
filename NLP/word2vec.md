# 标题
[Distributed Representations of Words and Phrases
and their Compositionality](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)
# 摘要
Skip-gram模型是学习高质量分布向量表示的一种有效方法，它可以捕获大量精确的语法和语义词关系。
本文提出了扩展：提高学习向量的质量和训练速度。通过采样高频词汇能够得到显著的加速和更规则的词汇表示。并且提出了一个替代分层softmax的方式名为负采样。
# 结论
展示了通过Skip-gram训练词汇的分布式表示并且这种表示显示了线形的结构，有助于精确地逻辑推理。同样可以将该模型应用于训练bag-of-words模型。
该模型可以显著提高词汇表示的质量，尤其是稀疏实体，高频词汇采样提高了训练速度和非常见词汇的表示。提出了负采样——一种很简单的训练方法。
对模型表现影响最大的是：模型结构、向量长度、下采样率、训练窗口大小。
向量相加就可以实现有意义的单词组合。还有一个学习段表示的方法是用一个token表示段落。
# 介绍
Skip-grap模型：
![模型](../image/word2vec%201.png)
本文扩展了Skip-gram模型，使用下采样可以获得显著加速(2x-10x)。并提高了低频词汇的表达准确率。此外，提出了一种Noise Contrastive Estimation（NCE）方法，可以提高训练速度。
基于词向基于段的模型扩展是相对简单的，将段表示为一个token在训练过程中。
Skip-gram模型的另一个有趣之处在于两次相加也有含义，例如vec("Germany") + vec("capital") 相似于 vec("Germany capital")。
# Skip-gram模型
假定输入词汇是一个序列：$w_1, w_2, ..., w_T$，Skip-gram的目的是最大化平均对数概率：$\frac{1}{T}\sum_{t=1}{T}\sum_{-c\le j \le c, j \ne0}log p(w_{t+1}|w_t)$,$c$是训练是$w_t$的上下文窗口。
Skip-gram的定义$p(w_{t+j}|w_t)$基础公式使用softmax后如下：$p(w_O|w_I)=\frac{exp(v\prime_{w_O}{^T}v_{w_I})}{\sum_{w=1}{W}exp(v\prime_{w}{^T}v_{w_I})}$.其中的$v_w$和$v\prime_w$分别是词汇w的输入向量和输出向量。$W$是词汇表的大小。因为$W$太大，所以这个表是不切实际的。
## 分层softmax
分层softmax的效率与全softmax效率相近，但是评估的节点数仅有$log_2W$。
分层softmax将全部的$W$个单词组织为一颗B树。单词作为叶子，每个节点显式的表明子节点的相对概率。
每个单词$w$都可以被表示为从root出发的路径，令$n(w, j)$表示从root到$w$的路径上的第$j$个节点。$L(w)$表示从root到$w$的路径上的节点数。所以$n(w, 1) = root$且$n(w, L(w)) = w$。并且定义$ch(n)$为内节点$n$的任意一个固定孩子。并且令
$$
I(x)=
\begin{cases}
1, x为真 \\
-1, x不为真
\end{cases}
$$
那么分层softmax的$p(w_O|w_I)$的定义为：
$p(w|w_I)=\prod_{j=1}^{L(w)-1}\sigma(I(n(w, j+1)=ch(n(w,j)))*v\prime_{n(w,j)}{^T}v_{w_I})$,其中$\sigma(x)=\frac{1}{1+exp(-x)}$,并且分层softmax对每个$w$有一个$v_w$和对每个节点$n$有一个$v\prime_n$。
## 负采样
NEG定义为以下目标$$
log\sigma(v\prime_{w_O}{^T}v_{w_I})+\sum_{i=1}^{k}E_{w_i\sim P_n(w)}[log\sigma(-v\prime_{w_i}{^T}v_{w_I})]
$$
## 高频词汇下采样
针对大型预料集中包含大量的无意义词汇"a", "is"等，包含了很少的信息。为了应对稀疏词与高频词的不品恒，使用了一种下采样方式:
$P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}$,其中$f(w_i)$表示词汇$w_i$的频率$t$是选择阈值，通常为$10^{-5}$。
# 段学习
通过以下公式识别短语：
$$score(w_i, w_j)=\frac{count(w_iw_j)-\delta}{count(w_i)*count(w_j)}$$
其中$\delta$表示降低联合计数系数，因为对于稀疏词，可能仅有一次联合，通过该系数减少类似的组合。
# 相加组合性
两个单词的相加可以表现出一定的语义，这种可相加型可以从训练目标中得到解释。词汇通常是由线性转化到softmax的非线性，它倾向于寻找附近的词汇，所以向量可以被视为表征分布在单词周围的上下文，两个向量相加表示同时搜寻两个词汇周围上下文。这个逻辑有点类似于AND：被两个词向量赋予高概率的词将具有高概率，而其他词将具有低概率。例如"Volga River"更常出现在'Russian'和'river'的附近，所以两个向量的和与"Volga River"的向量更接近。