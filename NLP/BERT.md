# 标题
[BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/abs/1810.04805)
# 摘要
BERT(Bidirectional Encoder Representations from Transformers)是一个预训练的双向Transformer模型。只需通过添加一个额外的输出层对BERT模型进行微调就可以创造一个新的模型以适用于不同的任务。
# 摘要
最近，语言模型迁移学习带来的经验改进表明，丰富、无监督的预培训是许多语言理解系统的组成部分。特别是，这些结果使即使是低资源任务也能从深层单向体系结构中受益。我们的主要贡献是将这些发现进一步推广到深层双向体系结构，从而允许相同的预先训练模型成功地处理广泛的NLP任务。
# 介绍
预训练的语言模型对很多自然语言处理任务提升较高。如句子级别的自然语言推理和解释——预测句子间的关系，或token级别的任务如NER和QA——需要生成token粒度的输出。
目前应用预训练语言表示于下游任务的有两个策略：
1. 基于特征(feature-based):如ELMo，使用包含预训练表示作为特征的特定任务结构
2. 微调(fine-tuning):如GPT，引入了少量特定任务所需参数，通过对预训练参数进行简单的微调用于下游任务。

该两种方法都是用了相同的目标函数，即使用单向语言模型学习通用语言表示。
现存的技术限制了预训练表示的能力，尤其是微调的。主要原因在于标准的语言模型是单向的，限制了预训练上的体系结构的选择性。如GPT，仅从左往右，在Transformers的自注意力层中，每个token仅会考虑之前的token，在句子级别表现尚可，在token级别的表示就不是很好了。
BERT使用MLM(masked language model)减少了单向注意约束的影响。通过MLM对输入的token进行随机覆盖，目标是基于上下文预测掩码单词的原始字典ID。MLM允许结合左右的语境，用于预训练一个深度双向Transformers。此外也能将MLM用于预测下个句子通过联合预训练文本对表示。以下是本文的贡献：
- 展示了应用于语言表示的双向预训练的重要性。BERT使用MLM允许与训练深度双向表示。这不同于独立的分别训练左到右和右到左两个模型再进行结合。
- 减少了对特定任务进行精心设计的需求。BERT是首个基于微调的表示模型并且在很多句子级别的和token级别的任务得到了最好的表现，比很多特定任务的结构表现更好。
- 在是一个NLP任务上取得了最好的结果。

# 相关工作
预训练生成语言表示有很久远的历史
## 基于特征的无监督方法
学习词汇的广泛表示成为了近年研究的热门领域，包括non-neural和neural方法。词嵌入对于嵌入式学习提供了显著的提升。
这些方法在句子嵌入和段嵌入生成粒度较大。为了训练句子表示，先前的工作对候选句子进行排序、根据上一句从左向右的生成下一句的单词、自动编码生成的目标去噪。
ELMo和他的前身生成传统的词嵌入用于不同的维度。他们提取**语境敏感**的特城从一个从左向右(LTR)和从右向左(RTL)的模型。上下文的表示由LTR和RTL的表示结合而成。在整合了几个现存特定任务结构的语境词嵌入后，取得了最好的结果。[context2vec: Learning generic context embedding with bidirectional LSTM](https://aclanthology.org/K16-1006.pdf)使用LSTM结合上下文预测单个词。
## 基于微调的无监督方法
最近的工作中，有句子或文档编码器从无标签文本和对有监督的下游任务的微调生成上下文的token表示。优点是只有少部分参数需要重新学习。因此GPT在句子级别任务中取得了最好的结果。LTR和自动编码可以用于预训练。
## 有监督数据的迁移学习
很多任务可以从有监督学习领域可以进行迁移学习，如自然语言推理与机器翻译。CV也显示迁移学习的重要性，例如对预训练的ImageNet进行微调。
# BERT
BERT模型分为**预训练(pre-training)**和**微调(fine-tuning)**两个部分。预训练期间模型在无标签数据上训练。微调需要把BERT模型用预训练的参数初始化，并进行微调以适应下游的有监督学习。不同的下游任务被微调的模型分类。
**模型结构**：定义Transformer块数为L，$d_{model}定义为H$，前馈层隐藏层大小$d_{hidden}=4H$，自注意力头数为A。
- 基础架构:L=12, H=768, A=12, Total Parameters(TP):110M
- 大型结构:L=24, H=1024, A=24, TP:340M

**输入/输出表示**：为适应不同的下游任务，输入表征需要明确是单个句子或者句子对。当然，句子可以任意长度的连续句子，而不是语言学上事实的一句话。从输入token序列得到的序列会被打包传送给BERT，而不论单格还是双个。序列的首个token是[CLS]。与该标记相对应的最终隐藏状态用作分类任务的聚合序列表示。句子对被打包为一个单个序列，但是需要token[SEP]进行分割，并且需要一个学到的嵌入表示每个token属于句子中的哪一个。如图所示，输入的embedding作为E，特殊token[CLS]被处理为一个隐藏向量$C\in R^H$，第i个输入被处理为隐藏向量$T_i\in R^H$。
![BERT架构](../image/BERT%2001.png "BERT架构")
并且每个给定的token都会组织为如图所示的token嵌入、语义嵌入、位置嵌入的和。
![input token embedding](../image/BERT%2002.png "input token embedding")
## 预训练BERT
**Task #1：Masked LM**：为了训练深层双向表示，随机对一定比例的输入token进行掩码，并预测掩码的token。在这种条件下，最后的对应掩码tokens的隐藏层向量输入到词典上的softmax层。实验中都是采用的15%。不同于自动编码的降噪，BERT仅预测掩码词汇而不是全部的输入。
尽管如此可以进行双向训练，但是会造成预训练和微调的不匹配，因为微调阶段没有[MASK]标记。为了克服该现象，训练数据生成器选择15%的token进行预测，并且如果$i-th$被选择到，将会以一定概率对其进行操作：
- 80%: 将$i-th$输入token替换为[MASK]
- 10%：将$i-th$输入token替换为随机token
- 10%：$i-th$不变

之后$T_i$可以用于预测原始token，损失函数使用交叉熵。
**Task #2: 下个句子预测(Next Sentence Prediction - NSP)**:
为了捕获句子间的关系，预训练一个二值化的**NSP**。对于句子样本A,B，每次输入的句子对有50%的概率是A-B(标签为IsNext)，还有50%是A-Random(标签NotNext)，Random表示为随机的一个语料库中的句子。在图1中，C是用于NSP的。
在之前只有句子嵌入可以转移到下游任务，但是BERT可以将所有参数用于初始化终端任务模型参数。语料为BooksCorpus和Wikipedia，Wikipedia仅提取文本段落而不取list、table等。文档级别的语料更有利于提取连续的长序列。
## 微调BERT
微调很简单，因为Transformer中的自注意力机制允许BERT对许多下游任务建模，无论是单个文本或文本对——通过交换适当的输入和输出。对于涉及文本对的应用，一种常见的模式是在应用双向交叉注意力机制之前对文本对进行独立编码。相反，BERT使用自注意力机制来统一这两个阶段，因为用自注意力编码串联的文本对有效地包括两个句子之间的双向交叉注意力。
对于每个任务，我们只需将特定于任务的输入和输出插入到BERT中，并端到端微调所有参数。在输入时，来自预训练的句子A和句子B类似于（1）释义中的句子对，（2）蕴涵中的假设前提对，（3）问答中的问题-段落对，以及（4）文本分类或序列标注中的的简化文本$-\empty$对。在输出时，token表示被馈送到token级任务的输出层，如序列标记或问答，而[CLS]表示被送到输出层进行分类，如蕴涵或情感分析。
与预培训相比，微调相对便宜。