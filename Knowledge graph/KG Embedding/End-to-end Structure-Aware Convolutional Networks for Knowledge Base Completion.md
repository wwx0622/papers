# title
[End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://arxiv.org/abs/1811.04441)
# Abstract
知识图嵌入是热门领域，从TransE, TransH, DistMult到最新的ConvE.ConvE使用了2D卷积和多层非线性特征建模知识图。然而，在ConvE的嵌入空间中没有强调结构。GCN通过利用图连通性结构提供了另一种学习图节点嵌入的方法。
本文中提出了一种新颖的端到端结构感知卷积网络（end-to-end Structure Aware Convolutional Network, SACN），它将GCN和ConvE结合起来。SACN由加权图卷积网络（WGCN）的编码器和卷积网络的解码器Conv-TransE组成。
- WGCN利用知识图节点结构、节点属性和边关系类型。它具有可学习的权重，可适应局部聚合中使用的邻居的信息量，从而实现更精确的图节点嵌入。图中的节点属性表示为WGCN中的附加节点。
- 解码器Conv-TransE使最先进的Conv能够在实体和关系之间平移，同时保持与ConvE相同的链路预测性能。

# Conclusion
介绍了SACN。编码网络是一个加权图卷积网络，利用知识图连接性结构、节点属性和关系类型。具有可学习权重的WGCN具有从相邻图节点收集自适应信息量的优点。此外，将实体属性添加为网络中的节点，以便将属性转换为知识结构信息，从而易于集成到节点嵌入中。SACN的评分网络是一种卷积神经模型，称为Conv-TransE。它使用卷积网络将关系建模为转换操作，并捕获实体和关系之间的转换特性。Conv-TransE达到了最优表现。SACN的性能总体上比现有技术（如Conv）提高了约10%。
在未来，我们希望将邻居选择思想纳入我们的框架，例如，重要性池，它在聚合邻居的向量表示时考虑了邻居的重要性。我们还希望扩展我们的模型，使其具有更大的知识图。
# Introduction
通用：(s,r,o)
知识图嵌入是最近知识库完成的一个活跃研究领域：它在连续的低维向量空间（称为嵌入）中编码实体和关系的语义。然后使用这些嵌入来预测新的关系。从一种称为TransE的简单有效方法开始，提出了许多知识图嵌入方法，如TransH、TransR、DistMult、TransD、ComplEx、STransE。一些调查给出了这些嵌入方法的细节和比较。
最新的ConvE模型使用嵌入和多层非线性特征上的二维卷积，并在知识图链接预测的常用基准数据集上实现了最先进的性能。在ConvE中，s和r的嵌入被重新整形并连接到输入矩阵中，并馈送到卷积层。n×n的卷积滤波器用于输出跨越不同维度嵌入条目的特征图。因此，ConvE没有保持TransE的平移特性，TransE是一种加法嵌入向量运算：$e_s+e_r≈ e_o$。在本文中，我们删除了ConvE的整形步骤，并直接在s和r的相同维度上操作卷积滤波器。与原始ConvE相比，这种修改提供了更好的性能，并且具有直观的解释，使嵌入三元组$（e_s、e_r、e_o）$中的s、r和o的全局学习度量保持相同。我们将这种嵌入命名为Conv-TransE。
ConvE也没有将知识图中的连通结构合并到嵌入空间中。相比之下，GCN是创建节点嵌入的有效工具，该嵌入可以聚集每个节点在图邻域中的局部信息.GCN模型还有其他好处，例如利用与节点相关的属性。他们还可以在计算每个节点的卷积时采用相同的聚合方案，这可以被视为一种正则化方法，并提高效率。虽然可扩展性最初是GCN模型的一个问题，但最新的数据高效GCN PinSage能够处理数十亿个节点和边缘。
在本文中，我们提出了一种端到端图结构感知卷积网络（SACN），它综合了GCN和ConvE的所有优点。SACN由加权图卷积网络（WGCN）的编码器和卷积网络的解码器Conv-TransE组成。
WGCN利用知识图节点结构、节点属性和关系类型。它具有可学习的权重来确定局部聚合中使用的邻居的信息量，从而实现更精确的图节点嵌入。节点属性作为附加属性添加到WGCN，以便于集成。WGCN的输出成为解码器Conv-TransE的输入。Conv-TransE与Conv相似，但不同之处在于Conv-TransE保持了实体和关系之间的平移特性。
我们证明了Conv-TransE的性能优于Conv，并且我们的SACN在标准基准数据集上比Conv-TransE进一步改进。我们的模型和实验的[代码](https://github.com/JD-AI-Research-Silicon-Valley/SACN)是公开的。
贡献：
- 我们提出了一个端到端的网络学习框架SACN，该框架同时利用了GCN和Conv TransE。编码器GCN模型利用图结构和图节点的属性。解码器Conv-TransE使用特殊卷积简化了ConvE，并保持了TransE的平移特性和Conv的预测性能
- 我们在标准FB15k-237和WN18RR数据集上证明了我们提出的SACN的有效性，并显示在以下方面比最先进的Conv相对提高了约10%HITS@1, HITS@3和HITS@10.

# Method
## Weighted Graph Convolutional Layer (WGCN)
WGCN在聚合时对不同类型的关系进行不同的加权，并且在网络训练期间自适应学习权重。通过这种自适应，WGCN可以控制来自聚合中使用的相邻节点的信息量。粗略地说，WGCN将多关系KB图视为多个单关系子图，其中每个子图都包含特定类型的关系(**注**：因为同类型的关系享有同一个权重，而不同关系有自适应的权重，因此KG被广义上分成了|E|个子图)。当组合节点的GCN嵌入时，WGCN确定给每个子图多少权重。

第l层WGCN：它将上一层每个节点的$F^l$维输出作为输入，并生成新的$F^{l+1}$维向量表示。
$h_i^l$:l层节点$v_i$的输入向量。所以${\cal H}^{l}\;\;\in\;\;\mathbb{R}^{N\times{F}^{l}}$是输入矩阵。
$H^1$：初始矩阵，通过高斯分布随机生成
如果有L层WGCN,则$H^{L+1}$为最终嵌入。
KB中有T类关系，E个边。节点间的强弱关系由关系类型决定，其参数为$\{\alpha_{t},1\,\leq\,{{t}}\leq\ T\}$，这个参数是自动学习的。

![SACN过程图](../../image/SACN%2001.png "SACN过程图")
WGCN的针对节点$v_i$计算方法：$h_{i}^{l+1}=\sigma\left(\sum_{j\in\bf{N}_{i}}\alpha_{t}^{l}g(h_{i}^{l},h_{j}^{l})\right)$,其中$g(x)$表示如何聚合邻居信息的函数。实际工作中使用的$g(h_{i}^{l},h_{j}^{l})\,=\,h_{j}^{l}W^{l},$其中${W}^{l}~\in \mathrm{R}^{{F ^ l}\times F^{l+1}}$。因为这样的g没有应用到节点$v_i$本身包含的信息$h_i^l$，因此修改传播过程为:$$h_{i}^{l+1}=\sigma\left(\sum_{j \in \mathbf{N}_{\mathbf{i}}} \alpha_{t}^{l} h_{j}^{l} W^{l}+h_{i}^{l} W^{l}\right)$$上述过程可以组织为矩阵乘法，通过邻接矩阵同时计算所有节点的嵌入。对于每种关系（边）类型，邻接矩阵$A_t$是一个二元矩阵，如果存在连接$v_i$和$v_j$的边，则其$A_{ij}$为1，否则为0。最后的邻接矩阵如下所示：$$A^{l}=\sum_{t=1}^{T}(\alpha_{t}^{l}A_{t})+I,$$单位阵对应本节点的信息。最终可以转化为矩阵运算：$$H^{l+1}=\sigma(A^{l}H^{l}W^{l}).$$
## Node Attributes
节点属性通常也以(entity, relation, attribute)的形式给出。单纯的将属性也视为边会有两个问题：
1. 每个节点的属性数量通常很小，并且各节点的属性不同。因此，属性向量将非常稀疏。
2. 属性向量中的零值可能具有模糊含义：节点没有特定属性，或者节点错过了该属性的值。这些零会影响嵌入的精度。

本文中，知识图中的实体属性由网络中另一组称为属性节点的节点表示。属性节点充当链接相关实体的“桥梁”。实体嵌入可以通过这些“桥”进行传输，以将实体的属性合并到其嵌入中。因为这些属性以三元组的形式出现，所以我们将属性表示为类似于实体o在三元组中的表示。请注意，每种类型的属性对应于一个节点。例如，在我们的示例中，性别由单个节点表示，而不是“男性”和“女性”的两个节点。这样，WGCN不仅利用了图连通性结构（关系和关系类型），而且有效地利用了节点属性（一种图结构）。这就是我们将WGCN命名为结构感知卷积网络的原因。
<font color="red">
疑问：
1. 每种类型的属性对应于一个节点，性别只有一个节点？
2. 什么是属性节点?


</font>
## Conv-TransE
Conv-TransE模型基于Conv但具有TransE平移特性的解码器：es+er≈ eo。区别在于，在堆叠es和er后没有重新成形。卷积核大小为2*k，上图的卷积核是2*3的。
在SACN的编码器中，关系嵌入的维数通常被选择为与实体嵌入的维数相同，即$F^L$。因此，两个嵌入可以堆叠。
对于解码器，输入是两个嵌入矩阵：一个来自WGCN的$N×F^L$矩阵用于所有实体节点，另一个$M\times F^L$用于关系嵌入矩阵，该矩阵也经过训练。由于我们使用了小批量随机训练算法，解码器的第一步对嵌入矩阵执行查找操作，以检索小批量中三元组的输入es和er。卷积过程为:$$\begin{aligned}
m_{c}\left(e_{s}, e_{r}, n\right)=& \sum_{\tau=0}^{K-1} \omega_{c}(\tau, 0) \hat{e}_{s}(n+\tau) \\
&+\omega_{c}(\tau, 1) \hat{e}_{r}(n+\tau)
\end{aligned}$$K是核的宽度，n是维度索引，$n\in [0, F^L-1]$；$\omega_c$是可训练的；$\hat{e}$是填充版本。如果核的维数s是奇数，除了中间的数字，其他用0填充，否则留下中间两个左边一个，留下的从e中选对应位置的元素。输出向量为：${\cal M}_{c}(e_{s},e_{r})\,\,\,\longrightarrow\,\,$$[m_{c}(e_{s},e_{r},\ 0),...,m_{c}(e_{s},e_{r},F^{L}\:-\:1)].$最后的输出为$\ M\big({\mathcal{C}}_{S},\,{\mathcal{C}}_{r}\big)\subset\mathbb{R}^{\mathcal{C}\times F^{L}}$.
评分函数为：$\psi(e_{s},e_{o})=f(v e c({\bf M}(e_{s},e_{r}))W)e_{o},$
预测概率为：$p(e_{s},e_{r},e_{o})=sigmoid(\psi(e_{s},e_{o})).$

总之，提出的SACN模型利用了知识图节点连通性、节点属性和关系类型。WGCN中的可学习权重有助于从相邻图节点收集自适应数量的信息。实体属性作为网络中的附加节点添加，并易于集成到WGCN中。Conv-TransE保持实体和关系之间的平移特性，以学习用于链接预测的节点嵌入。
其次，SACN在使用或不使用节点属性的情况下都比ConvE有显著的改进。

# 核心代码

```py
# 图卷积
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations+1, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, adj):

        alp = self.alpha(adj[1]).t()[0]
        # e1, rel, e2 -> 邻接矩阵，对应A(e1,e2)=rel,其他值为0
        # 对应文中的A
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2],adj[2]]), requires_grad = True)
        A = A + A.transpose(0, 1) # 对应文中的A
        support = torch.mm(input, self.weight) # 对应文中的H，W
        output = torch.sparse.mm(A, support) # AHW

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SACN:
    def __init__(self, num_entities, num_relations):
        super(SACN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.conv1 =  nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding= int(math.floor(Config.kernel_size/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim*Config.channels,Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)

        print(num_entities, num_relations)

    def farword(self, e1, rel, X, A):
        # e1, relation, 全1向量长N, A=[[e1, e2_multi], relation, N]
        # X: N -> N*init_embedding_size:100
        emb_initial = self.emb_e(X)
        # 图卷积：
        x = self.gc1(emb_initial, A) # x = AHW
        x = self.bn3(x)
        x = torch.tanh(x) # H' = s(AHW)
        x = F.dropout(x, Config.dropout_rate, training=self.training)

        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
        e1_embedded = e1_embedded_all[e1] # 实体嵌入
        rel_embedded = self.emb_rel(rel) # 关系嵌入
        # M(Cs, Cr)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1) # cat(es, er))
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1) # vec(x), batch*length
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0)) # 评分函数
        pred = torch.sigmoid(x)
```
# 总结
优点：
1. 针对不同关系设置了不同的**可学习的权重**
2. 通过**矩阵存储图结构**，提高运算速度

缺点：
1. 大型知识图，节点数和关系数很多，占用显存太大
2. 属性视作单独的节点，增加了节点的数量，导致矩阵的稀疏
3. 没有考虑实体的多重指代等
4. 关系通过embedding进行嵌入，无后续处理

改进思路：
1. 通过**自注意力机制**找到关系的权重关系
2. 属性可以作为节点的{key:value}对的集合，将所有key-value对进行嵌入，在图卷积时，除了要考虑节点，还要考虑属性。也即$h_i' = (\sum_{j=1}^{N_i}g(h_i,h_j))+(\sum_{j=1}^{A_i}f(h_i,attr_j))$
3. 节点有了天然的去重措施，也即比较节点的name和attribute

# 其他知识补充
知识图嵌入指标：HITS@1, HITS@3, HITS@10, MRR, MR
- **MRR**的全称是Mean Reciprocal Ranking，其中Reciprocal是指“倒数的”的意思。公式为:$$\mathrm{MRR}={\frac{1}{|S|}}\sum_{i=1}^{|S|}{\frac{1}{r a n k_{i}}}={\frac{1}{|S|}}({\frac{1}{r a n k_{1}}}+{\frac{1}{r a n k_{2}}}+\ldots+{\frac{1}{r a n k_{|S|}}})$$该指标越大越好。假如真实三元组(s,r,o)进行预测后的排名为rank,那么针对该三元组的预测得分为$\frac{1}{rank}$，然后把所有的预测三元组得分通过上述公式进行计算，得到MRR。
- **MR**的全称是Mean Rank。具体的计算方法如下：$$\mathrm{MR}={\frac{1}{|S|}}\sum_{i=1}^{|S|}r a n k_{i}={\frac{1}{|S|}}(r a n k_{1}+r a n k_{2}+\ldots+r a n k_{|S|})$$类似MRR计算，该指标越小越好。
- **HITS@n**是指在链接预测中排名小于等于n的三元组的平均占比。具体的计算方法如下：$$\mathrm{HITS@n}={\frac{1}{|S|}}\sum_{i=1}^{|S|}\mathbb{I}(r a n k_{i}\leqslant n)$$其中$\mathbb{I}(·)$表示指示器函数，真值得1，假值得0。该指标越大越好。

总结：MRR和HITS@n被认为是重要指标，其中n通常取1，3，10。MR则不被认为是好指标。