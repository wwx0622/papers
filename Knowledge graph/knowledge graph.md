# 标题knowledge graph
地址：[Knowledge Graph](https://arxiv.org/abs/2003.02320)

# 1 介绍
与其他NoSQL模型不同，专门的图形查询语言(GQL)不仅支持标准关系运算符（SQL操作符），还支持导航(**navigational**)运算符，用于递归查找通过任意长度路径连接的实体。标准的知识表示形式——比如本体和规则——可以用来定义和推理用于标记和描述图中节点和边的术语的语义。可扩展的图形分析框架可用于计算中心性、聚类、汇总等，以获得对所描述领域的见解。还开发了各种表示法，支持直接在图上应用机器学习技术。<br>
该论文主要通过以下几个方面介绍：

- 描述它们的基本数据模型以及如何查询它们
- 讨论与图式、身份和语境有关的表征
- 讨论演绎法和归纳法，使知识显性化
- 介绍可用于创建和丰富图形结构数据的各种技术
- 描述如何识别知识图的质量，以及如何对其进行细化
- 讨论发布知识图表的标准和最佳实践；并对实践中发现的现有知识图进行概述。

下面给出KG的一部分定义：
1. 数据图（data graph）符合基于图的数据模型，可以是有向边标记图、属性图等。
2. 知识，我们指的是已知的事物。这些知识可以从外部来源积累，也可以从知识图本身提取。
3. 简单语句可以在数据图中累积为边。

多源是KG的一个特点：一般通过schema、identity和context进行表示：
1. schema：描述知识的结构，比如本体、规则、关系等。
2. identity：描述知识的特征，比如名称、类型、属性等。
3. context：描述知识的语境，比如关系的条件、关系的关联等。

知识图谱有开源的和商业的：
- 开源：DBpedia, Freebase, Wikidata, YAGO等。
- 商业：Web search(Bing, Google), commerce(Airbnb, Amazon, eBay), social network(Facebook, LinkedIn), finance(Accenture, Banca d'Ttalia), etc.

# 2 数据图
## 2.1 Model
### 2.1.1 有向标签边图(Directed edge-labeled graph)
一个标准的基于有向标签边图(DELG)数据模型是Resource Description Framework(RDF)。RDF定义了不同类型的节点，包括 支持实体识别的Internationalized Resource Identifiers (IRIs)、表示字符串的字面常量(literals)、和其他类型（整型等），还有空白节点。
![fig 1](../image/Knowledge%20Graph%201.png "初始图")
### 2.1.2 异构图(Heterogeneous graphs)
异构图通常是指边和节点都有类型。一般也可以令边的类型=标签。DEL图把类型也作为一个节点，通过一条名为type的边从节点引向类型节点。异构图通常将类型作为节点的一部分。
![del graph和Heterogeneous graphs](../image/Knowledge%20Graph%202.png "del图和异构图")
**同构边**指边相接的的两个节点类型相同(上图的<font color=blue>borders</font>)。
**异构边**指其余的边(上图的<font color=blue>capital</font>)。
2.1.3 属性图(property graph)
将节点或关系的属性作为模式的一部分。
![del graph和Heterogeneous graphs](../image/Knowledge%20Graph%204.png "属性图")
### 2.1.4 图数据集(Graph dataset)
图形数据集是命名图和默认图的集合，为了管理多源的图。
## 2.2 查询
### 2.2.1 图模式
模式是图查询语言的核心。
用于评价模式的语义有：
- 同态：允许多个变量映射到同一术语(匹配时可以匹配到同一个关系或者节点多次)，如SPARQL采用同态。
- 同构：在节点或边上有唯一性约束，节点或者边只能出现一次，如Cypher采用关系同构。
### 2.2.2 复杂图模式
图模式可以将图转化成表，然后在在表上应用关系代数。  
### 2.2.3 导航图模式
图查询语言处理路径表达式的能力。路径表达式是用来匹配节点之间的任意长度路径的，一般表示为query(x, r, y)，其中x和y是节点，r是常量，一般为边的标签。
# 3 模式，同一性，语境
## 3.1 模式
### 3.1.1 语义模式
语义模式定义了更高层次的术语含义以用于图。
高级语义：
- class：可以对于数据自然分组，例如Food Festival可能是Event或Festival的子类。
- property：一些类别标签可以作为一个属性的子属性，如<font color=blue>city</font>和<font color=blue>venue</font>都是<font color=blue>location</font>的子属性。
- domain：一类节点通过一种属性连接另一类节点。揭示从某属性的关系的头节点的类别。$ p-{\color{blue}domain}\to c $意味着$ x-{\color{blue}{p}}\to y \quad implies \quad x-{\color{blue}{type}}\to c $
- range: 类似domain，不过是尾结点。$ p-{\color{blue}range}\to c $意味着$ x-{\color{blue}{p}}\to y \quad implies \quad y-{\color{blue}{type}}\to c $
### 3.1.2 验证模式
在大规模的异源的、不完整的数据上用图时，OWA是一个最恰当的方法对默认的语义来说。
一个定义验证模式的标准方法是使用形状(shape)，shape把节点集合作为目标并在其上声明**约束**.
### 3.1.3 涌现模式(emergent schema)
其中的一个框架是商图(quotient graphs)，它将数据节点进行紧凑。![del graph和Heterogeneous graphs](../image/Knowledge%20Graph%2014.png "涌现模式")
## 3.2 同一性
节点可能有歧义，如苹果可以指水果，也可以指代苹果公司。当没有更多的信息时，在很多困难的情形下，节点**消岐**可能需要启发式的算法，并且可能产生错误。为了消除节点歧义，可能需要全局唯一标识避免命名冲突以及增加额外的身份链接消除节点与外部源之间的差异。
### 3.2.1 永久标识
多源图做并集合并对数据进行建模，但是会产生**命名冲突**，为了避免这种冲突，可以创建持久标识符(long-lasting persistent identifiers)（PIDs），以便唯一地标识实体,例如文献的DOI，作者的ORCID，书籍的ISBN等。
### 3.2.2 外部身份连接
假如一个节点可以通过IRI定义命名空间，那么一个节点可以转移到对应的服务器中。例如定义`chile:`命名空间的IRI为`http://turismo.cl/entity`，那么节点`Santiago`的格式为`chile:Santiago`并自动连接到对应网站`http://turismo.cl/entity/Santiago`。但是不同源的对于节点的外部链接格式不一致，可能会导致知识图合并时无法识别相同节点。如`geo:SantiagoDeChile`表示某地理数据库的Chile的Santiago城市节点，但是两个知识图合并时会将其视为不同的节点。
避免这种情况可以通过：
1. 引入更多节点唯一标识信息如地理坐标等
2. 声明节点在其他源有相同的节点，例如`chile:Santiago`$-{\color{blue}{owl:sameAs}}\to$`geo:SantiagoDeChile`
### 3.2.3 数据类型
针对日期、整数等节点不需要引入外部变量。在RDF中使用XML Schema Datatypes(XSD)给定一个pair(l, d)表示节点，l(literals)是字符串，d是IRI生命的数据类型包括xsd:string, xsd:integer等等。如时间节点`"2020-03-29T20:00:00"^^xsd:dateTime`。
### 3.2.4 词汇化
使用一个计算机通用定义一个节点来标识真实世界的实体，然后在通过标签将该节点解释为人类所能明白的语言。如果wikidata定义`Santiago`为`wd:Q2887`，可以通过`wd:Q2887`$ -{\color{blue}{rdfs:label}}\to $`"Santiago"`将该节点解释为人能看得懂词汇。这其中的`Santiago`表示一个字符串，而不是标识，因此可以用pair(s, l)对其进行更详尽的定义，其中s是字符串，l表示一种语言。如`chile:City`$-{\color{blue}{rdfs:label}}\to$`"City"@en`.
### 3.2.5 存在节点
在对不完整信息建模时，在某些情况下，我们可能知道图中必须存在与其他节点具有特定关系的特定节点，但无法识别所讨论的节点。
例如，我们可能有两个位于智利的合办活动：`chile:EID42`和`chile：EID43`，其地点尚未公布。因此，一些图模型允许使用存在节点，也即一个空白节点：`chile:EID42`$-{\color{blue}{chile:venue}}\to$`None`$\gets{\color{blue}{chile:venue}}-$`chile：EID43`。
## 3.3 语境
### 3.3.1 直接表示
1. 通过数据与数据的不同表示语境。
2. 将关系表示为节点，然后在节点上附加额外信息。

### 3.3.2 具体化
通过一个额外的节点插入到关系中，对关系进行说明，对关系的其他信息进行建模。![三种具象化的方式](../image/Knowledge%20Graph%2018.png)
### 3.3.3 高参数表示
是具象化的另一种选择。
1. 通过一个命名图包含该关系，然后引出额外的关系进行说明。
2. 直接赋予关系额外属性
3. 使用扩展RDF(即RDF*)，边也会被视为节点。

三种方式的灵活度依次降低。
![高参数表示](../image/Knowledge%20Graph%2019.png)
### 3.3.4 注释
一些注释会对特定的上下文进行建模，例如时序RDF会为关系提供一个时间段，模糊RDF会为关系提供一个真实度。
其他的注释是域依赖的，如注释RDF允许表示建模为半环的各种形式的上下文：由域值（例如，时间间隔、模糊值等）和两个主要操作符（meet和join）组成的代数结构，以组合域值。
### 3.3.5 其他语境框架
其他上下文框架用于建模或解释图的上下文。例如**上下文知识库**，允许将个体的图或子图加入到自己的上下文中。
# 4 知识推理
给定一些数据作为前提，将一些常用规则作为**先验**，可以通过推理生成一些新的数据。这些人们共有的常用的前提和规则被称为**常用知识**。相对的，当只有少量某些领域的专家共有的基础知识被称作**域知识**，
机器缺少**先验**知识，需要通过前提和**牵连制度**做相似的推理。例如$x$有一个场所$y$，$y$在城市$z$中，那么通过设置牵连制度得出$x$位于$z$中。同时，可以根据$3.1.1$中的语义模式的domain和range或者sub-class等进行简单的推理。
## 4.1 本体论
本体论一般使用一个约定俗成的规则对一个术语进行定义。
### 4.1.1 解释
作为人来说可以将节点`Santiago`理解为现实中Chile的首都。可以把这样的数据图称之为**域图(domain graph)**。它由现实世界的真实实体所构成。这种演绎，要将数据图中的节点与关系**映射**到域图中的节点与关系。
数据图上的**演绎(interpretation)**由域图和一个将数据图的 **术语(terms)**向域图的映射。
### 4.1.2 个体
![本体特征](../image/Knowledge%20Graph%20t3.png "本体特征")
### 4.1.3 属性
![OWL扩展](../image/Knowledge%20Graph%20t4.png "OWL扩展")
### 4.1.4 类
![OWL扩展类别](../image/Knowledge%20Graph%20t5.png "OWL扩展类别")
### 4.1.5 其他特征
OWL还支持**注释属性(annotation properties)**，可以提供元数据。**数据类型**和**对象属性**，区别在于是否使用数据类型的值。
## 4.2 语义与推演
如$4.1.3$中表所示的，那些条件帮助了推演。如定义`nearby`$-{\color{blue}{type}}\to$`Symmetric`和边`Santiago`$-{\color{blue}{nearby}}\to$`Santiago Airport`，那么可以推理出边`Santiago Aorport`$-{\color{blue}{nearby}}\to$`Santiago`。
### 4.2.1 模型轮语义学
前几张表中描述的每一条公理，当添加到一个图中时，都会对**满足**该图的解释施加一些条件。满足图的解释称为图的**模型**。但是优势新加的公理会打破原有模型。例如`x`${-{\color{blue}{y}}\to}$`z`和`y`$-{\color{blue}{type}}\to$`Irreflexive`，那么`a`$-{\color{blue}{a}}\to$`a`,x=y=z=...=a会打破Irreflexive（非自反射）公理——节点的值域不能是自己。
### 4.2.2 蕴含
一个图蕴含另一个图意味着前一个图的所有模型也是后一个图的。直觉上来说，后一个图的对于前一个图来说没有新的东西。例如前一个图中有`x`${-{\color{blue}{type}}\to}$`y`$-{\color{blue}{subc.\quad of}}\to$`z`，后一个图中有`x`$-{\color{blue}{type}}\to$`z`。前一个图蕴含后一个图意味着前一个图中中所有的如上的模型，在后一个图中也有一个满足条件的对应模型。
### 4.2.3 if-then vs. if-and-only-if 语义
if-then:如果node匹配，则条件成立
if-and-only-if:当且仅当条件成立，node匹配
## 4.3 推理
根据先前的蕴含概念以及几个表给出的本体特征，无法在有限的解法定义两个图是否蕴含。
### 4.3.1 规则
通过**推理规则**（或简单规则）就一个编码if-then风格的结果。规则由身体（if）和头部（then）组成。身体和头部都以图形模式给出。
一条规则表明，如果我们可以用数据图中的术语替换身体的变量，并形成给定数据图的子图，那么在头部中使用相同的变量替换将产生有效的蕴涵。头部通常必须使用正文中出现的变量子集，以确保结论中没有未替换的变量。这种形式的规则对应于数据库中的（正）数据日志、逻辑编程中的Horn子句等
### 4.3.2 描述逻辑(Description Logics)
DL用于形式化框架和语义网的含义。DL基于三个元素：个体如`Santiago`、类如`City`、属性如`flight`。Assertional axioms(断言公理)可以用于一元操作如`City(Santiago)`和二元操作`flight(Santiago, Arica)`
# 5 知识归纳(INDUCTIVE KNOWLEDGE)
## 5.1 图分析
分析是发现、解释和交流（通常是大型）数据收集所固有的有意义模式的过程。图分析是应用于图数据的。
### 5.1.1 技术
图形分析可以应用多种技术。
![图24](../image/Knowledge%20Graph%2024.png)
- **中心性**：旨在识别图中最重要的（也称为中心）节点或边。具体的节点中心性度量包括度、介数、贴近度、特征向量、PageRank、HITS、Katz等。介数中心性也可以应用到边。
- **簇检测**：旨在识别图中的簇，即内部连接比图中其他部分更紧密的子图。簇检测算法，如最小割算法、标签传播、Louvain模块化等。
- **连通性**：旨在估计图形的连通性，例如，揭示图形元素的弹性和（非）可达性。具体技术包括测量图形密度或𝑘-连通性、检测强连通组件和弱连通组件、计算生成树或最小割集等。在图24中，此类分析可能会告诉我们，通往`Grey Glacier`、`Osorno Volcano`和`Piedras Rojas`的路线最“脆弱的”，如果两条公交线路中的一条发生故障，就会断开连接。
- **节点相似性**：旨在通过节点在其邻域内的连接方式找到与其他节点相似的节点。节点相似性度量可以使用结构等效、随机游动、扩散核等进行计算。这些方法可以了解节点之间的联系，以及节点之间的相似性。
- **路径查找**：旨在查找图形中的路径，通常在作为输入给定的成对节点之间。存在限制此类节点之间有效路径集的各种技术定义，包括不两次访问同一节点的简单路径、访问最少边数的最短路径，或如第2.2节所述，限制路径可遍历边标签的常规路径查询。
### 5.1.2 框架
已经提出了各种用于大规模图形分析的框架，通常是在分布式（集群）环境中。其中我们可以提到Apache Spark（GraphX）、GraphLab、Pregel、Signal–Collect、Shark等。这些图形并行框架应用了基于有向图的**脉动抽象**，其中节点是可以沿边缘向其他节点发送消息的处理器。然后，计算是迭代的，在每次迭代中，每个节点读取通过内部边缘接收的消息（可能还有它自己以前的状态），执行计算，然后根据结果通过外部边缘发送消息。然后，这些框架在正在处理的数据图上定义了脉动计算抽象：数据图中的节点和边成为脉动图中的节点和边。
![图24](../image/Knowledge%20Graph%2024.png)
举个例子，假设希望计算图24所示路线最容易（或最不容易）到达的位置。衡量这一点的一个好方法是使用中心度，选择PageRank，它计算出游客在给定数量的“跳跃”后沿着图中随机路线到达特定地点的概率。可以使用图并行框架在大型图上实现PageRank。在图25中，我们为图24的说明性子图提供了PageRank迭代的示例。节点初始化为$\frac{1}{V}=\frac{1}{6}$，假设游客在任何时候都有同等的机会出发。在**消息阶段（MSG）**，每个节点$v$通过分数$\frac{\mathrm{d} R_i(v)}{|E(v)|}$在其每个输出边上，其中我们表示为$𝑑$用于确保收敛的恒定阻尼系数（通常为𝑑 = 0.85，表示游客随机“跳”到任何地方的概率），$R_𝑖 (𝑣)$计算第$i$次迭代中节点𝑣的得分（经过$i$跳后游客在节点$v$的概率），以及$|𝐸(𝑣)|$表示$v$的出边。在聚合阶段（AGG），对于节点$v$，将接收到的所有传入消息——$\sum\frac{\mathrm{d} R_i(v)}{|E(v)|}$，和阻尼因子的恒定份额——$\frac{1-d}{|V|}$ ，进行求和来计算$R_{i+1}(v)$。然后，我们继续进行下一次迭代的消息阶段，直到达到某种终止标准（例如，迭代计数或剩余阈值等），并输出最终分数。

虽然给定的示例是针对PageRank的，但脉动抽象足够通用，可以支持各种各样的图形分析，包括前面提到的那些。该框架中的算法由计算消息阶段（Msg）中的消息值和在聚合阶段（Agg）中累积消息的函数组成。该框架将负责分发、消息传递、容错等。然而，这种基于邻居之间消息传递的框架具有局限性：并非所有类型的分析都可以在这种框架中表达。因此，框架可能允许其他功能，例如在所有节点上执行全局计算的**全局步骤**，将结果提供给每个节点；或者是允许在处理过程中添加或删除节点和边的**突变步骤**。
### 5.1.3 数据图分析
如上所述，迄今为止提出的大多数分析“本机”形式，适用于无边元数据的无向或有向图，即边标签或属性-值对，这是典型的图数据模型。
- **投影**涉及简单地“投影”一个无向或有向图，方法是从数据图中选择一个子图，从中删除所有边缘元数据；例如，图25可能是从一个较大的数据图中提取由边缘标签总线和航班诱导的子图的结果，然后删除标签以创建一个有向图。
- **加权**包括根据某些函数将边缘元数据转换为数值。上述许多技术很容易适应加权（有向）图的情况；例如，我们可以考虑图25中表示行程持续时间（或价格、交通等）的图上的权重，然后计算最短路径加上每条航段的持续时间。在没有外部权重的情况下，我们可以根据一些标准将边缘标签映射到权重，将相同的权重分配给所有航班边缘、所有巴士边缘等。
- **转换**涉及将图形转换为较低的arity模型。转换可能有损，这意味着无法恢复原始图形；或无损，这意味着可以恢复原始图形。图26提供了从有向边标记图到有向图的有损和无损转换示例。例如，在有损变换中，我们无法判断原始图是否包含边`Iquique`-flight-`Santiago`等。无损变换必须引入新节点（类似于物化），以保持有关有向标记边的信息。两个变换图都进一步尝试保持原始图的方向性。
- **定制**涉及更改分析过程以合并边缘元数据，例如基于路径表达式的路径查找。其他示例可能包括节点相似性的结构度量，这些度量不仅考虑公共邻居，还考虑由具有相同标签的边连接的公共邻居，或者聚集中心度度量，这些度量捕获按标签分组的边的重要性，等等。

### 5.1.4 查询分析
如$2.2$节所述，已经提出了各种查询图的语言。人们可能会考虑查询语言和分析可以相互补充的各种方式。首先，我们可以考虑使用查询语言来投影或转换适合特定分析任务的图，例如从较大的数据图中提取图24的图。SPARQL、Cypher和G-CORE等查询语言允许输出图，这些查询可用于选择子图进行分析。这些语言还可以表示一些有限的（非递归）分析，例如，可以使用聚合来计算度中心度；它们还可能具有一些内置的分析支持，例如，Cypher允许查找最短路径。在另一个方向上，分析有助于优化查询过程，例如，连接性分析可以建议如何更好地将大型数据图分布在多台机器上，以便使用最小切割进行查询。分析还被用于对大型图上的查询结果进行排名，选择最重要的结果，以向用户展示。
在某些用例中，可能还希望交错查询和分析过程。例如，从旅游局收集的完整数据图中，考虑一次即将到来的航空公司罢工，该局希望在罢工期间发现事件，因为罢工，圣地亚哥的公共交通无法到达城市的场馆。假设，我们可以使用查询来提取不包括航空公司路线的运输网络（假设，根据图3，航空公司信息可用），使用分析来提取包含圣地亚哥的强连接组件，最后使用查询来查找给定日期不在圣地亚哥组件中的城市中的事件。虽然可以使用命令式语言（如Gremlin、GraphX或R）来解决这一任务，但也在探索更多的声明性语言，以更容易地表达这类任务，其建议包括扩展具有递归功能的图查询语言，将线性代数与关系（查询）代数相结合，等等。
### 5.1.5 蕴含分析
知识图通常与定义领域术语语义的语义模式或本体相关联，从而产生蕴涵。应用有或无此类要求的分析可能会产生截然不同的结果。
然而，从考虑边缘方向的分析技术的角度来看，这些边缘远非等效的，因为包括一种类型的边缘或另一种类型的边缘，或两者都可能对最终结果产生重大影响。据我们所知，分析和蕴涵的结合还没有得到很好的探索，留下了许多有趣的研究问题。沿着这些思路，可能有兴趣探索语义不变的分析，这些分析在语义等价图（即相互关联的图）上产生相同的结果，从而分析知识图的语义内容，而不仅仅是数据图的拓扑特征；
## 5.2 知识图嵌入
近年来，机器学习方法受到了广泛的关注。在知识图的上下文中，机器学习可用于直接精炼知识图（在第8节中进一步讨论）；或者对于使用知识图的**下游任务**，如推荐、信息提取、问答、查询松弛、查询近似等（在第10节中进一步讨论）。那么，如何将图形或其节点、边等编码为数字向量呢？
one-hot:为每个长度节点生成一个长度为$|L|*|V|$的向量——$|V|$表示输入图形中的节点数，$|L|$表示边标签数。对应索引位置表示边的存在性。

知识图嵌入技术的主要目标是在连续的低维向量空间中创建图的密集表示（即嵌入图），然后将其用于机器学习任务。嵌入纬度d是固定的，通常较低（通常为1000 ≥ 𝑑 ≥ 50）。通常，图嵌入由每个节点的**实体嵌入**组成：一个d维向量，被称为**e**；以及每个边标签的**关系嵌入**：一个d维向量，被称为**r**。给定一条边$s-p\rightarrow o$，一种特定的嵌入方法定义了一个**得分函数**，该函数接受$\mathbf{e_s}$（节点s的实体嵌入）、$\mathbf{r_p}$（边标签p的实体嵌入）和$\mathbf{e_o}$（节点o的实体嵌入），并计算边的合理性。给定一个数据图，目标根据评分函数计算一个d维嵌入，它能够最大化正边（通常是图中的边）的合理性，并最小化负示例（通常是图中的边，节点或边标签已更改，因此它们不再在图中）的合理性。由此产生的嵌入可以被视为通过自我监督学习的模型，对图的（潜在）特征进行编码，将输入边映射到输出合理性分数。

然后，嵌入可以用于许多低级任务，这些任务涉及从中计算它们的图的节点和边标签。首先，可以使用似然性评分函数为可能。其次，为了链接预测的目的，合理性评分函数可用于完成缺少节点/边缘标签的边缘。第三，嵌入模型通常会将类似向量分配给类似节点和类似边缘标签，因此它们可以用作相似性度量的基础，这可能有助于找到引用相同实体的重复节点，或用于推荐。
人们提出了一系列知识图嵌入技术[549]。首先讨论了采用几何视角的平移模型，其中关系嵌入将低维空间中的主体实体转换为客体实体。然后，述了张量分解模型，该模型提取出接近图结构的潜在因子。此后，我们讨论了使用神经网络来训练嵌入的神经模型，这些嵌入可以提供准确的合理性得分。最后，讨论了利用现有单词嵌入技术的语言模型，提出了为预期（文本）输入生成类似于图形的类比的方法。

### 5.2.1 翻译模型
**翻译模型**将边标签解释为从subject节点（又名**源节点**或**头部**节点）到object节点（又名**目标节点**或**尾部节点**）的转换；例如，在`San Pedro`-`bus`$\rightarrow$`Moon Valley`中，`bus`被视为将`San Pedro`转化为`Moon Valley`，同样，对于其他总线边缘也是如此。这个家族中最基本的方法是**TransE**。在所有正边缘$s-p\rightarrow o$上，TransE学习向量$\mathbf{e_s}$、$\mathbf{r_p}$和$\mathbf{e_o}$，目的是使$\mathbf{e_s}+\mathbf{r_p}$尽可能接近$\mathbf{e_o}$。相反，如果边缘是一个负面的例子，TransE尝试学习一种使$\mathbf{e_s}+\mathbf{r_p}$远离$\mathbf{e_o}$的表示法。

除了这个玩具的例子外，TransE可能过于简单化；如果同类型的边有多个尾结点，TransE的目标是为所有此类目标位置提供类似的向量，这在其他条件时可能不可行。TransE也会倾向于将周期关系(cyclical relation，环？)分配为零向量，因为方向分量会倾向于相互抵消。为了解决这些问题，研究了TransE的许多变体。例如，其中**TransH**使用不同的超平面表示不同的关系，其中对于边$s-p\rightarrow o$，在学习到o的翻译之前，s首先投影到p的超平面上（不受带s和o的其他标签的边的影响）。**TransR**通过将s和o投影到特定于p的向量空间来推广这种方法，这涉及到将s和o的实体嵌入乘以特定于p的投影矩阵。**TransD**通过将实体和关系与次级向量(第二个向量)相关联来简化TransR，其中这些次级向量用于将实体投影到特定于关系的向量空间。最近，**RotatE**提出了复杂空间中的平移嵌入，这允许捕获关系的更多特征，例如方向、对称、反转、反对称和合成。在非欧几里德空间中也提出了嵌入，例如，**MuRP**使用关系嵌入来转换Poincaré ball（庞加莱球）模式的双曲空间中的实体嵌入，其曲率提供了更多的“空间”来分离实体。关于其他平移模型的讨论，我们参考Wang等人的调查。

### 5.2.2 张量分解模型
推导图嵌入的第二种方法是应用基于张量分解的方法。张量是将标量（0阶张量）、向量（1阶张量）和矩阵（2阶张量）推广到任意维/阶的多维数值字段。张量已成为机器学习中广泛使用的抽象概念。张量分解涉及将张量分解为更多的“基本”张量（例如，低阶张量），从中可以通过固定的基本操作序列重新组合（或近似）原始张量。这些元素张量可以被视为捕捉原始张量中包含的信息背后的潜在因素。张量分解有很多方法，现在我们将简要介绍秩分解背后的主要思想。
暂时撇开图形不谈，考虑一个二维张量$C_{a*b}$，其中$𝑎$ 是智利的城市数量，$𝑏$ 是一年中的月数，$C(i, j)$表示第$i$个城市第$j$个月的平均温度。注意到智利是一个又长又瘦的国家——从南部的亚极地气候到北部的沙漠气候——我们可能会发现$C$分解为两个代表潜在因子的向量–特别是$x$（带$𝑎$元素）为纬度较低的城市提供较低的值，$y$（具有$𝑏$元素）在温度较低的月份给出较低的值——因此计算两个向量的外积相当接近$C$:$x\otimes y\approx C$、 在（不太可能）的情况下，存在向量$x$和$y$，$C=x\otimes y$ 我们称C为rank-1矩阵；然后我们可以使用$𝑎 + 𝑏$而非$𝑎 × 𝑏$编码$C$，大多数时候，为了精确地得到C，我们需要对多个rank-1矩阵求和，其中$C$的秩$𝑟$是需要求和以精确导出$C$的rank-1矩阵的最小数目，例如$x_1\otimes y_1+...+x_r\otimes y_r=C$、 在温度示例中，$x_2⊗ y_2$可能对应高度修正值,$x_3⊗ y_3$表示南方有更高的温度。矩阵的（低）秩分解然后设置一个极限$𝑑$ 并计算向量$(x_1，y_1，…，x_𝑑, y_𝑑)$这样$x_1⊗ y_1+…+x_𝑑 ⊗ y_𝑑$是C的最好的rank-d近似。这种方法被称为典型多元（CP）分解。例如，我们可能有一个三阶张量$C$，其中包含智利城市一天中四个不同时间的月温度，可以近似为$x_1⊗ y_1⊗ z_1+...x_𝑑 ⊗ y_𝑑 ⊗ z_𝑑$（例如，$x_1$可能是纬度因子，$y_1$可能是月变化因子，$z_1$可能是日变化因子)。然后存在各种算法来计算（近似）$CP$分解，包括交替最小二乘法、$Jennrich$算法和张量幂法。

针对图:每个图分成一个三维向量：
$\mathbb{G}(i, j, k)=
\left\{\begin{matrix}
 1 & i(subject)-j(edge)\rightarrow k(object)\\
 0 & ohterwise
\end{matrix}\right.$ 
这个张量大且稀疏，分解为$x_1⊗ y_1⊗ z_1+...x_𝑑 ⊗ y_𝑑 ⊗ z_𝑑$,令$X=[x_1,...,x_d]$，另外俩类似。把$Y_i$作为$i^{th}$关系的嵌入，$X_j$和$Z_j$作为$j^{th}$实体的两个嵌入。

**DistMult**是一种基于秩分解计算知识图嵌入的开创性方法，其中每个实体和关系都与一个d维向量相关联, 对于边$s-p\rightarrow o$，合理性评分函数$\sum^{d}_{i=1}(e_s)_i(r_p)_i(e_o)_i$，其中$(e_s)_i、(r_p)_i、(e_o)_i$表示向量$e_s、r_p、e_o$的第$i$元素。然后，目标是学习每个节点和边标签的向量，以最大化正边的合理性并最小化负边的合理性。这种方法相当于图张量G的CP分解，但其中实体有一个使用了两次的向量：$x_1⊗ y_1⊗ x_1+…+x_𝑑 ⊗ y_𝑑 ⊗ x_𝑑 ≈ G$，这种方法的一个缺点是，根据评分函数，$s-p\rightarrow o$的合理性将始终等于$o-p\rightarrow s$的合理性；换句话说，DistMult不考虑边缘方向。
**RESCAL**没有使用向量作为关系嵌入，而是使用一个矩阵，该矩阵允许在所有维度上组合$e_s$和$e_o$的值，因此可以捕获（例如）边缘方向。然而，RESCAL在空间和时间方面的成本高于DistMult。**HolE**使用向量进行关系和实体嵌入，但建议使用**循环相关算子**-它沿着两个向量的外积的对角线求和，从而将它们组合起来。此运算符不可交换，因此可以考虑边方向。另一方面，**ComplEx**使用复数向量（即包含复数的向量）作为关系嵌入，这同样允许打破上述DistMult评分函数的对称性，同时保持较低的参数数量。**SimplE**提出计算标准CP分解，计算X和Z中实体的两个初始向量，然后平均X、Y、Z中的项，以计算最终的合理性得分。**TuckER**采用了一种不同类型的分解——称为塔克分解（TuckER decomposition，它计算较小的“核心”张量T和三个矩阵A、B和C的序列，使得$G≈ T⊗A⊗ B⊗ C$–其中实体嵌入来自A和C，而关系嵌入来自B。在这些方法中，TuckER目前提供了标准基准的最新结果。

### 5.2.3神经模型
前面讨论的方法的一个局限性是，它们假设嵌入上的线性（保留加法和标量乘法）或双线性（例如矩阵乘法）运算来计算合理性分数。许多方法更倾向于使用神经网络来学习具有非线性评分函数的嵌入，以获得合理性。
- (Semantic Matching Energy)SME:学习$W, W\prime$,对$f_W(e_s,r_p)和g_{W\prime}(e_o,r_p)$，然后计算二者的点积$f_W(e_s,r_p)·g_{W\prime}(e_o,r_p)$
- (Neural Tensor Networks)NTN，它更倾向于保持内部权重的张量W，这样似然性得分由$e_s\otimes W\otimes e_o$，然后与$r_p$相结合，以产生一个合理性得分。
- 多层感知器（MLP）是一个更简单的模型，其中$e_s、r_p和e_o$串联并馈送到一个隐藏层，以计算合理性得分。

在模型中使用卷积核:
- ConvE:通过将每个向量 "包装 "在几行上并将两个矩阵连接起来，从es和rp生成一个矩阵。串联矩阵用作一组（2D）卷积层的输入，该卷积层返回特征映射张量。使用参数化线性变换标注将特征映射张量矢量化并投影到𝑑维 。然后根据该向量与eo的点积计算似然性得分。ConvE的一个缺点是，通过将向量包装到矩阵中，它在嵌入上施加了一个人工的二维结构。
- HypER将一个全连接的层（称为“超网络”）应用于rp，并用于生成特定于关系的卷积滤波器矩阵。这些过滤器直接应用于es，以提供矢量化的特征映射。然后应用与ConvE中相同的过程：生成的向量投影到𝑑 尺寸，以及与eo一起应用的点积，以生成合理性得分。结果表明，该模型在标准基准上的表现优于Conv。

### 5.2.4 语言模型
嵌入技术最初是作为一种在机器学习框架内表示自然语言的方法而被探索的，**word2vec**和**GloVe**是两种开创性的方法。这两种方法都基于大型文本语料库计算单词的嵌入，以便在类似上下文中使用的单词（例如，“青蛙”、“蟾蜍”）具有相似的向量。Word2vec使用经过训练的神经网络，从周围的单词（连续的单词包）预测当前单词，或预测给定当前单词的周围单词（连续的跳过图）。CloVe在词对的共现概率矩阵上应用回归模型。这两种方法生成的嵌入被广泛应用于自然语言处理任务中。

因此，图形嵌入的另一种方法是利用经过验证的语言嵌入方法。然而，图由三个术语的无序序列（即一组边）组成，类似的，自然语言中的文本由任意长度的术语序列（即单词的句子）组成。沿着这些路线,**RDF2Vec**在图上执行（偏置）随机游动，并将路径（遍历的节点和边标签序列）记录为“句子”，然后将其作为输入word2vec模型。其中论文以500条路径进行实验，每个实体的长度为8。RDF2Vec还提出了第二种模式，即从规范标记的子树中为节点生成序列，这些子树是其根节点，该文在这种模式下对深度为1和2的子树进行了实验。对应的，**KGloVe**基于GloVe模型。就像原始GloVe模型认为文本窗口中频繁出现的单词更相关一样，KGloVe使用个性化页面PageRank来确定与给定节点最相关的节点，然后将其结果输入GloVe模型。
直观地说，个性化的PageRank从给定的节点开始，然后确定在给定的步骤数之后，随机行走在特定节点的概率。较大量的步骤收敛到强调全局节点中心性的标准PageRank，而较小数量的步骤则强调与起始节点的接近性/关联性。

### 5.2.5包含感知模型
到目前为止，嵌入只考虑数据图。但如果提供了一个本体或一组规则怎么办？这种演绎知识可以用来改进嵌入。一种方法是使用约束规则来优化嵌入所做的预测；例如，Wang使用函数和逆函数定义作为约束（在UNA下），例如，如果我们定义一个事件最多可以有一个地点值，这将用于降低将多个地点分配给一个事件的边的合理性。
最近的方法更倾向于提出联合嵌入，在计算嵌入时同时考虑数据图和规则。**KALE**使用翻译模型（特别是TransE）计算实体和关系嵌入，该模型适用于使用t范数模糊逻辑进一步考虑规则。例如有规则$?x-bus\rightarrow ?y \Rightarrow ?x-connects to \rightarrow ?y$，我们可以使用嵌入为新边分配合理性分数，例如$𝑒1： Piedras Rojas-bus\rightarrow Moon Vally$。可以应用前面的规则来从$e_1$生成新边$𝑒2： Piedras Rojas-connects to\rightarrow MoonVally$。但我们应该赋予第二条边什么样的合理性呢？令𝑝1和𝑝2是𝑒1和𝑒2（使用标准嵌入初始化）当前的合理性得分，然后**t-范数模糊逻**辑建议似然性分数更新为$𝑝_1*𝑝_2−𝑝_1 + 1$. 然后，对嵌入进行训练，将更大的似然性分数联合分配给正面示例，而不是边缘和基本规则的负面示例。一个正面基本原则的例子是$Arica -bus\rightarrow San Pedro⇒ Arica-connects to \rightarrow San Pedro$。反面基本规则随机替换关系；例如，$Arica -bus\rightarrow San Pedro\nRightarrow Arica-flight \rightarrow San Pedro$。郭等人后来提出了**RUGE**，该模型使用了基本规则上的联合模型（可能是具有置信度得分的软规则）和合理性得分，以对齐两种形式的未看到边缘的得分。
## 5.3 图神经网络
图神经网络（GNN）基于数据图的拓扑结构构建神经网络；i、 例如，节点根据数据图与其邻居相连。通常，然后学习一个模型来映射节点的输入特征，以有监督的方式输出特征；例如节点的输出特征可以手动标记，也可以从知识图中获取。与知识图嵌入不同，GNN支持针对特定任务的端到端监督学习：
给定一组标记的示例，GNN可用于对图的元素或图本身进行分类。GNNs已被用于对编码化合物的图形、图像中的对象、文档等进行分类。；以及预测流量、构建推荐系统、验证软件等。给出了标记的例子，GNN甚至可以取代图形算法；例如，GNN已被用于以有监督的方式在知识图中查找中心节点[396397462]。
现在，我们讨论两种类型的GNN（递归GNN和卷积GNN）的基本思想，其中我们参考附录B.6.3了解有关GNN的更多正式定义。

### 5.3.1 递归图神经网络。
递归图神经网络（RecGNNs）是图神经网络的开创性方法。该方法在概念上类似于图25所示的脉动抽象，其中消息在邻居之间传递，以递归计算某些结果。然而，我们不是定义用于决定要传递的消息的函数，而是标记节点训练集的输出，让框架学习生成预期输出的函数，然后将它们应用于标记其他示例。
在一篇开创性的论文中，Scarselli等人提出了他们通常称之为图神经网络（GNN）的方法，该方法将节点和边与**特征向量**相关联的有向图作为输入，这些特征向量可以捕获节点和边标签、权重等。这些特征向量在整个过程中保持不变。图中的每个节点还与一个状态向量相关联，该状态向量根据节点邻居的信息进行递归更新，即相邻节点的特征和状态向量以及向/从其延伸的边的特征向量使用参数化函数，称为**过渡函数**。第二个参数函数称为**输出函数**，用于根据节点自身的特征和状态向量计算节点的最终输出。这些函数递归应用到固定点。这两个参数函数都可以使用神经网络来实现，其中，在图中给定一部分**受监督节点**集，即标记有其所需输出的节点，可以学习最接近受监督输出的过渡和输出函数的参数。因此，可以将结果视为递归神经网络结构。为了确保收敛到一个固定点，应用了某些限制，即过渡函数是一个**contractor**，这意味着在每次应用函数时，数值空间中的点会更紧密地聚集在一起（直观地说，在这种情况下，数值空间在每次应用时“收缩”，确保唯一的固定点）。
举例来说，为了说明我们希望找到建立新的旅游信息办公室的优先位置。一个好的策略是将它们安装在许多游客访问热门目的地的枢纽。沿着这些线，在图29中，我们展示了Scarselli等人为图24的子图提出的GNN架构，其中我们突出了`Punta Arenas`场的邻里。在该图中，节点使用特征向量$n_x$进行注释,和步骤中的隐藏状态$t(h_x^{(t)})$, 而边则使用特征向量进行注释$a_{xy}$. 节点的特征向量可能是，一个one-hot节点类型（城市、景点等），直接编码统计数据，如每年到访的游客数量等。边缘的特征向量可以，例如，一个one-hot编码边缘标签（交通类型），直接编码统计数据，如距离或每年售出的门票数量等。隐藏状态可以随机初始化。图29的右侧提供了GNN转换和输出函数，其中N(𝑥) 表示的相邻节点𝑥, 𝑓w（·）表示具有参数w的过渡函数，并且𝑔w′（·）表示参数为w′的输出函数。还为`Punta Arenas`提供了一个示例(𝑥 = 1). 这些函数将递归应用，直到达到不动点。为了训练网络，我们可以列举已经（或应该）有旅游办公室的地方和已经（或应该）没有旅游办公室的地方的例子。这些标签可以取自知识图，也可以手动添加。然后，GNN可以学习参数w和w′，这些参数为标记的示例提供预期输出，随后可以用于标记其他节点。
这种GNN模型是灵活的，并且可以以各种方式进行调整：我们可以不同地定义相邻节点，例如包括用于输出边的节点，或者一个或两个跳跃距离的节点；我们可以允许节点对通过具有不同向量的多条边连接；我们可以考虑每个节点具有不同参数的转移和输出函数；我们可以为边添加状态和输出；我们可以将总和更改为另一个聚合函数等
### 5.3.2 卷积图神经网络
卷积神经网络（CNN）已经获得了很多关注，尤其是在涉及图像的机器学习任务中。图像设置的核心思想是使用卷积算子对图像的局部区域应用小核（亦称过滤器），以从该局部区域提取特征。当应用于所有局部区域时，卷积输出图像的特征映射。通常应用多个核，形成多个卷积层。只要有足够的标记示例，就可以学习这些内核。
人们可能会注意到前面讨论的GNN与应用于图像的CNN之间的类比：
在这两种情况下，操作符都应用于输入数据的局部区域。在GNNs的情况下，转移函数应用于图中的节点及其邻居。在CNN的情况下，卷积应用于图像中的像素及其邻域。根据这一直觉，提出了许多**卷积图神经网络**(ConvGNNs)，其中通过卷积实现过渡函数。ConvGNNs的一个关键考虑因素是如何定义图的区域。与图像的像素不同，图中的节点可能具有不同数量的邻居。这带来了一个挑战：CNN的一个好处是相同的内核可以应用于图像的所有区域，但ConvGNNs的情况下，这需要更仔细的考虑，因为不同节点的邻域可能不同。解决这些挑战的方法包括使用图的谱或空间表示，从图中归纳出更规则的结构。另一种方法是使用注意机制来学习对当前节点最重要的特征的节点。
除了架构方面的考虑之外，RecGNNs和ConvGNNs之间还有两个主要区别。首先，RecGNNs递归地将邻居的信息聚合到一个固定点，而ConvGNNs通常应用固定数量的卷积层。其次，RecGNNs通常在统一的步骤中使用相同的函数/参数，而ConvGNNs的不同卷积层可以在每个不同的步骤中应用不同的内核/权重。
## 5.4 符号学习
迄今为止讨论的有监督技术，即知识图嵌入和图神经网络，在图上学习数值模型。然而，此类模型往往难以解释或理解。例如，以图30的图表为例，知识图嵌入可能会预测边`SCL`-`flight`$\rightarrow$`ARI`高度可信，但它们不会提供可解释的模型来帮助理解为什么会出现这种情况：结果的原因可能在于学习的参数矩阵，以拟合训练数据的可信分数。此类方法还存在词汇表外问题，无法提供涉及以前未看到的节点或边的边的结果；例如，如果我们添加了一个edge `SCL`-`flight`$\rightarrow$`CDG`，其中`CDG`是图中的新成员，则知识图嵌入将不会有`CDG`的实体嵌入，需要重新训练以估计edge `CDG`-`flight`$\rightarrow$`SCL`的合理性。
另一种（有时是互补的）方法是采用符号学习，以便用符号（逻辑）语言学习“解释”给定正负边集的假设。这些边通常以自动方式从知识图生成（类似于知识图嵌入的情况）。这些假设可以作为可解释的模型，用于进一步的演绎推理。例如，给出图30的图表，我们可以学习规则？x-`flight`$\rightarrow$？y⇒ ?y-`flight`$\rightarrow$？x从观察飞行路线往往是返回路线。或者，我们可以学习DL公理，即机场可以是国内机场，也可以是国际机场，或者两者兼有：机场⊑ 国内机场⊔ 国际机场。这样的规则和公理可以用于演绎推理，并为包含/预测的新知识提供可解释的模型；例如，根据上述回程航班规则，可以解释为什么会预测新的edge `SCL`-`flight`$\rightarrow$`ARI`。这进一步为领域专家提供了验证这些过程得出的模型（例如规则和公理）的机会。
最后，对规则/公理进行量化（所有航班都有返程航班，所有机场都是国内或国际机场等），以便将其应用于看不见的示例（例如，使用上述规则，我们可以从具有看不见节点CDG的新edge `SCL`-`flight`$\rightarrow$`CDG`推导`CDG`-`flight`$\rightarrow$`SCL`。
在本节中，我们将讨论两种形式的符号学习：规则挖掘，它从知识图中学习规则；公理挖掘，它学习其他形式的逻辑公理。
### 5.4.1 规则挖掘
一般来说，规则挖掘是指从大量背景知识集合中以规则的形式发现有意义的模式。在知识图的上下文中，我们假设一组给定的正边和负边。通常，正边是观察到的边（即知识图给定或包含的边），而负边是根据给定的完整性假设定义的（稍后讨论）。规则挖掘的目标是识别新的规则，这些规则需要较高的正边与其他正边的比率，但需要较低的负边与正边的比率。考虑的规则类型可能不同于更简单的案例。
以国际机场为例，规则并非在所有情况下都适用，而是与衡量规则是否符合正反两方面的标准相关联。更详细地说，我们称规则包含的边和正边集（不包括包含的边本身）为该规则的**正包含**。正的蕴涵数称为规则的**支持度**，而正的蕴涵数比率称为规则的**置信度**。因此，支持度和置信度分别表示规则中“确认”的蕴涵数和比率为真，目标是确定具有高支持度和高置信度的规则。事实上，在**归纳逻辑编程**（ILP）的背景下，长期以来一直在探索关系设置中的规则挖掘技术。然而，由于数据的规模和不完整数据（OWA）的频繁假设，知识图提出了新的挑战，其中提出了专门的技术来解决这些问题。
当处理一个不完整的知识图时，如何定义负边还不是很清楚。一种常见的启发式方法（也用于知识图嵌入）是采用部分完整性假设（PCA），该假设将正边集视为数据图中包含的边，将负示例集的例子，所有形如𝑥 𝑝 𝑦′的的不在图中的边的集合，但是存在节点𝑦 使得𝑥 𝑝 𝑦在图中。如图30所示，PCA下的负边缘示例为SCL flight ARI（假设存在SCL flight LIM）；相反，SCL domestic flight ARI既不为正也不为负。PCA置信度是支持与正集合或负集合中所有蕴涵的比率。例如，对规则？x domestic flight ？y⇒ ?y domestic flight ？x支持度为2（因为它在图中包含IQQ domestic flight ARI和ARI domestic flight IQQ），而置信度为2/2=1（注意，SCL domestic flight ARI虽然包含，但既不是正的也不是负的，因此被测量忽略）。对规则？x flight ？y⇒ ?y flight ？x的支持度4，而置信度为4/5=0.8（注意SCL flight ARI为负值）。
然后，目标是找到满足给定支持度和置信度阈值的规则。一个有影响力的图形规则挖掘系统是**AMIE**，它采用PCA置信度度量，以自上而下的方式建立规则，从规则头开始，如⇒ ?x country ？y对于此表单的每个规则头（每个边标签一个），将考虑三种类型的优化，每种类型都会向规则体添加一条新边。此新边从图中获取边标签，并可能使用以前未出现在规则中的新变量、已出现在规则中的现有变量或图中的节点。然后，可能会出现下面三个重定义：
1. 添加带有一个现有变量和一个新变量的边；例如，优化上述规则头可能会给出：？z航班？x个⇒ ?x国家？y
2. 从图中添加一条带有现有变量和节点的边；例如，完善上述规则可能会给出：Domestic Airport <- type-？z-flight->？x个⇒ ?x-country->？y
3. 添加带有两个现有变量的边；例如，优化上述规则可能会得到：
国内机场类型？z航班？x？y国家/地区⇒ ?x国家？y

这些优化可以任意组合，这会产生一个潜在的指数搜索空间，在该空间中，满足给定阈值的支持度和置信度规则将保持不变。为了提高效率，可以修剪搜索空间；例如，这三种优化总是会降低支持率，因此，如果规则不满足支持阈值，则无需探索其优化。
对生成的规则类型施加了进一步的限制。首先，只考虑达到某个固定大小的规则。其次，规则必须关闭，这意味着每个变量至少出现在规则的两条边上，这确保了规则是安全的，这意味着头部的每个变量都出现在身体上；例如，之前由第一次和第二次优化生成的规则既不是闭合的（变量y出现一次），也不是安全的（变量y只出现在头部）。为确保关闭规则，将应用第三次优化，直到关闭规则。有关基于修剪和索引的可能优化的进一步讨论，请参阅关于**AMIE**的论文。
后来的工作是基于这些技术从知识图中挖掘规则。Gad Elrab等人提出了一种学习非单调规则的方法，即身体中具有否定边的规则，以便捕捉基本规则的例外情况；例如，该方法可以学习国际机场类型规则？z航班？x？y国家/地区⇒ ?x国家？y，表示航班在同一国家内，除非（起飞）机场是国际机场，例外情况显示为虚线，我们使用­来抵消边缘。**RuLES**——也能够学习非单调规则——建议通过扩展置信度来考虑未出现在图中的包含边的知识图嵌入的合理性得分，从而缓解PCA启发式的局限性。在可用的情况下，可以使用关于知识图完整性的明确声明（如以形状表示）代替PCA来识别负边。沿着这些思路，**CARL**利用关于关系基数的额外知识来完善负面示例集和候选规则的置信度度量。或者，在可用的情况下，可以使用本体来推导OWA下的逻辑确定负边，例如，通过不相交公理。达马托（d'Amato）等人提出的系统利用本体论上包含的负边来确定通过进化算法生成的规则的可信度。
虽然之前的工作涉及应用固定置信度评分函数的候选规则的离散扩展，但另一个研究方向是一种称为**可微规则挖掘的技术**，该技术允许对规则进行端到端学习。其核心思想是，规则体中的连接可以表示为矩阵乘法。更具体地说，我们可以表示边标签的关系𝑝 通过邻接矩阵A𝑝 （尺寸|𝑉 | × |𝑉 |) 这样第i行𝑗列如果为1，则表示有边p在第i个到第j个节点之间。否则，该值为0。现在我们可以将规则体中的连接表示为矩阵乘法；例如，给定？x domestic flight ？y country ？z⇒ ?x country ？z，我们可以用矩阵乘法$A_{df}.A_c$来表示物体，它给出了一个表示包含的country边缘的邻接矩阵，我们应该期望$A_{df}.A_c$中的1由头部邻接矩阵Ac覆盖。因为我们得到了所有边标签的邻接矩阵，所以我们只能学习单个规则的置信度得分，并学习具有阈值置信度的规则（长度不同）。沿着这些思路，NeuralLP使用注意机制为形状的路径规则选择可变长度的边标签序列？x p1？y1 p2。p𝑛 ?y𝑛 p𝑛+1.z⇒ ?x p？z，同样也会学到置信。DRUM还学习类似路径的规则，其中，观察到一些边缘标签更可能/不太可能遵循规则中的其他标签–例如，在图24的图表中，由于连接为空，flight后面不会跟着大写字母–系统使用双向递归神经网络（一种学习序列数据的流行技术）来学习规则的关系序列，以及他们的信任。然而，这些可微规则挖掘技术目前仅限于学习类路径规则。
### 5.4.2 Axiom挖掘
除了规则之外，还可以从知识图中挖掘出更一般形式的公理，这些公理以逻辑语言（如DLs）表示。我们可以将这些方法分为两类：挖掘特定公理的方法和更一般的公理的方法。
在挖掘特定类型公理的系统中，不相交公理是一个流行的目标；例如，不相交公理DomesticAirport⊓ 国际机场≡ ⊥ 声明两个等级的相交等同于空等级，或者更简单地说，没有节点可以同时为国内机场和国际机场类型。Völker等人提出的系统基于（负）关联规则挖掘提取不相交公理，发现成对的类，其中每个类在知识图中有许多实例，但这两个类的实例相对较少（或没有）。Töpper等人更倾向于为具有低于固定阈值的余弦相似性的类对提取不相交性。为了计算这种余弦相似性，使用TF–IDF类比计算类向量，其中每个类的“文档”是从其所有实例构建的，而该文档的“术语”是类实例上使用的属性（保留多重性）。虽然前两种方法发现了命名类之间的不相交约束（例如，城市与机场不相交），Rizzo等人提出了一种方法，可以捕获类描述之间的不相交约束（例如，附近没有机场的城市与国家首都城市不相交）。该方法首先对知识库的相似节点进行聚类。接下来，提取术语聚类树，其中每个叶节点表示先前提取的聚类，每个内部（非叶）节点是一个类定义（例如，城市），其中左子节点是具有该类中所有节点的集群或子类描述（例如，没有机场的城市），右子节点是没有该类中节点的集群或不相交的类描述（例如，有事件的非城市）。最后，针对树中不必具有子类关系的类描述对，提出了候选不相交公理。
其他系统提出了学习更一般公理的方法。一个突出的此类系统是DLLearner，它基于类学习算法（又名概念学习），通过给定一组正节点和负节点，目标是找到划分正节点和负节点的逻辑类描述。例如，给定{Iquique，Arica}作为正集，{Santiago}作为负集，我们可以学习（DL）类描述∃nearby.Airport⊓¬(∃capital.⊤), 表示机场附近的非首都实体，其中所有正节点都是实例，没有负节点是实例。此类类描述的学习方式与前述系统（如AMIE学习规则）类似，使用细化操作符从更一般的类移动到更具体的类（反之亦然）、置信评分函数和搜索策略。该系统还支持通过评分学习更一般的公理函数，该函数使用count查询确定在图中确实找到的预期边（如果公理为真，则会产生的边）的比率；例如，为公理打分∃航班−.国内机场⊑ 国际机场在图30中，我们可以使用图形查询来计算有多少节点有来自国内机场的入境航班（有3个），有多少节点有来自国内机场的入境航班，有多少节点有来自国际机场的入境航班（有1个），其中两个计算之间的差异越大，公理的证据就越弱。
# 6 创造和丰富
在本节中，我们将讨论创建知识图的主要技术，并随后从各种遗留数据源（从纯文本到结构化格式（以及介于两者之间的任何数据）中丰富知识图。创建知识图时应遵循的适当方法取决于所涉及的参与者、领域、设想的应用程序、可用的数据源等。然而，一般而言，知识图的灵活性有助于从一个初始核心开始，该核心可以根据需要从其他资源中逐步丰富（通常遵循敏捷方法或“按量付费”方法）。在我们的运行示例中，我们假设旅游局决定从头开始构建一个知识图，旨在最初描述智利的主要旅游景点——地点、事件等，以帮助来访游客确定他们最感兴趣的景点。委员会决定推迟添加更多数据，如运输路线、犯罪报告等，稍后再添加。
## 6.1 人员协作
创建和丰富知识图的一种方法是征求人工编辑的直接贡献。这些编辑可以在内部找到（例如旅游局的员工），使用众包平台，通过反馈机制（例如，游客对景点添加评论），通过协作编辑平台（例如，开放给公众编辑的景点维基），等等。尽管人的参与会带来高昂的成本，一些著名的知识图表主要基于人类编辑的直接贡献。然而，根据征集贡献的方式，该方法有许多关键缺点，主要是由于人为错误、分歧、偏见、故意破坏等。成功的协作创作进一步提出了许可、工具和文化方面的挑战。有时，人们更倾向于验证和管理通过其他方式提取的知识图的添加内容（例如，通过有目的的视频游戏），定义来自其他来源的高质量映射，定义适当的高级模式等等。
## 6.2文本来源
文本语料库——比如来自报纸、书籍、科学文章、社交媒体、电子邮件、网络爬虫等——是丰富信息的丰富来源。然而，为了创建或丰富知识图，以高精度和高召回率提取此类信息是一项非常艰巨的挑战。为了解决这个问题，可以应用自然语言处理（NLP）和信息提取（IE）的技术。尽管文本提取框架的过程差异很大，但在图31中，我们展示了一个示例句子中文本提取的四个核心任务。我们将依次讨论这些任务。
### 6.2.1预处理
预处理任务可能涉及对输入文本应用各种技术，其中图31说明了标记化，它将文本解析为原子术语和符号。
应用于文本语料库的其他预处理任务可能包括：
- 词性标记，以识别代表动词、名词、形容词等的术语；
- 依赖分析，提取句子的语法树结构，其中叶节点表示单独的单词，这些单词共同构成短语（例如，名词短语、动词短语），最终形成从句和句子；
- 词义消歧（WSD）用于识别单词使用的意义（又名意义），将单词与词义词典（例如WordNet或BabelNet）联系起来，其中，例如，术语flights可能与WordNet的意义“航空旅行的实例”联系起来而不是“一层楼与另一层楼之间的楼梯”。

应用适当类型的预处理通常取决于管道中后续任务的要求。
### 6.2.2命名实体识别（NER）。
NER任务识别文本中提及的命名实体，通常针对提及的人员、组织、地点和潜在的其他类型。存在多种NER技术，许多现代方法基于学习框架，利用词汇特征（例如，POS标记、依赖关系解析树等）和地名录（例如，常见的名字、姓氏、国家、知名企业等）。监督方法需要手动标记训练语料库中的所有实体提及，而基于自举的方法更需要一小组实体提及的种子示例，从中可以学习模式并应用于未标记的文本。远程监控使用知识图中的已知实体作为种子示例，通过这些示例可以检测到类似的实体。除了基于学习的框架之外，手动起草的规则由于其更可控和可预测的行为，有时仍被使用。NER标识的命名实体可用于为知识图生成新的候选节点（称为新兴实体，如图31中的虚线所示），或可根据以下描述的实体链接任务链接到现有节点。
### 6.2.3实体链接（EL）。
EL任务将文本中提到的实体与目标知识图的现有节点相关联，目标知识图可能是正在创建的知识图或外部知识图的核心。在图31中，我们假设节点Santiago和Easter Island已经存在于知识图中（可能从其他来源提取）。EL然后可以将给定的提及链接到这些节点。EL任务提出了两个主要挑战。首先，可能有多种方式提及同一实体，如拉帕努伊岛和复活节岛；如果我们创建了一个节点Rapa Nui来表示该提及，我们会将两个提及下的可用信息分割到不同的节点，因此目标知识图必须捕获各种别名和多语言标签，通过这些别名和标签可以引用实体。其次，不同语境中的同一提及可以指代不同的实体；例如，圣地亚哥可以指智利、古巴、西班牙等国的城市。因此，EL任务考虑了消歧阶段，其中提及与知识图中的候选节点相关联，对候选节点进行排序，并选择最可能提及的节点。在此阶段可以使用上下文；例如，如果复活节岛是与圣地亚哥并列的相应提及对象，我们可能会提高提及智利首都的可能性，因为两位候选人都位于智利。其他消除歧义的启发式方法考虑先验概率，例如，圣地亚哥最常指智利首都（例如，智利最大的城市）；知识图上的中心性度量可用于此类目的。
### 6.2.4关系提取（RE）。
RE任务提取文本中实体之间的关系。最简单的情况是在封闭环境中提取二元关系，其中考虑了一组固定的关系类型。虽然传统方法通常使用手工制作的模式，但现代方法更倾向于使用基于学习的框架，包括手动标记示例的监督方法。其他基于学习的方法再次使用自举（bootstrapping）和远程监控来放弃手动标记的需要；前者需要手动标记种子示例的子集，而后者在一个大型文本语料库中查找句子，其中提到具有已知关系/边缘的成对实体，这些实体用于学习该关系的模式。二进制RE也可以在开放环境中使用无监督的方法应用，通常称为开放信息提取(OIE)，其中目标关系集不是预定义的，而是基于依赖关系解析树从文本中提取的。
已经提出了多种RE方法来提取𝑛-元关系，用于捕捉实体之间关系的进一步上下文。在图31中，我们看到𝑛-ary关系捕获了额外的时间上下文，表示拉帕努伊何时被命名为世界遗产；在这种情况下，将创建一个匿名节点来表示有向标记图中较高的arity关系。各种方法𝑛-ary RE基于框架语义，对于给定的动词（例如。“命名”），捕获所涉及的实体及其相互关联的方式。然后，FrameNet等资源定义单词的框架，例如，确定“命名”的语义框架包括说话人（命名对象的人）、实体（命名对象）和名称。可选的框架元素是解释、目的、地点、时间等，可以为关系添加上下文。其他RE方法则基于话语表征理论（DRT），该理论考虑了基于存在事件的文本逻辑表征。例如，根据这一理论，将复活节岛命名为世界遗产被认为是一个（存在的）事件，其中复活节岛是患者（受影响的实体），由此产生了合乎逻辑的（新大卫主义）公式：如前3.3节所述，这种公式类似于物化的概念。
最后，虽然在封闭环境中提取的关系通常直接映射到知识图，但在开放环境中提取的关系可能需要与知识图对齐；例如，如果OIE过程提取出一个二元关系Santiago has flights to Easter Island，则可能是知识图没有标记为has flights to的其他边的情况，如果知识图中使用了flight，则alignment可能会将这种关系映射到边缘Santiago flight Easter Island。为此，已经应用了多种方法，包括映射和对齐规则𝑛-三元关系；基于分布和依赖关系的相似度、关联规则挖掘、马尔可夫聚类和语言技术，用于对齐OIE关系；除其他外。
### 6.2.5联合任务。
在介绍了从文本构建知识图的四个主要任务之后，需要注意的是，框架并不总是遵循这一特定的任务序列。
例如，一种常见的趋势是将相互依存的任务结合起来，共同执行WSD和EL，或NER和EL，或NER和RE等，以便相互改进多个任务的性能。有关从文本中提取知识的更多详细信息，请参阅梅纳德等人的书和马丁内斯·罗德里格斯等人最近的调查。
## 6.3标记源
Web建立在互连标记文档的基础上，其中标记（又名标记）用于分隔文档的元素（通常用于格式化目的）。Web上的大多数文档都使用超文本标记语言（HTML）。图32显示了一个关于智利世界遗产的HTML网页示例。其他标记格式包括Wikipedia使用的Wikitext、用于排版的TeX、内容管理系统使用的Markdown等。从标记文档中提取信息的一种方法——为了创建和/或丰富知识图——是去除标记（例如HTML标记），只留下纯文本，可以应用上一节中的技术。然而，标记可用于提取目的，其中上述文本提取任务的变体已被改编为利用此类标记[324、327、338]。我们可以将标记文档的提取技术分为三大类：与特定格式中使用的标记无关的通用方法，通常基于将文档元素映射到输出的包装器；
以文档中特定形式的标记为目标的集中方法，最典型的是web表（但有时也包括列表、链接等）；以及基于表单的方法，根据Deep Web的概念提取网页底层的数据。这些方法通常可以从给定网站的网页共享的规则中获益，无论是由于关于如何跨网页发布信息的非正式约定，还是由于重复使用模板来跨网页自动生成内容；例如，直观地说，虽然图32的网页是关于智利的，但我们可能会在同一个网站上找到其他国家采用相同结构的网页。
### 6.3.1基于包装器的提取。
许多通用方法都基于包装器，包装器可以直接从标记文档中定位和提取有用的信息。虽然传统的方法是手动定义这种包装器，这是一项定义了各种声明性语言和工具的任务，但这种方法对于网站布局的更改很脆弱。因此，其他方法允许（半）自动诱导包装物。现代的此类方法–用于丰富LODIE等系统中的知识图–是为了应用远程监控，其中EL用于识别网页中的实体并将其链接到知识图中的节点，从而可以提取、排序标记中连接已知边的成对节点的路径，并应用于其他示例。例如，以图32为例，远程监控可以使用EL将拉帕努伊岛和世界遗产地与知识图中的节点复活节岛和世界遗产地联系起来，并在知识图中给出命名为世界遗产地的边缘复活节岛（根据图31提取），确定候选路径(𝑥, td【1】− · tr公司− · 桌子− · h1，𝑦) 作为形状的反射边缘𝑥 已命名𝑦 , 哪里𝑡 [𝑛] 指示𝑛tag的第个子级𝑡, 𝑡− 其逆方向，以及𝑡1 · 𝑡2串联。最后，可以使用具有高置信度的路径（例如，知识图中许多已知边“见证”的路径）来提取新边，例如Qhapaqña命名世界遗产地，无论是在该页面上还是在具有类似结构的网站的相关页面上（例如，对于其他国家）
### 6.3.2 Web表提取。
其他方法针对特定类型的标记，最常见的是web表，即HTML网页中嵌入的表。然而，web表的设计是为了增强人的可读性，而人的可读性往往与机器的可读性相冲突。许多web表用于布局和页面结构（如导航栏），而那些确实包含数据的web表可能采用不同的格式，如关系表、列表、属性值表、矩阵等。因此，第一步是对表进行分类，以找到适合给定提取机制的表。接下来，web表可以包含列跨、行跨、内部表，或者可以垂直拆分以提高人类审美。因此，需要一个表规范化阶段来识别标头、合并拆分表、取消嵌套表、转置表。随后，方法可能需要确定主角——表中描述的主要实体–在网页的其他地方可以找到；例如，尽管世界遗产地是图31表格的主角，但表格中并未提及。最后，可以应用提取过程，潜在地将单元格与实体、列与类型以及列对与关系相关联。为了丰富知识图，最近的方法再次应用远程监控，首先将表单元格链接到知识图节点，这些节点用于生成类型和关系提取的候选对象。统计分布也可以帮助链接数字列。还为特定网站上的表设计了专门的提取框架，其中突出的知识图，如DBpedia和YAGO侧重于从Wikipedia的信息框表中提取。
### 6.3.3深网爬虫
Deep Web提供了一个只有通过搜索Web表单才能访问的丰富信息源，因此需要使用Deep Web爬虫技术来访问。有人提议使用系统从深层网络资源中提取知识图。这些方法通常试图生成合理的表单输入（可能基于用户查询或参考知识生成），然后使用上述技术从生成的响应（标记文档）中提取数据。
## 6.4结构化来源
组织内部和Web上可用的许多遗留数据都是以结构化格式表示的，主要是以关系数据库、CSV文件等形式的表，但也有树结构格式，如JSON、XML等。与文本和标记文档不同，结构化通常可以将源映射到知识图，从而根据映射（精确）转换结构，而不是（不精确）提取结构。映射过程包括两个步骤：1）创建从源到图形的映射，以及2）使用映射以将源数据具体化为图形或虚拟化源（在遗留数据上创建图形视图）。
### 6.4.1表格映射
表格数据源非常普遍，例如，许多组织、网站等的结构化内容都存放在关系数据库中。在图33中，我们展示了一个关系数据库实例的示例，我们希望将其集成到正在构建的知识图中。然后有两种将内容从表映射到知识图的方法：直接映射和自定义映射。
直接映射会从表中自动生成图形。我们在图34中展示了标准直接映射的结果，该映射为表的每个（非标题、非空、非空）单元格创建一个边x y z，这样x表示单元格的行，y表示单元格的列名，z表示单元格的值。特别地，x通常对一行的主键的值进行编码（例如，claimer.id）；否则，如果未定义主键（例如，根据报表表），x可以是匿名节点或基于行号的节点。节点x和边标签y进一步对表的名称进行编码，以避免在具有不同含义的相同列名的表之间发生冲突。对于每一行x，我们可以根据其表的名称添加一个类型边缘。值z可以基于源域映射到相应图形模型中的数据类型值（例如，日期类型的SQL列中的值可以映射到RDF数据模型中的xsd:Date）。如果值为null（或空），通常会忽略相应的边。关于图34，我们强调了节点claimer-XY12SDA和XY12SDA之间的区别，其中前者表示由后者主键值标识的行（或实体）。如果两个表之间存在外键，例如Report。申请人引用申请人。id–例如，我们可以链接到claimer-XY12SDA，而不是XY12SDA，其中前一个节点还具有索赔人的名称和国家/地区。沿着这些路线的直接映射已经标准化，用于将关系数据库映射到RDF，其中Stoica等人最近提出了一种类似的属性图直接映射。已为CSV和其他表格数据定义了另一种直接映射，进一步允许指定列名、主键/外键和数据类型，这些数据格式中通常缺少这些数据类型，作为映射本身的一部分。
尽管直接映射可以自动应用于表格数据源，并保留原始源的信息，即允许从输出图重建表格源的确定性反向映射，但在许多情况下，需要定制映射，例如将边标签或节点与正在丰富的知识图对齐等。按照这些思路，声明性映射语言允许手动定义自定义映射从表格来源到图表。沿着这些路线的标准语言是RDB2RDF映射语言（R2RML），它允许从表的单个行映射到一个或多个自定义边，节点和边定义为常量、单个单元格值，或使用模板将行中的多个单元格值和静态子字符串连接到单个项中；例如，模板{id}-{country}可以从claimer表中生成诸如XY12SDA-美国这样的节点。如果无法从一行定义所需的输出边，R2RML允许（SQL）查询生成可以从中提取边的表，例如，可以通过定义与查询的映射来生成边，该查询将报告和索赔人表连接到索赔人=id，按国家分组，并应用计数，从而生成美国犯罪2等边。然后，可以在结果表上定义映射，以便源节点表示国家的值，边缘标签表示常量犯罪，目标节点表示计数值。
还存在一个类似的标准，用于将CSV和其他表格数据映射到RDF图，再次允许选择键、列名和数据类型作为映射的一部分。
定义映射后，一种选择是使用它们按照提取转换负载（ETL）方法实现图形数据的具体化，从而使用映射转换表格数据并显式序列化为图形数据。第二种选择是通过查询重写（QR）方法使用虚拟化，从而将图形上的查询（例如使用SPARQL、Cypher等）转换为表格数据上的查询（通常使用SQL）。通过比较这两个选项，ETL允许使用图数据，就像它们是知识图中的任何其他数据一样。然而，ETL要求将对底层表格数据的更新显式传播到知识图，而QR方法只维护要更新的数据的一个副本。
然后，基于本体的数据访问（OBDA）领域涉及支持第4节中讨论的本体蕴涵的QR方法。虽然大多数QR方法只支持可表示为单个（非递归）查询的非递归蕴涵，但一些QR方法通过重写递归查询来支持递归蕴涵。
### 6.4.2树木制图
许多流行的数据格式都基于树，包括XML和JSON。而我们可以想象，撇开诸如树中孩子的排序等问题不谈–只需创建表单的边，即可实现从树到图的简单直接映射𝑥 小孩𝑦对于每个节点𝑦 那是𝑥 在源树中，通常不使用这种方法，因为它表示源数据的文本结构。相反，可以使用自定义映射更自然地将树结构数据的内容表示为图形。沿着这些思路，GRDLL标准允许从XML映射到（RDF）图，而JSON-LD标准允许从JSON映射到（RDF）图。相比之下，XSPARQL等混合查询语言允许以集成的方式查询XML和RDF，从而支持在树结构的遗留数据源上实现图形的实体化和虚拟化。
### 6.4.3其他知识图的映射。
构建或丰富知识图的另一种途径是利用现有的知识图作为源。例如，在我们的场景中，智利旅游局的大量兴趣点可能存在于现有的知识图中，如DBpedia、LinkedGeoData、Wikidata、YAGO、BabelNet等。然而，根据正在构建的知识图，并非所有实体和/或关系都有兴趣。提取相关数据子图的一个标准选项是使用SPARQL构造查询，生成图形作为输出。为了更好地集成（部分）外部知识图，可能还需要知识图之间的实体和模式对齐；这可以使用图形链接工具，基于外部标识符的使用来完成，也可以手动完成。例如，Wikidata使用Freebase作为源；Gottschalk和Demidova提取了以事件为中心的知识Wikidata、DBpedia和YAGO的图表；而Neumaier和Polleres则利用地理名称、Wikidata和PeriodO（以及表格数据）构建了时空知识图。
## 6.5模式/本体创建
到目前为止，讨论的重点是从外部来源提取数据，以创建和丰富知识图。在本节中，我们将讨论基于外部数据源（包括人类知识）生成模式的一些主要方法。有关从知识图本身提取模式的讨论，请参阅第3.1.3节。通常，该领域的大部分工作都集中在使用本体工程方法和/或本体学习创建本体上。我们依次讨论这两种方法。
### 6.5.1本体工程
本体工程（ontologyengineering）是指开发和应用构建本体的方法，提出原则性流程，通过这些流程可以轻松构建和维护质量更好的本体。早期的方法论通常基于瀑布式流程，在开始用逻辑语言实现本体之前，需求和概念化是固定的，例如使用本体工程工具。然而，对于涉及大型或不断演化的本体的情况，已经提出了构建和维护本体的更多迭代和敏捷方法。DILIGENT是敏捷方法论的早期范例，提出了本体生命周期管理和知识进化的完整过程，并将局部变化（局部知识视图）与本体核心部分的全局更新分离，使用审查过程授权将变化从局部传播到全球。该方法类似于维护和发展大型临床参考术语SNOMED CT（也可用作本体），其中（国际）核心术语根据全球需求维护，而国家或地方对SNOMED CT的扩展则根据当地需求维护。然后，一组作者决定将哪些国家或地方扩展传播到核心术语。更现代的敏捷方法包括极限设计（XD）、模块化本体建模（MOM）、简化的本体开发敏捷方法（SAMOD）等。此类方法通常包括两个关键要素：本体需求和（最近）本体设计模式。本体需求基于本体作为其模式，指定产生的本体的预期任务，或者实际上是知识图本身。表达本体需求的一种常见方式是通过能力问题（CQ），这是一种自然语言问题，说明需要本体（或知识图）提供的典型知识。如果本体还应该包含用于推断新知识或检查数据一致性的限制和一般公理，那么这样的CQs可以补充额外的限制和推理需求。测试本体（或基于本体的知识图）的一种常见方法是将CQs形式化为对一些测试数据集的查询，并确保得到预期的结果。例如，我们可以考虑CQ“圣地亚哥发生的所有事件是什么？”，哪个可以表示为图形查询事件类型？活动地点圣地亚哥。以图1的数据图和图12的公理为例，我们可以检查预期结果EID15是否由本体和数据所包含，因为它不是，我们可以考虑扩展公理来断言位置类型是可传递的。
本体设计模式（ODP）是现代方法学的另一个共同特征，它指定了可推广的本体建模模式，这些模式可以作为类似模式建模的灵感，作为建模模板，或作为可直接重用的组件。
一些模式库已经在网上提供，从精心策划的模式库到开放的和社区主持的模式库。例如，在为在我们的场景中，我们可能会决定遵循Krinadhi和Hitzler提出的核心事件本体模，该模式指定了事件的时空范围、子事件和参与者，进一步建议能力问题、正式定义等，以支持该模式。
### 6.5.2本体学习
以前的方法概述了可以手动构建和维护本体的方法。相反，本体学习可以用于（半）自动地从文本中提取信息，这对于本体工程过程非常有用。早期的方法侧重于从可能代表相关领域类的文本中提取术语；例如，从一组关于旅游业的文本文档中，一个术语提取工具——使用统一性度量，确定𝑛-gram是一个单一短语，确定该短语与域的相关性的术语可以识别𝑛-GRAM，如“游客签证”、“世界遗产”、“非高峰费率”等，作为对旅游领域特别重要的术语，因此可能值得纳入此类本体中。公理也可以从文本中提取，其中子类公理通常以修改名词和形容词为目标，这些名词和形容词逐渐专门化概念（例如，从名词短语“Visitor Visa”中提取Visa的Visitor Visa subc，并在其他地方单独出现“Visa”），或者使用赫斯特模式（例如，根据模式“X，如Y”，从“许多折扣，如非高峰价格可用”中提取折扣的非高峰价格分包）。还可以从大型文本中获取文本定义，以提取超词关系，并从零开始归纳分类法。最近的作品旨在从文本中提取更多的表达性公理，包括不连续性公理；以及涉及类的并集和交集的公理，以及存在的、普遍的和限定的基数限制。然后，本体学习过程的结果可以作为更通用的本体工程方法的输入，使我们能够验证本体的术语覆盖范围，识别新的类和公理等。
# 7 质量评估
与创建知识图的来源（种类）无关，为初始知识图提取的数据通常不完整，并且会包含重复、矛盾甚至错误的陈述，尤其是从多个来源提取的数据。在最初从外部来源创建和丰富知识图之后，关键的一步是评估生成的知识图的质量。就质量而言，我们这里指的是健身目的。然后，质量评估有助于确定知识图可用于哪些目的。
在下文中，我们将讨论质量维度，这些维度反映了多方面数据质量的各个方面，这些数据质量从传统的数据库领域发展到知识图领域，其中一些是通用的，另一些则更为知识图所特有。虽然质量维度旨在捕获数据的定性方面，但我们也讨论了质量度量，这些度量提供了度量这些维度的定量方面的方法。我们讨论了巴蒂尼（Batini）和斯坎纳皮科（Scannapieco）所启发的维度和指标分组。
## 7.1 准确性
准确性是指由图中的节点和边编码的实体和关系正确表示现实生活现象的程度。准确性可以进一步细分为三个维度：**句法准确性**、**语义准确性**和**及时性**。
### 7.1.1 句法准确性
句法准确性是指数据相对于为域和/或数据模型定义的语法规则的准确程度。语法错误的一个常见示例发生在数据类型节点上，它可能与定义的范围不兼容或格式不正确。例如，假设属性start是使用范围xsd:dateTime定义的，并取一个值例如“2019年3月29日20:00”^^xsd:string将与定义的范围不兼容，而“2019年3月29日20:00”^^xsd:dateTime的格式不正确（应输入“2019-03-29T20:00:00”^^xsd:dateTime等值）。对应的语法准确性指标是给定属性的错误值数量与相同属性的总值数量之间的比率。这种形式的句法准确性通常可以使用验证工具进行评估。
### 7.1.2 语义准确性
语义准确性是数据值正确表示真实世界现象的程度，可能会受到不精确提取结果、不精确蕴涵、故意破坏等的影响。例如，鉴于智利国会位于瓦尔帕莱索，这可能会产生边``智利首都瓦尔帕莱索``（通过包含、提取、完成等），这实际上在语义上是不准确的：智利首都是圣地亚哥。评估语义不准确的程度很有挑战性。虽然一个选项是应用手动验证，但一个自动选项可能是对照多个来源检查所述关系。另一种选择是验证用于生成知识图的各个过程的质量，基于精度等度量，可能需要借助人类专家或黄金标准。
### 7.1.3时效性
时效性是指知识图当前与真实世界状态的最新程度；换言之，知识图现在可能在语义上是准确的，但如果没有及时更新的程序，它可能很快就会变得不准确（过时）。例如，考虑一个用户检查从一个城市到另一个城市的航班的游客知识图。假设航班时刻表每分钟更新一次当前航班状态，但知识图仅每小时更新一次。在这种情况下，我们发现知识图中的时效性存在质量问题。时效性可以根据知识图相对于基础源更新的频率来评估，这可以使用知识图中变化的时间注释以及捕获数据时间有效性的上下文表示来完成。
## 7.2覆盖范围
覆盖率是指避免遗漏与领域相关的元素，否则可能会产生不完整的查询结果或蕴涵、有偏差的模型等。
### 7.2.1完整性
完整性是指特定数据集中存在所有必需信息的程度。完整性包括以下几个方面：（i）模式完整性是指模式的类和属性在数据图中表示的程度，（ii）属性完整性是指特定属性缺失值的比率，（iii）总体完整性提供了数据集中表示的特定类型的所有真实实体的百分比，以及（iv）可链接性完整性是指数据集中实例的互连程度。直接测量完备性是非常重要的，因为它需要一个假设的理想知识图的知识，该知识图包含所讨论的知识图应该“理想”表示的所有元素。具体策略包括与提供理想知识图样本的黄金标准进行比较（可能基于完整性陈述），或测量从完整来源中提取方法的召回率，等等。
### 7.2.2代表性
代表性是一个相关维度，它不是侧重于缺失的领域相关元素的比率，而是侧重于评估知识图中包含/排除的内容中的高级偏差。因此，该维度假设知识图是不完整的，即它是理想知识图的一个样本，并询问该样本的偏差有多大。数据、模式或推理过程中可能会出现偏差。数据偏差的例子包括地理偏差，它低估了世界某些地区的实体/关系，语言偏差，低估了多语言资源（e、 g.标签和描述）对于某些语言，低估了特定性别或种族的人的社会偏见，等等。相反，模式偏差可能源于从有偏差的数据中提取的高级定义、不涵盖罕见情况的语义定义等。未识别的偏差可能会导致不利影响；例如，如果我们的旅游知识图表对圣地亚哥市附近的活动和景点有地理偏见——可能是由于用于创作的资源、该市馆长的雇佣等原因——那么这可能会导致圣地亚哥市内和周边的旅游业得到不成比例的推广（可能会加剧未来的偏见）。代表性度量包括将已知的统计分布与知识图的统计分布进行比较，例如，将地理位置实体与已知的人口密度进行比较，将语言分布与已知的说话人分布进行比较等。另一种选择是将知识图与一般统计规律进行比较，其中，Soulet等人使用（不）符合Benford定律来衡量知识图中的代表性。
## 7.3一致性
一致性是指知识图与模式级定义的形式语义和约束的一致性。
### 7.3.1一致性
意味着知识图在所考虑的特定逻辑蕴涵方面没有（逻辑/形式）矛盾。例如，在知识图的本体中，我们可以定义flight -range -> Airport -disj.c ->City,当与边Arica-flight-> Santiago-type->City结合时，会产生不一致性，这意味着圣地亚哥是城市和机场不相交的类别的一员。更一般地说，表3-5中的任何语义特征如果包含否定条件，则“not”条件可能会导致不一致。一致性的度量可以是在知识图中发现的不一致的数量，可以细分为每个语义特征识别的此类不一致的数量。
### 7.3.2有效性
意味着知识图不存在约束冲突，如形状表达式捕获的约束冲突（见第3.1.2节）。例如，我们可以指定一个形状城市，其目标节点最多有一个国家。然后，考虑到边Chile<-country-Santiago-country->Cuba，假设Santiago成为City的目标，我们就违反了约束。相反，即使我们在本体中定义了类似的基数限制，这也不一定会导致不一致，因为如果没有UNA，我们首先会推断智利和古巴指的是同一实体。有效性的一个简单度量是计算每个约束的违规数量。
## 7.4简洁
简洁是指仅包含以简洁易懂的方式表示的相关内容（避免“信息过载”）。
### 7.4.1简洁性
是指避免包含与域无关的模式和数据元素。Mendes等人区分了内涵简洁性（模式级），即数据不包含冗余模式元素（属性、类、形状等）的情况，以及扩展简洁性（数据级），即数据不包含冗余实体和关系的情况。例如，将圣地亚哥的事件包括在我们专门针对智利旅游业的知识图中，可能会影响知识图的扩展简洁性，可能会返回与给定领域无关的结果。一般来说，简洁性可以通过与领域相关的属性、类别、形状、实体、关系等的比率来衡量，这反过来可能需要一个黄金标准或评估领域相关性的技术。
### 7.4.2表示简洁性
是指内容在知识图中紧凑表示的程度，知识图可以是内涵的，也可以是外延的。例如，有两个属性flight和files to以服务于同一目的会对代表性简洁的内涵形式产生负面影响，而有两个节点Santiago和Santiago de Chile代表智利首都（两者都不关联）会影响代表性简洁的外延形式。表达简洁性的另一个例子是不必要地使用复杂的建模结构，例如不必要地使用物化，或者在元素顺序不重要时使用链表。尽管代表性的简洁性很难评估，但可以使用冗余节点数量等衡量指标。
### 7.4.3可理解性
是指人类用户能够轻松地解释数据，而不会产生歧义，这至少涉及到提供人类可读的标签和描述（最好使用不同的语言），使他们能够理解正在谈论的内容。回过头来看图1，虽然节点EID15和EID16用于确保事件的唯一标识符，但它们也应该与标签相关联，如Nam和Food Truck。理想情况下，人类可读的信息足以消除特定节点的歧义，例如将描述“Santiago, the capital of Chile”与Santiago联系起来，以消除城市与同义词之间的歧义。可理解性的度量可能包括具有人类可读标签和描述的节点的比率、此类标签和描述的唯一性、支持的语言等。
## 7.5其他质量尺寸
我们已经讨论了一些关键的质量维度，这些维度已经针对知识图进行了讨论，并且通常适用于知识图。进一步的维度可能与特定域、特定应用程序或特定图形数据模型的上下文相关。更多详情，请参阅Zaveri等人的调查，以及巴蒂尼和斯坎纳皮科的书。