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