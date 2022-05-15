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
# 3 模式，特性，语境
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
### 3.1.3 涌现模式
