# title
[AMIE: Association Rule Mining under Incomplete Evidence
in Ontological Knowledge Bases](https://dl.acm.org/doi/epdf/10.1145/2488388.2488425)

# Abstract
**Inductive Logic Programming(ILP)** 可以挖掘KB中的信息。
KB的数据挖掘规则异化为两个方面：
- 现有规则挖掘系统不适合大量数据的KB
- ILP要求反例，现有KB不能提供。

本文提出了新的挖掘规则以支持OWA场景。

# Conclusion
1. 提出了一个方法用于挖掘Horn规则在RDF的KB上。
2. 介绍了一个用于在OWA场景下进行规则挖掘的标准模型，一个新的方式用来模拟反例和一个可扩展的挖掘算法。
3. AMIE不需额外输入和参数调整。
4. AMIE在百万的事实上，运行只需几分钟，比SOTA方法更好。
5. 置信度方法可以合理地预测规则的准确性。
6. 未来的工作：
   - 计划考虑KB的T-Box生成更准确的规则。
   - 探索几个规则在同一事实上的协同效应，并且扩展给予Horn规则的规则集，所以可以预测更复杂的事实和隐藏的知识。

# Introduction
&#8195;&#8195;如果一个KB中存在一个节点包含motherOf和marriedTo关系，那么可以推理出$motherOf(m,c)^marriedTo(m,f)\rightarrow fatherOf(f,c)$。这种规则在大多时候都适用。寻找类似的规则有四个目的：
1. 推理新的事实
2. 找到KB中潜在的错误
3. 用于推理
4. 更好的理解数据
---
&#8195;&#8195;本文目的在于从KB中挖掘规则。聚焦在RDF风格的KB上。因为RDF是在OWA下的。
- OWA：不包含在OWA中的事实不一定是错的。
- (Closed World Assumption)CWA：不包含在CWA中的事实是错的。
---
&#8195;&#8195;已经有了在**关联规则挖掘(Association rule mining)** 和**ILP** 下的规则挖掘研究。关联规则挖掘是类似于“如果客户买了啤酒和葡萄酒，他们也可能买阿司匹林”。不过置信度无法保证，因为关联规则挖掘是基于CWA的，规则预测实体在数据库只有较低的置信度。

---
&#8195;&#8195;ILP通过一般的事实推理逻辑规则。不过在现有KB上应用ILP是不合适的：
- RDF风格数据库不包含反例。并且不在数据库中的事实也不一定是反例。
- ILP系统很慢，不适合大规模的KB。
---
&#8195;&#8195;本文具体来说提供了一下贡献：
1. 用于在正KB中模拟反例的方法(the Partial Completeness Assumption)
2. 一个高效的规则挖掘算法
3. 一个名为AMIE的系统，可以在几分钟内挖掘数百万个事实的规则，而无需参数调整或专家输入。