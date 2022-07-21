# title
![AMIE: Association Rule Mining under Incomplete Evidence
in Ontological Knowledge Bases](https://dl.acm.org/doi/epdf/10.1145/2488388.2488425)

# Abstract
**Inductive Logic Programming(ILP)**可以挖掘KB中的信息。
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