# PairRE

> - Linlin Chao, Jianshan He, Taifeng Wang, and Wei Chu. 2021. [PairRE: Knowledge Graph Embeddings via Paired Relation Vectors](https://aclanthology.org/2021.acl-long.336). In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 4360–4369, Online. Association for Computational Linguistics.



## Methodology



Model 1-to-N/N-to-1/N-to-N Complex Relation & enrich the capabilities for different relation patterns

为每个关系提出了一个具有配对向量的模型 (paired vectors for each relation)

($h $, $r $, $t $) $\rightarrow\text{PairRE} \rightarrow r$ 将关系嵌入作为成对向量，表示为 $[{\bf r}^H, {\bf r}^T]$

${\bf r}^H$  和 ${\bf r}^T$ 分别将头实体 $h$ 和尾实体 $t$ 投射到欧氏空间

投影操作是这两个向量之间的 Hadamard product

PairRE 然后计算两个投影向量的距离作为三元组的合理性

We want that ${\bf h} \circ {\bf r}^H \approx {\bf t} \circ {\bf r}^T $ when ($h, r, t$) holds, while  ${\bf h} \circ {\bf r}^H$ should be far away from ${\bf t} \circ {\bf r}^T$ otherwise.

在本文采用$L_1$-范数来衡量距离

![image-20221130212312047](https://gitee.com/knqmuzzi/picdata/raw/master/image-20221130212312047.png)

为了消除缩放自由，我们还在嵌入上添加了与之前基于距离的模型类似TransE的约束条件而这个约束只加在实体嵌入上。我们希望关系嵌入能够轻松而充分地捕捉到关系向量（如$PeopleBornHere$和$PlaceOfBirth$）和复杂特征（如1-N）之间的语义联系。对于实体嵌入，$L_2$-norm被设定为 1。

Source Function:
$$
\begin{equation}
\Large f_r(\bf{h}, \bf{t}) = -{||\bf{h}\circ\bf{r}^{\it H} - \bf{t}\circ\bf{r}^{\it T}||},
\end{equation}
$$
where ${\bf h}, {\bf r}^H, {\bf r}^T, {\bf t} \in \mathbb{R}^d $ and ${||\bf{h}||}^2 = {||\bf{t}||}^2 = 1$.

The model parameters are, all the entities’ embeddings, $\{\bf{e}_j\}_{j=1}^{\mathcal{E}} $ and all the relations’ embeddings,$\{\bf{r}_j\}_{j=1}^{\mathcal{R}} $.

	与TransE/RotatE相比，PairRE使实体在参与不同关系时有分布式的表示。我们还发现成对的关系向量能够对损失函数中的边际进行自适应调整，从而缓解了复杂关系的建模问题。
	
	让我们以1-N关系为例。为了更好地说明问题，我们将嵌入维度设为1，并删除对实体嵌入的约束。
	
	给定三联体$(h; r; ?)$，其中正确的尾部实体属于集合$S = \{t_1, t_2, ..., t_N\}$，PairRE通过让尾部实体预测
$$
\Large ||\bf{h} \circ \bf{r}^H - \bf{t}_i \circ \bf{r}^T|| < \gamma,
$$
where $\gamma$ is a fixed margin for distance based embedding models and $t_i \in S$.

The value of $\bf{t}_i$ should stay in the following range: 
$$
\Large 
\bf{t}_i\in 
\begin{cases}
((\bf{h}\circ\bf{r}^H-\gamma)/\bf{r}^T, (\bf{h}\circ\bf{r}^H+\gamma)/\bf{r}^T),\text{if } \bf{r}^T>0, \\ 
((\bf{h}\circ\bf{r}^H+\gamma)/\bf{r}^T, (\bf{h}\circ\bf{r}^H-\gamma)/\bf{r}^T),\text{if } \bf{r}^T<0, \\
(-\infty, +\infty), \text{otherwise}.
\end{cases}
$$
上述分析表明PairRE可以调整 ${\bf r}^T$的值以适应$S$中的实体。$S$的规模越大，${\bf r}^T$的绝对值越小。而像TransE或RotatE这样的模型对所有复杂关系类型都有一个固定的余量。当$S$的大小足够大时，这些模型将很难适合数据。对于N-1的关系，PairRE也可以自适应地调整 ${\bf r}^H$的值以适应数据。

同时，不添加特定关系的翻译矢量，使模型能够编码几个关键的关系模式。我们在下面展示这些能力。

### Proposition 1. 

> PairRE can encode symmetry/antisymmetry relation pattern.

$$
\Large

\text{If }(e_1, r_1, e_2) \in \mathcal{T}\text{ and }(e_2, r_1, e_1) \in \mathcal{T},

\\\Large

{\bf e}_1 \circ {\bf r}_1^H = {\bf e}_2 \circ {\bf r}_1^T \land {\bf e}_2 \circ {\bf r}_1^H = {\bf e}_1 \circ {\bf r}_1^T \\\Large \Rightarrow {{\bf r}_1^H}^2 = {{\bf r}_1^T}^2

\\\Large

\text{if }(e_1, r_1, e_2) \in \mathcal{T}\text{ and }(e_2, r_1, e_1) \notin \mathcal{T},

\\\Large

{\bf e}_1 \circ {\bf r}_1^H = {\bf e}_2 \circ {\bf r}_1^T \land {\bf e}_2 \circ {\bf r}_1^H \neq {\bf e}_1 \circ {\bf r}_1^T \\\Large \Rightarrow {{\bf r}_1^H}^2 \neq {{\bf r}_1^T}^2
$$

### Proposition 2.

> PairRE can encode inverse relation pattern.

$$
\Large \text{If }(e_1, r_1, e_2) \in \mathcal{T}\text{ and }(e_2, r_2, e_1) \in \mathcal{T}

\\\Large 
{\bf e}_1 \circ {\bf r}_1^H = {\bf e}_2 \circ {\bf r}_1^T \land {\bf e}_2 \circ {\bf r}_2^H = {\bf e}_1 \circ {\bf r}_2^T \\\Large \Rightarrow {{\bf r}_1^H} \circ\ {{\bf r}_2^H}  = {{\bf r}_1^T} \circ {{\bf r}_2^T}
$$

### Proposition 3.

> PairRE can encode composition relation pattern.

$$
\Large \text{If }(e_1, r_1, e_2) \in \mathcal{T}, (e_2, r_2, e_3) \in \mathcal{T}\text{ and }(e_1, r_3, e_3) \in \mathcal{T}
\\\Large 
{\bf e}_1 \circ {\bf r}_1^H = {\bf e_2} \circ {\bf r}_1^T  \land  {\bf e}_2 \circ {\bf r}_2^H = {\bf e_3} \circ {\bf r}_2^T  \land  &\\\Large  {\bf e}_1 \circ {\bf r}_3^H = {\bf e_3} \circ {\bf r}_3^T
\\\Large  \Rightarrow {\bf r}_1^T \circ {\bf r}_2^T \circ {\bf r}_3^H = {\bf r}_1^H \circ  {\bf r}_2^H  \circ  {\bf r}_3^T
$$

> Moreover, with some constraint, PairRE can also encode subrelations. 
>
> For a subrelation pair, $\forall h, t \in \mathcal{E}$ : $(h, r_1, t) \rightarrow (h, r_2, t)$, it suggests triple $(h, r_2, t)$ should be always more plausible than triple $(h, r_1, t)$. 
>
> In order to encode this pattern, PairRE should have the capability to enforce $f_{r_2}(h, r_2, t) \geq f_{r_1}(h, r_1, t)$.

### Proposition 4.

> PairRE can encode subrelation relation pattern using inequality constraint.

Assume a subrelation pair $r_1$ and $r_2$ that $\forall h, t \in \mathcal{E}$: $(h, r_1,t) {\rightarrow} (h, r_2, t)$.

We impose the following constraints:
$$
\begin{equation}
\Large \frac{{\bf r}_{2, i}^H}{{\bf r}_{1,i}^H} = \frac{{\bf r}_{2,i}^T}{{\bf r}_{1,i}^T} = {\bf \alpha}_i, |{\bf \alpha}_i| \leq 1,
\text{where }\alpha \in \mathbb{R}^d.
\end{equation}
$$

Then we can get

$$
\begin{equation}
\begin{aligned}
\Large &\Large f_{r_2}(h, t) - f_{r_1}(h, t) \\
&\Large = || {\bf h} \circ {\bf r}_1^H - {\bf t} \circ {\bf r}_1^T || -  || {\bf h} \circ {\bf r}_2^H - {\bf t} \circ {\bf r}_2^T || \\
&\Large = ||{\bf h} \circ {\bf r}_1^H - {\bf t} \circ {\bf r}_1^T ||- ||{\bf \alpha} \circ ({\bf h} \circ {\bf r}_1^H - {\bf t} \circ {\bf r}_1^T)|| \\
&\Large  \geq 0.
\end{aligned}
\end{equation}
$$

When the constraints are satisfied, PairRE forces triple $(h, r_2, t)$ to be more plausible than triple $(h, r_1, t)$.

### Optimization

> To optimize the model, we utilize the self-adversarial negative sampling loss as objective for training:

$$
\begin{equation}
  \begin{aligned}
\Large L = &\Large  -\log{\sigma(\gamma - f_r({\bf h}, {\bf t}))} \\
 &\Large - \sum_{i=1}^{n}{p(h_{i}^{'}, r, t_{i}^{'})}\log{\sigma(f_r({\bf h_{i}^{'}}, {\bf t_{i}^{'}}) - \gamma)},
 \end{aligned}
\label{Eq:loss}
\end{equation}
$$

where $\gamma $ is a fixed margin and $\sigma$ is the sigmoid function. 

($h_{i}^{'} $, $r $, $t_{i}^{'} $) is the $i^{th}$ negative triple and  $p(h_{i}^{'}, r, t_{i}^{'}) $ represents the weight of this negative sample. 

$p(h_{i}^{'}, r, t_{i}^{'}) $ is defined as follows:
$$
\begin{equation}
  \begin{aligned}
\Large p((h_{i}^{'}, r, t_{i}^{'})|(h, r, t)) = \frac{{\exp{ f_r(h_{i}^{'}, t_{i}^{'})}}}{{\sum_j{\exp{
f_r(h_{j}^{'}, t_{j}^{'})}}}}.
  \end{aligned}
\end{equation}
$$