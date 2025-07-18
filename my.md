图检索排序



我来详细解释一下`graph_search_with_fact_entities`方法的实现。这是一个基于事实实体进行图搜索的核心方法，主要用于结合事实相似度和图结构来检索相关文档。

主要实现步骤：

1. **初始化权重映射**:
```python
linking_score_map = {}  # 存储短语到事实分数的映射
phrase_scores = {}      # 存储每个短语的所有事实分数
phrase_weights = np.zeros(len(self.graph.vs['name']))  # 初始化短语权重数组
passage_weights = np.zeros(len(self.graph.vs['name'])) # 初始化段落权重数组
```

2. **处理top-k事实并计算短语权重**:
```python
for rank, f in enumerate(top_k_facts):
    subject_phrase = f[0].lower()
    predicate_phrase = f[1].lower()
    object_phrase = f[2].lower()
    fact_score = query_fact_scores[top_k_fact_indices[rank]]
    
    # 对主语和宾语短语进行处理
    for phrase in [subject_phrase, object_phrase]:
        phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
        phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
        
        if phrase_id is not None:
            # 根据短语出现的文档数量调整权重
            phrase_weights[phrase_id] = fact_score
            if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])
```

3. **计算短语的平均事实分数**:
```python
for phrase, scores in phrase_scores.items():
    linking_score_map[phrase] = float(np.mean(scores))
```

4. **应用top-k过滤**:
```python
if link_top_k:
    phrase_weights, linking_score_map = self.get_top_k_weights(
        link_top_k,
        phrase_weights,
        linking_score_map
    )
```

5. **结合密集检索分数**:
```python
# 获取密集检索的文档分数
dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

# 将密集检索分数加入到段落权重中
for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
    passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
    passage_dpr_score = normalized_dpr_sorted_scores[i]
    passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
    passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
```

6. **合并权重并运行PPR**:
```python
# 合并短语权重和段落权重
node_weights = phrase_weights + passage_weights

# 运行个性化PageRank算法
ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
    node_weights, 
    damping=self.global_config.damping
)
```

这个方法的主要特点是：

1. **多源信息融合**：
   - 结合了事实相似度（通过`query_fact_scores`）
   - 图结构信息（通过`ent_node_to_chunk_ids`）
   - 密集检索分数（通过`dense_passage_retrieval`）

2. **权重调整机制**：
   - 根据短语在文档中的出现频率调整权重
   - 使用`passage_node_weight`参数控制密集检索分数的影响

3. **图算法应用**：
   - 使用个性化PageRank（PPR）算法进行最终的文档排序
   - 考虑了图的结构信息进行传播

4. **结果验证**：
   - 确保返回的文档数量与语料库大小一致
   - 验证至少有一个短语在图中被找到

这个实现体现了HippoRAG系统的核心思想：通过结合事实提取、图结构和密集检索来提供更准确的文档检索结果。




```text

提取下面文本所有命名实体，命名实体是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。指的是可以用专有名词（名称）标识的事物，一个命名实体一般代表唯一一个具体事物个体，包括人名、地名等。


文本内容：
2.1 主演-主演
剧名：都市方程式类型：生活播放时间：1988年4月11日监制：潘嘉德编审：彭济材、余咏珊主演：郭富城、胡枫、南红、林利、梁洁华、梁艺龄集数：152
相关资讯由阎建钢执导，闫妮、赵立新、冯远征、姜武等实力演员倾力打造的都市轻喜剧《煮妇也疯狂》正在热拍
其实两人在还珠到情深深雨濛濛之后，两人在各忙各的道路上越走越远，反倒是苏有朋和林心如还保持着亲密的友谊关系
3. 作者

3.1 作者-作者
《王爷通缉令：嫩妃哪里逃》是一部连载于新浪读书的小说，作者是蔷薇晚
《世界如此酷，单纯要有度》是2012年中国商业出版社出版的图书，作者是李兰芳
《大唐开国》，是2007年中华书局出版的一本历史类图书，作者是刘后滨
《一生八至》是2007年中原农民出版社出版的图书，作者是王立娜
《重生媚娘》是依偎在你怀里创作的网络小说，发表于17K小说网
《火影之青梅已死》是佐阿青创作的网络小说，发表于晋江文学网
《居民防空防灾应急手册（普及读本）》是2004年原子能出版社出版的图书，作者是北京市人民防空办公室
《星空的传说》是一部在17K小说网连载的小说，作者是浮木小生


```




### 修改
- [x] EmbeddingStore增加recall，召回doc和fact
  - search query
- [x] get_query_embeddings 修改只调用一次即可
- [x] 修改为孔明doc id & 接口
- [ ] dense_passage_retrieval 改为向量召回
- [ ] 构造新的核心对象，仅使用对应组建和函数，优化变量存储、数据的拉取、存储



