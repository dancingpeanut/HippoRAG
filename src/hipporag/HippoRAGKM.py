import functools
import logging
from typing import List, Set, Dict

from filelock import FileLock

from hipporag import HippoRAG
from hipporag.utils import db_util
from hipporag.utils.misc_utils import compute_mdhash_id

DB_FILE = 'outputs/db.sqlite'
LOCK_FILE = 'outputs/db.sqlite.lock'


def lock_exec(func):
    """无参数的文件锁装饰器，阻塞等待锁"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with FileLock(LOCK_FILE):  # 阻塞直到获取锁
            return func(*args, **kwargs)
    return wrapper


class HippoRAGKM(HippoRAG):

    def __init__(self, **kwargs):
        super().__init__(
            global_config=kwargs.get('global_config'),
            save_dir=kwargs.get('save_dir'),
            llm_model_name=kwargs.get('llm_model_name'),
            llm_base_url=kwargs.get('llm_base_url'),
            embedding_model_name=kwargs.get('embedding_model_name'),
            embedding_base_url=kwargs.get('embedding_base_url'),
            azure_endpoint=kwargs.get('azure_endpoint'),
            azure_embedding_endpoint=kwargs.get('azure_embedding_endpoint')
        )
        self.ent_id = kwargs['ent_id']
        self.kl_id = kwargs['kl_id']
        self.task_dict = kwargs.get('task_dict', {})

    def check_segs(self, seg_ids: List[str] | Set[str]):
        real_seg_ids = set(seg_ids)
        seg_infos = self.get_all_seg_infos(self.kl_id)
        db_seg_map = { seg['seg_id'] : seg['hipporag_seg_id'] for seg in seg_infos }
        db_seg_ids = set(db_seg_map.keys())

        # 计算增量和要删除的seg_id
        to_add_seg_ids = real_seg_ids - db_seg_ids
        to_delete_seg_ids = db_seg_ids - real_seg_ids

        logging.info(f"需要新增的seg_id: {to_add_seg_ids}")
        logging.info(f"需要删除的seg_id: {to_delete_seg_ids}")
        to_delete_hipporag_seg_ids = [db_seg_map[seg_id] for seg_id in to_delete_seg_ids]
        return to_add_seg_ids, to_delete_seg_ids, to_delete_hipporag_seg_ids

    def graph_retrieve(self, query: str, num_to_retrieve: int = 5):
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings([query])

        sorted_fact_hash_ids, sorted_fact_scores = self.get_fact_scores(query)
        logging.info(f'Get facts, query: {query}, length: {len(sorted_fact_hash_ids)}')
        top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, sorted_fact_hash_ids)
        logging.info(f'Rerank facts, query: {query}, top_k_fact_indices: {top_k_fact_indices}, top_k_facts: {top_k_facts}')

        if len(top_k_facts) == 0:
            logging.info('No facts found after reranking, searching documents. query: {query}')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
        else:
            logging.info(f'Graph search, query: {query}')
            sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                     link_top_k=self.global_config.linking_top_k,
                                                                                     sorted_query_fact_scores=sorted_fact_scores,
                                                                                     top_k_facts=top_k_facts,
                                                                                     top_k_fact_indices=top_k_fact_indices,
                                                                                     passage_node_weight=self.global_config.passage_node_weight)

        segments_map = {}
        sorted_hipporag_seg_ids = []
        for i, idx in enumerate(sorted_doc_ids[:num_to_retrieve]):
            hipporag_seg_id = self.passage_node_keys[idx]
            sorted_hipporag_seg_ids.append(hipporag_seg_id)
            segments_map[hipporag_seg_id] = {
                'content': self.chunk_embedding_store.get_row(hipporag_seg_id)["content"],
                'score': sorted_doc_scores[i]
            }
        seg_infos = self.get_seg_info_by_chunk_ids(self.kl_id, list(sorted_hipporag_seg_ids))
        for seg_info in seg_infos:
            segments_map[seg_info['hipporag_seg_id']]['seg_id'] = seg_info['seg_id']

        segments = [segments_map[sid] for sid in sorted_hipporag_seg_ids]
        logging.info(f'Graph search result, query: {query}, num_to_retrieve: {num_to_retrieve}, result:\n{segments}\n')

        entities = []
        relationships = []
        for fact in top_k_facts:
            entities.append({
                "id": "",
                "entity": fact[0],
                "description": ""
            })
            entities.append({
                "id": "",
                "entity": fact[2],
                "description": ""
            })
            relationships.append({
                "id": "",
                "source": fact[0],
                "target": fact[2],
                "description": fact[1]
            })

        result = {
            "kl_id": self.kl_id,
            "segments": segments,
            "entities": entities,
            "relationships": relationships
        }
        return result

    def graph_info(self):
        """

        返回：
{
  "nodes": [
    {
      "level": 0,
      "title": "大语言模型",
      "type": "ORGANIZATION",
      "description": "大语言模型是使用大量文本数据训练的深度学习模型，能够生成自然语言文本并理解语言含义，提供关于各种主题的深入知识和语言生产。",
      "source_id": "seg_b2023738621045f98f28e1498238b1bf",
      "degree": 0,
      "human_readable_id": 0,
      "id": "b45241d70f0e43fca764df95b2b81f77",
      "size": 0,
      "graph_embedding": null,
      "community": null,
      "top_level_node_id": "b45241d70f0e43fca764df95b2b81f77",
      "x": 0,
      "y": 0
    }
  ],
  "relationships": [
    {
      "source": "GPT-4",
      "target": "LLAMA",
      "weight": 8,
      "description": "GPT-4和LLaMA都是大语言模型的代表，具有相似的技术特性和应用场景。",
      "text_unit_ids": [
        "seg_49cb69d97d06417aab2d233c0c966842"
      ],
      "id": "4a67211867e5464ba45126315a122a8a",
      "human_readable_id": "0",
      "source_degree": 4,
      "target_degree": 4,
      "rank": 8
    }
  ],
  "communities": [],
  "community_reports": []
}
        """
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        # eval(fact_row_dict[id]['content'])
        nodes = {}
        relationships = []
        chunk_seg_id_map = {}

        for triple_content, chunks in self.proc_triples_to_docs.items():
            triple = eval(triple_content)
            seg_ids = set([])
            for chunk_id in chunks:
                if chunk_id not in chunk_seg_id_map:
                    r = self.get_seg_info_by_chunk_ids(self.kl_id, [chunk_id])
                    if r:
                        r = r[0]
                        chunk_seg_id_map[r['hipporag_seg_id']] = r['seg_id']
                if chunk_id in chunk_seg_id_map:
                    seg_ids.add(chunk_seg_id_map[chunk_id])

            for n in [triple[0], triple[2]]:
                if n not in nodes:
                    nodes[n] = {
                        "level": 0,
                        "title": n,
                        "type": "ORGANIZATION",
                        "description": "",
                        "source_id": seg_ids,
                        "degree": 0,
                        "human_readable_id": 0,
                        "id": "",
                        "size": 0,
                        "graph_embedding": None,
                        "community": None,
                        "top_level_node_id": "",
                        "x": 0,
                        "y": 0
                    }
                else:
                    nodes[n]['source_id'] = nodes[n]['source_id'] | seg_ids
            relationships.append({
                "source": triple[0],
                "target": triple[2],
                "weight": 8,
                "description": triple[1],
                "text_unit_ids": list(seg_ids),
                "id": "",
                "human_readable_id": "0",
                "source_degree": 0,
                "target_degree": 0,
                "rank": 1
            })

        for v in nodes.values():
            v['source_id'] = ','.join(list(v['source_id']))

        communities = self.graph.community_multilevel()
        community_infos: List[Dict] = []
        for i, community in enumerate(communities):
            community_id = str(i)
            community_infos.append({
                "id": community_id,
                "name": community_id,
                "description": ""
            })
            for v_id in community:
                entity_info = self.entity_embedding_store.get_row(self.graph.vs[v_id]['name'])
                if entity_info and entity_info['content'] in nodes:
                    nodes[entity_info['content']]['community'] = community_id

        result = {
            "nodes": list(nodes.values()),
            "relationships": relationships,
            "communities": community_infos,
            "community_reports": []
        }
        return result

    def create_graph(self, *args, **kwargs):
        """
 {
  "kl_id": "kl_testxx_001",
  "ent_id": "en_bb3c23f0277d11ee93163448edf8b044",
  "user_id": null,
  "seg_ids": [
    "seg_90be56999af4426786eea6bbec4b4d9b"
  ],
  "env": {
    "GRAPHRAG_LLM_TOKENS_PER_MINUTE": 60000,
    "GRAPHRAG_LLM_REQUESTS_PER_MINUTE": 100,
    "GRAPHRAG_LLM_CONCURRENT_REQUESTS": 1,
    "GRAPHRAG_LLM_REQUEST_TIMEOUT": 30000,
    "GRAPHRAG_API_KEY": "haizhi123456",
    "GRAPHRAG_LLM_API_BASE": "http://llm.demo.haizhi.com/v1/",
    "GRAPHRAG_LLM_MODEL": "Qwen3-32B-AWQ",
    "GRAPHRAG_LLM_TEMPERATURE": 0.9,
    "GRAPHRAG_LLM_TOP_P": 0.4,
    "GRAPHRAG_LLM_MAX_TOKENS": 19661,
    "GRAPHRAG_EMBEDDING_API_BASE": "http://llm.demo.haizhi.com/v1/",
    "GRAPHRAG_EMBEDDING_MODEL": "acge_text_embedding",
    "GRAPHRAG_ENTITY_EXTRACTION_ENTITY_TYPES": null
  },
  "seg_datas": [
    {
      "seg_content": "filename: llm.txt\ncontent: 大语言模型（英语：Large Language Model，简称LLM）是指使用大量文本数据训练的深度学习模型，使得该模型可以生成自然语言文本或理解语言\n文本的含义。这些模型可以通过在庞大的数据集上进行训练来提供有关各种主题的深入知识和语言生产 [1]。",
      "seg_id": "seg_90be56999af4426786eea6bbec4b4d9b",
      "seg_title": null,
      "seq_no": 1,
      "doc_id": "doc_517ac9508c8846fe993584a56d9f746c",
      "doc_name": "llm.txt"
    }
  ]
}
        """
        logging.info(f"新建graph: {kwargs['kl_id']}, length: {len(kwargs['seg_datas'])}")
        docs = [d['seg_content'] for d in kwargs['seg_datas']]
        self.index(docs)
        # hash_id = compute_mdhash_id(docs[0], self.chunk_embedding_store.namespace + "-")
        # r = self.chunk_embedding_store.get_row(hash_id)
        # logging.info(r)
        seg_infos = []
        for seg in kwargs['seg_datas']:
            info = {
                'kl_id': kwargs['kl_id'],
                'seg_id': seg['seg_id'],
                'hipporag_seg_id': compute_mdhash_id(seg['seg_content'], self.chunk_embedding_store.namespace + "-"),
            }
            seg_infos.append(info)
        self.save_seg_infos(seg_infos)

    def update_graph(self, *args, **kwargs):
        """
{
  "kl_id": "kl_501ba5fb39d545a5885dd493b38ec1fb",
  "ent_id": "en_bb3c23f0277d11ee93163448edf8b044",
  "user_id": null,
  "use_cache": true,
  "seg_ids": [
    "seg_90be56999af4426786eea6bbec4b4d9b"
  ],
  "env": {
    "GRAPHRAG_LLM_TOKENS_PER_MINUTE": 60000,
    "GRAPHRAG_LLM_REQUESTS_PER_MINUTE": 100,
    "GRAPHRAG_LLM_CONCURRENT_REQUESTS": 1,
    "GRAPHRAG_LLM_REQUEST_TIMEOUT": 30000,
    "GRAPHRAG_API_KEY": "haizhi123456",
    "GRAPHRAG_LLM_API_BASE": "http://llm.demo.haizhi.com/v1/",
    "GRAPHRAG_LLM_MODEL": "Qwen3-32B-AWQ",
    "GRAPHRAG_LLM_TEMPERATURE": 0.9,
    "GRAPHRAG_LLM_TOP_P": 0.4,
    "GRAPHRAG_LLM_MAX_TOKENS": 19661,
    "GRAPHRAG_EMBEDDING_API_BASE": "http://llm.demo.haizhi.com/v1/",
    "GRAPHRAG_EMBEDDING_MODEL": "acge_text_embedding",
    "GRAPHRAG_ENTITY_EXTRACTION_ENTITY_TYPES": null
  },
  "seg_datas": [
    {
      "seg_content": "filename: llm.txt\ncontent: 大语言模型（英语：Large Language Model，简称LLM）是指使用大量文本数据训练的深度学习模型，使得该模型可以生成自然语言文本或理解语言\n文本的含义。这些模型可以通过在庞大的数据集上进行训练来提供有关各种主题的深入知识和语言生产 [1]。",
      "seg_id": "seg_90be56999af4426786eea6bbec4b4d9b",
      "seg_title": null,
      "seq_no": 1,
      "doc_id": "doc_517ac9508c8846fe993584a56d9f746c",
      "doc_name": "llm.txt"
    }
  ]
}
        """
        kl_id = kwargs['kl_id']
        real_seg_ids = set(kwargs['seg_ids'])

        to_add_seg_ids, to_delete_seg_ids, to_delete_hipporag_seg_ids = self.check_segs(real_seg_ids)

        logging.info(f"删除段落数: {len(to_delete_hipporag_seg_ids)}")
        if to_delete_hipporag_seg_ids:
            to_delete_docs = [d['content'] for d in self.chunk_embedding_store.get_rows(to_delete_hipporag_seg_ids).values() if d]
            if to_delete_docs:
                self.delete(to_delete_docs)
            self.delete_seg_infos(kl_id, list(to_delete_seg_ids))

        logging.info(f"新增段落数: {len(to_add_seg_ids)}")
        if to_add_seg_ids:
            to_add_docs = []
            seg_infos = []
            for seg in kwargs['seg_datas']:
                if seg['seg_id'] in to_add_seg_ids:
                    to_add_docs.append(seg['seg_content'])
                    info = {
                        'kl_id': kwargs['kl_id'],
                        'seg_id': seg['seg_id'],
                        'hipporag_seg_id': compute_mdhash_id(seg['seg_content'], self.chunk_embedding_store.namespace + "-"),
                    }
                    seg_infos.append(info)
            self.index(to_add_docs)
            self.save_seg_infos(seg_infos)

    @lock_exec
    def save_seg_infos(self, seg_infos: List[dict[str, str]]):
        """
        将seg_ids保存到数据库，使用sqlite，表为SEG_INFO(string kl_id, string seg_id, string hipporag_seg_id)
        """
        sqls = []
        params = []
        for seg in seg_infos:
            kl_id = seg['kl_id']
            seg_id = seg['seg_id']
            hipporag_seg_id = seg['hipporag_seg_id']
            sqls.append("INSERT OR REPLACE INTO SEG_INFO (kl_id, seg_id, hipporag_seg_id) VALUES (?, ?, ?)")
            params.append((kl_id, seg_id, hipporag_seg_id))
        db_util.execute_sqls(sqls, params)

    @lock_exec
    def get_all_seg_infos(self, kl_id: str):
        """
        从数据库中获取所有seg_ids，返回 List[dict]，每个dict包含 kl_id、seg_id、hipporag_seg_id
        """
        rows = db_util.execute_sql('SELECT seg_id, hipporag_seg_id FROM SEG_INFO where kl_id = ?', (kl_id,))
        return [
            {'seg_id': row[0], 'hipporag_seg_id': row[1]}
            for row in rows
        ]

    @lock_exec
    def get_seg_info_by_chunk_ids(self, kl_id: str, chunk_ids: List[str]):
        """
        从数据库中获取seg_ids，返回 List[dict]，每个dict包含 kl_id、seg_id、hipporag_seg_id
        """
        if not chunk_ids:
            return []
        placeholders = ','.join('?' for _ in chunk_ids)
        sql = f'SELECT seg_id, hipporag_seg_id FROM SEG_INFO WHERE kl_id = ? AND hipporag_seg_id IN ({placeholders})'
        rows = db_util.execute_sql(sql, [kl_id] + chunk_ids)
        return [
            {'kl_id': kl_id, 'seg_id': row[0], 'hipporag_seg_id': row[1]}
            for row in rows
        ]

    @lock_exec
    def get_seg_info_by_seg_ids(self, kl_id: str, seg_ids: List[str]):
        """
        从数据库中获取seg_ids，返回 List[dict]，每个dict包含 kl_id、seg_id、hipporag_seg_id
        """
        if not seg_ids:
            return []
        placeholders = ','.join('?' for _ in seg_ids)
        sql = f'SELECT seg_id, hipporag_seg_id FROM SEG_INFO WHERE kl_id = ? AND seg_id IN ({placeholders})'
        rows = db_util.execute_sql(sql, [kl_id] + seg_ids)
        return [
            {'kl_id': kl_id, 'seg_id': row[0], 'hipporag_seg_id': row[1]}
            for row in rows
        ]

    @lock_exec
    def delete_seg_infos(self, kl_id: str, seg_ids: List[str]):
        """
        从数据库中删除指定kl_id和seg_ids的记录
        """
        if not seg_ids:
            return
        placeholders = ','.join('?' for _ in seg_ids)
        sql = f'''
            DELETE FROM SEG_INFO
            WHERE kl_id = ?
            AND seg_id IN ({placeholders})
        '''
        db_util.execute_sql(sql, [kl_id] + seg_ids)
