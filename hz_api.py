import glob
import json
import logging
import os
import re
import threading
import time
import traceback
from copy import deepcopy
from enum import Enum
from contextlib import asynccontextmanager
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

from fastapi import FastAPI, Request, Query
import uvicorn

from queue import Queue


from hipporag.HippoRAGKM import HippoRAGKM
from hipporag.utils import db_util
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.lock_util import LockDict

TASK_QUEUE = Queue()
# key: kl_id, value: 对象的queue
GRAPH_WORKERS = LockDict()

# 全局变量存储工作线程
worker_threads = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global worker_threads
    for i in range(2):
        t = threading.Thread(target=graph_run, daemon=True)
        t.start()
        worker_threads.append(t)
    logging.info(f"已启动 {len(worker_threads)} 个工作线程")
    
    yield
    
    # 关闭时执行
    logging.info("应用关闭中...")

app = FastAPI(lifespan=lifespan)


class TaskType(Enum):
    CREATE = "create"
    UPDATE = "update"


DEFAULT_ENV = {
    "GRAPHRAG_LLM_API_BASE": "http://llm.demo.haizhi.com/v1/",
    "GRAPHRAG_LLM_MODEL": "qwen-plus",
    "GRAPHRAG_EMBEDDING_API_BASE": "http://llm.demo.haizhi.com/v1/",
    "GRAPHRAG_EMBEDDING_MODEL": "acge_text_embedding",
}


CREATE_LOCK = threading.Lock()

OUTPUT_DIR = 'outputs'


def get_kl_model_info(kl_id: str, ent_id: str):
    """
    递归查找save_dir下所有包含graph.pickle的子目录名，以及openie_results_ner_xxx.json的文件名（只查顶层）。
    返回：{'graph_dirs': [目录名列表], 'openie_files': [文件名列表]}
    """
    import os
    save_dir = f'{OUTPUT_DIR}/{ent_id}/{kl_id}'
    graph_dirs = []
    for root, dirs, files in os.walk(save_dir):
        if 'graph.pickle' in files:
            graph_dirs.append(os.path.relpath(root, save_dir))  # 返回相对路径
    openie_files = []
    pattern = os.path.join(save_dir, 'openie_results_ner_*.json')
    for file_path in glob.glob(pattern):
        openie_files.append(os.path.basename(file_path))

    if len(graph_dirs) > 1:
        raise Exception(f"知识库`{kl_id}`下存在多个不同配置的图")
    if len(openie_files) == 0:
        logging.warning(f"知识库`{kl_id}`下不存在相关文件，格式：openie_results_ner_*.json")
        return None, None
    llm_name = re.findall(r'openie_results_ner_(.*?).json', openie_files[0])[0]
    embedding_name = graph_dirs[0].replace(llm_name + '_', '')
    return llm_name, embedding_name


def create_hipporag(ent_id: str, kl_id: str, env: dict):
    os.environ["OPENAI_API_KEY"] = env.get("GRAPHRAG_API_KEY", "empty")
    global_config = BaseConfig()
    global_config.ie_max_workers = int(os.environ.get("IE_MAX_WORKERS", 4))
    return HippoRAGKM(
        save_dir=f'{OUTPUT_DIR}/{ent_id}/{kl_id}',
        llm_model_name=env['GRAPHRAG_LLM_MODEL'],
        embedding_model_name=env['GRAPHRAG_EMBEDDING_MODEL'],
        llm_base_url=env['GRAPHRAG_LLM_API_BASE'],
        embedding_base_url=env['GRAPHRAG_LLM_API_BASE'],
        ent_id=ent_id,
        kl_id=kl_id,
        global_config=global_config
    )


class get_hipporag:

    def __init__(self, params: dict):
        self.params = params
        self.queue = None
        self.hipporag = None

    def __enter__(self):
        # 在进入上下文时获取session
        kl_id = self.params['kl_id']
        queue = GRAPH_WORKERS.get(kl_id)
        if not queue:
            with CREATE_LOCK:
                # 双重判断
                queue = GRAPH_WORKERS.get(kl_id)
                if not queue:
                    logging.info(f"初始化`{kl_id}`的runner队列")
                    queue = Queue(maxsize=1)
                    GRAPH_WORKERS.set(kl_id, queue)
                    ent_id = self.params['ent_id']
                    hipporag = create_hipporag(ent_id=ent_id, kl_id=kl_id, env=self.params['env'])
                    queue.put(hipporag)
        self.hipporag = queue.get()
        self.queue = queue
        return self.hipporag

    def __exit__(self, exc_type, exc_value, traceback):
        # 在退出上下文时关闭文件
        if self.hipporag:
            self.queue.put(self.hipporag)

        # 返回 False 表示任何异常都不会被捕捉，直接向外传播
        return False


def parse_segments(segments: List[Dict]):
    for seg in segments:
        res = re.findall(r'filename:[\s\S]*?content:([\s\S]+)', seg['seg_content'])
        if res:
            seg['seg_content'] = res[0]


def graph_run():
    while True:
        task = TASK_QUEUE.get()
        params = task['params']
        kl_id = params['kl_id']
        logging.info(f"接收到任务：{task['type']}, {params.get('kl_id')}, {params.get('env')}, {params.get('seg_ids')}")
        logging.info(f"当前任务队列：{TASK_QUEUE.qsize()}")
        with get_hipporag(params) as runner:
            logging.info(f"开始执行任务：{task['type']}, {params.get('kl_id')}")

            try:
                parse_segments(params.get('seg_datas', []))
                if task['type'] == TaskType.CREATE:
                    runner.create_graph(**params)
                elif task['type'] == TaskType.UPDATE:
                    runner.update_graph(**params)
                else:
                    raise Exception("未知的任务类型")
                update_task({
                    "kl_id": kl_id,
                    "state": "success",
                    "message": ""
                })
                logging.info(f"任务执行完毕：{task['type']}, {params.get('kl_id')}")
            except Exception as e:
                logging.error(f"任务执行失败：{task['type']}, {params.get('kl_id')}\n")
                logging.error(traceback.format_exc())
                update_task({
                    "kl_id": kl_id,
                    "state": "error",
                    "message": str(e)
                })


def update_task(data: dict):
    kl_id = data['kl_id']
    status = data['state']
    now_t = time.strftime('%Y-%m-%d %H:%M:%S')
    res = db_util.execute_sql("SELECT 1 FROM task_status WHERE kl_id = ?", (kl_id,))
    if not res:
        sql = f"INSERT INTO task_status(kl_id, status, data, ctime, etime) VALUES (?, ?, ?, ?, ?)"
        params = (kl_id, status, json.dumps(data), now_t, None)
    else:
        sql = f"UPDATE task_status SET status = ?, data = ?, etime = ? WHERE kl_id = ?"
        params = (status, json.dumps(data), now_t, kl_id)
    db_util.execute_sql(sql, params)
    logging.info(f"更新任务状态：{data['state']}, {data['message']}")


def get_task(kl_id: str):
    sql = "SELECT status, data FROM task_status WHERE kl_id = ?"
    rows = db_util.execute_sql(sql, (kl_id,))
    if rows:
        data = rows[0][1]
        return json.loads(data)
    else:
        return None


def put_task(task_type: TaskType, params: dict):
    update_task({
        "kl_id": params['kl_id'],
        "state": "running",
        "message": ""
    })
    TASK_QUEUE.put({
        'type': task_type,
        'params': params
    })
    logging.info(f"当前任务队列：{TASK_QUEUE.qsize()}")


@app.post("/controller/index/detail")
async def detail(request: Request):
    body = await request.body()
    params = json.loads(body)
    logging.info(f"获取详情：{params.get('kl_id')}")
    llm_name, embedding_name = get_kl_model_info(params['kl_id'], params['ent_id'])
    if llm_name is None:
        return {
            "nodes": [],
            "relationships": [],
            "communities": [],
            "community_reports": []
        }
    env = deepcopy(DEFAULT_ENV)
    env['GRAPHRAG_EMBEDDING_MODEL'] = embedding_name
    env['GRAPHRAG_LLM_MODEL'] = llm_name
    hipporag = create_hipporag(ent_id=params['ent_id'], kl_id=params['kl_id'], env=env)
    return hipporag.graph_info()


@app.post("/controller/index/seg_hit_test")
async def seg_hit_test(request: Request):
    body = await request.body()
    params = json.loads(body)
    logging.info(f"获取命中测试：{params.get('kl_id')}, {params.get('seg_ids')}")
    if not os.path.exists(f"{OUTPUT_DIR}/{params['ent_id']}/{params['kl_id']}"):
        logging.info(f"未找到实体：{OUTPUT_DIR}/{params['ent_id']}")
        return {
          "hit_segs": [],
          "miss_segs": params['seg_ids']
        }
    llm_name, embedding_name = get_kl_model_info(params['kl_id'], params['ent_id'])
    if llm_name is None:
        return {
            "hit_segs": [],
            "miss_segs": params['seg_ids']
        }
    env = deepcopy(DEFAULT_ENV)
    env['GRAPHRAG_EMBEDDING_MODEL'] = embedding_name
    env['GRAPHRAG_LLM_MODEL'] = llm_name
    hipporag = create_hipporag(ent_id=params['ent_id'], kl_id=params['kl_id'], env=env)
    to_add_seg_ids, to_delete_seg_ids, _ = hipporag.check_segs(params['seg_ids'])
    result = {
      "hit_segs": list(set(params['seg_ids']) - to_add_seg_ids),
      "miss_segs": list(to_add_seg_ids)
    }
    return result


@app.post("/controller/index/create")
async def create(request: Request):
    body = await request.body()
    params = json.loads(body)
    logging.info(f"接收到创建任务：{params.get('kl_id')}, {params.get('env')}, {params.get('seg_ids')}")
    put_task(TaskType.CREATE, params)
    return {"id": 0}


@app.post("/controller/index/update")
async def update(request: Request):
    body = await request.body()
    params = json.loads(body)
    logging.info(f"接收到更新任务：{params.get('kl_id')}, {params.get('env')}, {params.get('seg_ids')}")
    put_task(TaskType.UPDATE, params)
    return {"id": 0}


@app.get("/controller/index/state_2")
async def state(kl_id: str = Query(...)):
    return get_task(kl_id)


@app.post("/controller/index/delete")
async def delete():
    return {"errcode": 0, "message": ""}


@app.post("/controller/query/search_graph")
async def search(request: Request):
    body = await request.body()
    params = json.loads(body)
    result = []
    for kl_info in params['kl_id_infos']:
        hipporag_params = {
            "ent_id": params['ent_id'],
            "kl_id": kl_info['kl_id'],
            "env": kl_info['env']
        }
        logging.info(f"开始检索：{kl_info}")
        with get_hipporag(hipporag_params) as hipporag:
            logging.info(f"获取到实例：{kl_info}")
            res = hipporag.graph_retrieve(query=params['question'], num_to_retrieve=params.get('top_k', 5))
            result.append(res)
    return {"content": result}


if __name__ == "__main__":
    api_workers = int(os.environ.get('API_WORKERS', 1))
    uvicorn.run("demo_api:app", host="0.0.0.0", port=12085, reload=False, workers=api_workers)
