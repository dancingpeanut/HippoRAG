import logging
import os
import sqlite3
from typing import List, Tuple

from filelock import FileLock

DB_FILE = 'outputs/db.sqlite'
if os.environ.get('DB_FILE'):
    DB_FILE = os.environ.get('DB_FILE')
DB_FILE_LOCK = DB_FILE + '.lock'


class get_connection:
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.conn = None
        self.lock = FileLock(DB_FILE_LOCK, timeout=self.timeout)

    def __enter__(self):
        # 获取文件锁并建立数据库连接
        self.lock.acquire()
        self.conn = sqlite3.connect(DB_FILE)
        return self.conn

    def __exit__(self, exc_type, exc_value, traceback):
        # 关闭数据库连接并释放文件锁
        if self.conn:
            self.conn.close()
        self.lock.release()
        # 不捕获异常，直接传播
        return False


def initialize_db():
    logging.info('initialize_db waiting for lock')
    with get_connection() as conn:
        logging.info('initialize_db got lock')
        c = conn.cursor()

        c.execute("""
            DROP TABLE IF EXISTS task_status
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS TASK (
                task_id TEXT PRIMARY KEY,
                kl_id TEXT,
                status TEXT,
                data TEXT,
                ctime TIMESTAMP,
                etime TIMESTAMP
            )
        """)

        c.execute('''
            CREATE TABLE IF NOT EXISTS SEG_INFO (
                kl_id TEXT,
                seg_id TEXT,
                hipporag_seg_id TEXT,
                PRIMARY KEY (kl_id, seg_id)
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS KG (
                kl_id TEXTPRIMARY KEY,
                env TEXT
            )
        ''')
        conn.commit()
        logging.info('initialize_db done')


initialize_db()


def execute_sql(sql, params=None):
    logging.info(f'execute_sql: {sql}, params: {params}')
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(sql, params)
        conn.commit()
        rows = c.fetchall()
    return rows


def execute_sqls(sqls: List[str], params: List[Tuple]):
    results = []
    with get_connection() as conn:
        for sql, params in zip(sqls, params):
            c = conn.cursor()
            c.execute(sql, params)
            rows = c.fetchall()
            results.append(rows)
        conn.commit()
    return results
