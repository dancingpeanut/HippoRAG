import os
import sqlite3
from typing import List, Tuple

from filelock import FileLock

DB_FILE = 'outputs/db.sqlite'
if os.environ.get('DB_FILE'):
    DB_FILE = os.environ.get('DB_FILE')
DB_FILE_LOCK = DB_FILE + '.lock'


def initialize_db(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS task_status (
            kl_id TEXT PRIMARY KEY,
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
    conn.commit()


def execute_sql(sql, params=None):
    with FileLock(DB_FILE_LOCK):
        conn = sqlite3.connect(DB_FILE)
        try:
            initialize_db(conn)
            c = conn.cursor()
            c.execute(sql, params)
            conn.commit()
            rows = c.fetchall()
        finally:
            conn.close()
    return rows

def execute_sqls(sqls: List[str], params: List[Tuple]):
    results = []
    with FileLock(DB_FILE_LOCK):
        conn = sqlite3.connect(DB_FILE)
        initialize_db(conn)
        for sql, params in zip(sqls, params):
            try:
                c = conn.cursor()
                c.execute(sql, params)
                conn.commit()
                rows = c.fetchall()
                results.append(rows)
            finally:
                conn.close()
    return results
