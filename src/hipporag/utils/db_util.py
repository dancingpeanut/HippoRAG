import os
import sqlite3

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
