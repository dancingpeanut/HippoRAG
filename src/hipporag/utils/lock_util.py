import threading
from collections import defaultdict


class LockDict:
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def delete(self, key):
        with self._lock:
            if key in self._data:
                del self._data[key]

    def items(self):
        with self._lock:
            return list(self._data.items())


class KeyLockDict:
    def __init__(self):
        self._data = {}
        self._locks = defaultdict(threading.Lock)  # 每个 key 一个 Lock
        self._locks_lock = threading.Lock()        # 锁保护 _locks 字典本身

    def _get_lock(self, key):
        with self._locks_lock:
            return self._locks[key]

    def set(self, key, value):
        lock = self._get_lock(key)
        with lock:
            self._data[key] = value

    def get(self, key, default=None):
        lock = self._get_lock(key)
        with lock:
            return self._data.get(key, default)

    def update(self, key, update_func):
        """
        对 key 的值执行函数操作，如累加、修改字段等，线程安全。
        """
        lock = self._get_lock(key)
        with lock:
            old_value = self._data.get(key)
            self._data[key] = update_func(old_value)

    def delete(self, key):
        lock = self._get_lock(key)
        with lock:
            self._data.pop(key, None)
