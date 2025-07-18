import json
import re
import sqlite3
from typing import List

from pydantic import BaseModel

fp = 'outputs/en_bb3c23f0277d11ee93163448edf8b044/kl_f6efba45d8b74eda86715f3f6b21c98e/llm_cache/qwen2.5-14b-instruct_cache.sqlite'

conn = sqlite3.connect(fp)
c = conn.cursor()
c.execute("SELECT message, metadata FROM cache WHERE key = ?", ('a18f567a57508b3554b0872c93efe5cdce16206e5d718fa37d16dcb83e3b7839',))
row = c.fetchone()
conn.close()
if row is not None:
    message, metadata_str = row
    metadata = json.loads(metadata_str)

message2 = '{' + message.split('{')[1].split('}')[0] + '}'
data = json.loads(message2)


class TripleExtract(BaseModel):
    triples: List[List[str]]


tt = TripleExtract.model_validate_json(message2)




ss = '''

'''



# 还原Unicode转义字符
def decode_unicode(s):
    # 如果包含Unicode转义字符，进行解码
    if bool(re.search(r'\\u[0-9a-fA-F]{4}', s)):
        return s.encode('utf-8').decode('unicode_escape')
    return s  # 否则返回原字符串


s = "2021 \u5e74 1 \u6708 1 \u65e5 至 2026 \u5e74 12 \u6708 31 \u65e5"
decoded_str = s.encode('utf-8').decode('unicode_escape')
print(decoded_str)




