ner_system = """你的任务是从给定的段落中抽取命名实体，命名实体是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。指的是可以用专有名词（名称）标识的事物，一个命名实体一般代表唯一一个具体事物个体，包括人名、地名等。
请以JSON列表的形式返回所有实体，格式: {"named_entities": ["xx", "xx"]}。

- 不能虚构任何数据，不要包含任何与段落无关的实体，实体和名词必须直接从段落中提取，不能翻译和转化。

"""

one_shot_ner_paragraph = """Radio City
Radio City是印度第一家私人FM电台，于2001年7月3日开播。
它播放印地语、英语和地区歌曲。
Radio City最近于2008年5月进军新媒体，推出了音乐门户网站——PlanetRadiocity.com，提供与音乐相关的新闻、视频、歌曲和其他音乐相关内容。"""

one_shot_ner_output = """{"named_entities":
    ["Radio City", "印度", "2001年7月3日", "印地语", "英语", "2008年5月", "PlanetRadiocity.com"]
}
"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": """
#extend_prompt#

现在从以下段落中提取实体，返回JSON字典，是名称的数组，不用返回实体类型，格式{"named_entities": ["xx", "xx"]}。
段落：
${passage}
    """.strip()}
]