import json

from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """你的任务是根据给定的段落和命名实体列表，构建一个RDF（资源描述框架）图。
请以JSON列表的形式返回三元组，每个三元组代表RDF图中的一个关系，格式: {"triples": [["{subject}", "{predicate}", "{object}"]]}。

请注意以下要求：
- 每个三元组应包含至少一个，最好包含两个，来自每个段落命名实体列表中的实体。
- 明确消解代词，确保指代清晰。
- 三元组中实体和关系的顺序不重要。
- 请注意，段落中可能包含一些无关信息，这些信息不应出现在三元组中。
- 三元组的内容必须是真实的，不能是虚构的。也不要包含任何与段落无关的实体，实体和名词必须直接从段落中提取，不能翻译和转化。
- 不能出现\\u等转义字符。

"""


ner_conditioned_re_frame = """请将下列段落内容转换为一个JSON字典，包含三元组列表。

#extend_prompt#

段落内容：
```
{passage}
```

命名实体列表：
{named_entities}
"""


def named_entity_to_string(named_entities):
    return json.dumps(named_entities, ensure_ascii=False)


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entities=named_entity_to_string(json.loads(one_shot_ner_output)["named_entities"]))


ner_conditioned_re_output = """{"triples": [
            ["Radio City", "位于", "印度"],
            ["Radio City", "是", "私营FM电台"],
            ["Radio City", "开播于", "2001年7月3日"],
            ["Radio City", "播放语言", "印地语"],
            ["Radio City", "播放语言", "英语"],
            ["Radio City", "涉足", "新媒体"],
            ["Radio City", "推出", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "上线于", "2008年5月"],
            ["PlanetRadiocity.com", "是", "音乐门户"],
            ["PlanetRadiocity.com", "提供", "新闻"],
            ["PlanetRadiocity.com", "提供", "视频"],
            ["PlanetRadiocity.com", "提供", "歌曲"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]