import logging
import os

from hipporag.utils.config_utils import BaseConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

from src.hipporag import HippoRAG


def get_doc1():
    return  [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]


def get_doc2():
    """
    读取~/Downloads/doc/a目录下的txt文件，将所有文件内容返回
    对内容进行分段，每段130个字，分段之间冗余20个字
    """
    def split_text(text: str, chunk_size: int = 500, overlap: int = 200) -> list[str]:
        """
        将文本分段，每段指定长度，段间有重叠
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # 计算当前段的结束位置
            end = min(start + chunk_size, text_length)

            # 如果不是最后一段，尝试在句子或标点处截断
            if end < text_length:
                # 在chunk_size范围内找最后一个标点符号
                last_punct = max(
                    text.rfind('。', start, end),
                    text.rfind('！', start, end),
                    text.rfind('？', start, end),
                    text.rfind('；', start, end),
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind(';', start, end),
                )

                if last_punct != -1 and last_punct > start + chunk_size // 2:
                    end = last_punct + 1

            # 添加当前段
            chunks.append(text[start:end])

            # 更新下一段的起始位置，考虑重叠
            start = end - overlap if end < text_length else text_length

        return chunks

    # 存储所有文件内容的列表
    all_contents = []

    # 遍历目录下的所有文件
    # target_dir = "/Users/zhoutongsheng/Downloads/doc/a1"
    target_dir = "/Users/zhoutongsheng/Desktop/demo_data"
    # target_dir = "/data/docs"
    for filename in os.listdir(target_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(target_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 只添加非空内容
                        # 对内容进行分段
                        chunks = split_text(content)
                        all_contents.extend(chunks)
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")

    return all_contents


def main():
    # Prepare datasets and evaluation
    docs = get_doc2()

    save_dir = 'outputs/demo_for_me'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    os.environ["OPENAI_API_KEY"] = "empty"
    llm_base_url = "http://llm.demo.haizhi.com/v1"
    embedding_base_url = llm_base_url
    llm_model_name = 'qwen2.5-14b-instruct'
    embedding_model_name = 'acge_text_embedding'

    # save_dir = 'outputs/demo_for_me2'
    # os.environ["OPENAI_API_KEY"] = "sk-5b155b3fb7624508ad0ef09e3be776e2"
    # llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # embedding_base_url = "http://llm.demo.haizhi.com/v1"
    # llm_model_name = 'qwen3-4b'
    # embedding_model_name = 'acge_text_embedding'

    # Startup a HippoRAG instance
    global_config = BaseConfig()
    global_config.ie_max_workers = 1
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url,
        global_config=global_config
    )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        # "神雕侠侣谁导演的",
        # "胡歌和谁合作过？有哪些作品",
        # "女人不容易的主演是谁",
        # "芒果传媒有哪些作品",
        # "李兰芳有哪些作品",
        "心灵之花枯萎的原因是什么"
    ]

    g = hipporag.get_graph_info()
    print("graph_info", g)

    retrieve_result = hipporag.retrieve(queries=queries, num_to_retrieve=3)
    print("retrieve", retrieve_result)
    retrieve_result = hipporag.retrieve(queries=queries, num_to_retrieve=3)
    print("retrieve", retrieve_result)

    # result = hipporag.rag_qa(queries=queries)
    # print(f"1. {result[1]}")
    # print(f"2. {result[2]}")
    # print(f"3. {result[0]}")
    # print(result)


if __name__ == "__main__":
    print("-- start --")
    # dd = {'method': 'post', 'url': '/chat/completions', 'files': None, 'json_data': {'messages': [{'role': 'system', 'content': 'Your task is to extract named entities from the given paragraph. \nRespond with a JSON list of entities.\n'}, {'role': 'user', 'content': "Radio City\nRadio City is India's first private FM radio station and was started on 3 July 2001.\nIt plays Hindi, English and regional songs.\nRadio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."}, {'role': 'assistant', 'content': '{"named_entities":\n    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]\n}\n'}, {'role': 'user', 'content': 'IVORY）与来自于德国并两度获得奥斯卡的编剧鲁丝·普罗厄·贾布瓦拉（RUTHPRAWERJHABVALA）的第23次合作，而影片的故事依据则改编自彼特·卡梅隆（PETERCAMERON）所创作的一本同名小说\n记者 曾乐武警特战剧《利刃出击》将于28日在江苏卫视首播,该剧是著名导演刘猛继《我是特种兵》系列、《特警力量》后,首部表现武警特战题材的电视剧,被誉为武警题材的“雄性力作”\n乔·怀特是一位很擅长把握英国风韵的导演，从平凡的农家到华贵的古堡，《傲慢与偏见》的画面总能使观众感到19世纪初岛国上的清新空气参与影片《安娜·卡列尼娜 ANNA KARENINA》（2012年）\n11. 制片人\n12. 编剧\n13. 改编自'}], 'model': 'qwen-plus', 'max_tokens': 2048, 'n': 1, 'seed': None, 'temperature': 0}}
    # dd = {'method': 'post', 'url': '/chat/completions', 'files': None, 'json_data': {'messages': [{'role': 'system', 'content': 'Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. \nRespond with a JSON list of triples, with each triple representing a relationship in the RDF graph. \n\nPay attention to the following requirements:\n- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.\n- Clearly resolve pronouns to their specific names to maintain clarity.\n\n'}, {'role': 'user', 'content': 'Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\nParagraph:\n```\nRadio City\nRadio City is India\'s first private FM radio station and was started on 3 July 2001.\nIt plays Hindi, English and regional songs.\nRadio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.\n```\n\n{"named_entities":\n    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]\n}\n\n'}, {'role': 'assistant', 'content': '{"triples": [\n            ["Radio City", "located in", "India"],\n            ["Radio City", "is", "private FM radio station"],\n            ["Radio City", "started on", "3 July 2001"],\n            ["Radio City", "plays songs in", "Hindi"],\n            ["Radio City", "plays songs in", "English"],\n            ["Radio City", "forayed into", "New Media"],\n            ["Radio City", "launched", "PlanetRadiocity.com"],\n            ["PlanetRadiocity.com", "launched in", "May 2008"],\n            ["PlanetRadiocity.com", "is", "music portal"],\n            ["PlanetRadiocity.com", "offers", "news"],\n            ["PlanetRadiocity.com", "offers", "videos"],\n            ["PlanetRadiocity.com", "offers", "songs"]\n    ]\n}\n'}, {'role': 'user', 'content': 'Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\nParagraph:\n```\n的回忆》第18页\n作者 ：张利华出版社：机械工业出版社《华为研发》内容简介：《华为研发》是一本讲创业的书，其中有华为早期创业时的艰难、苦涩、屈身民宅的那段时光\n《唐诗鉴赏辞典·宋词鉴赏辞典》是高等教育出版社出版的图书，作者是张傲飞\n《建筑工程施工总承包招标文件编写范本》是2004年中国建筑工业出版社出版的图书，作者是刘小强\n《我主海洋》是在17K小说网连载的一部都市生活类小说，作者是生富\n（二）信件劳埃德一九四八年七月八日讨论三方合同补充修订内容，附以合同修订条款数页的来信，以及他（代理人）、老舍（作者）和浦爱德（译者）共同签署的《四世同堂》英译版权代理合同\n《邓广铭治史丛稿》(邓广铭,著)（北京大学出版社1997）\n4. 出品公司\n\n4.1 出品公司-出品公司\n现湖南芒果传媒有限公司（芒果影业）总裁，开拍电视剧有《一不小心爱上你》、《另一种灿烂生活》等\n5. 毕业院校\n\n5.\n```\n\n{"named_entities": ["\\u5f20\\u5229\\u534e", "\\u673a\\u68b0\\u5de5\\u4e1a\\u51fa\\u7248\\u793e", "\\u534e\\u4e3a\\u7814\\u53d1", "\\u9ad8\\u7b49\\u6559\\u80b2\\u51fa\\u7248\\u793e", "\\u5f20\\u50b2\\u98de", "\\u5efa\\u7b51\\u5de5\\u7a0b\\u65bd\\u5de5\\u603b\\u627f\\u5305\\u62db\\u6807\\u6587\\u4ef6\\u7f16\\u5199\\u8303\\u672c", "\\u4e2d\\u56fd\\u5efa\\u7b51\\u5de5\\u4e1a\\u51fa\\u7248\\u793e", "\\u5218\\u5c0f\\u5f3a", "\\u6211\\u4e3b\\u6d77\\u6d0b", "17K\\u5c0f\\u8bf4\\u7f51", "\\u751f\\u5bcc", "\\u52b3\\u57c3\\u5fb7", "\\u4e00\\u4e5d\\u56db\\u516b\\u5e74\\u4e03\\u6708\\u516b\\u65e5", "\\u8001\\u820d", "\\u6d66\\u7231\\u5fb7", "\\u56db\\u4e16\\u540c\\u5802", "\\u9093\\u5e7f\\u94ed", "\\u5317\\u4eac\\u5927\\u5b66\\u51fa\\u7248\\u793e", "\\u6e56\\u5357\\u8292\\u679c\\u4f20\\u5a92\\u6709\\u9650\\u516c\\u53f8", "\\u8292\\u679c\\u5f71\\u4e1a", "\\u4e00\\u4e0d\\u5c0f\\u5fc3\\u7231\\u4e0a\\u4f60", "\\u53e6\\u4e00\\u79cd\\u707f\\u70c2\\u751f\\u6d3b"]}\n'}], 'model': 'qwen-plus', 'max_tokens': 2048, 'n': 1, 'seed': None, 'temperature': 0}}
    # import json
    # print(json.dumps(dd))
    main()
