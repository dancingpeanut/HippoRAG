import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


class NERExtract(BaseModel):
    named_entities: List[str]


class TripleExtract(BaseModel):
    triples: List[List[str]]


def clear_char(text: str) -> str:
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\\(?!u)', '', text)
    text = text.strip()
    return text


def _extract_ner_from_response(real_response) -> List[str]:
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    response = clear_char(real_response)
    match = re.search(pattern, response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    return NERExtract.model_validate_json(match.group()).named_entities


def _extract_triples_from_response(real_response):
    pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
    response = clear_char(real_response)
    match = re.search(pattern, response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    # return eval(match.group())["triples"]
    return TripleExtract.model_validate_json(match.group()).triples


EXTENDED_PROMPT = """
提取要求：
{requirements}

段落内容：
{passage}
"""


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, max_workers: int, graph_entity_config=None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.graph_entity_config = graph_entity_config

    def extend_prompt(self, messages, is_ner=False):
        last_message = messages[-1]

        if self.graph_entity_config:
            extend_prompt = "提取要求：\n提取的实体类型必须是以下类型："
            entity_info = '"%s"' % '","'.join(
                [e['name'].strip() for e in self.graph_entity_config['entities']])
            extend_prompt += entity_info
            if not is_ner and self.graph_entity_config.get('relationships'):
                edge_info = '"%s"' % '","'.join(
                    [e['name'].strip() for e in self.graph_entity_config['relationships']])
                extend_prompt += "\n提取的实体关系必须是以下类型：" + edge_info
            extend_prompt += "\n" + self.graph_entity_config.get('extend_prompt', '').strip()
            last_message['content'] = extend_prompt + "\n段落内容：\n" + last_message['content']

        if self.llm_model.llm_name.lower().startswith('qwen3'):
            last_message['content'] = "/no_think\n" + last_message['content']
        logger.info(f"Extend prompt: {last_message['content']}")

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        self.extend_prompt(ner_input_message)
        raw_response = ""
        metadata = {}
        result = None
        for _ in range(3):
            try:
                # LLM INFERENCE
                raw_response, metadata, cache_hit = self.llm_model.infer(messages=ner_input_message)
                metadata['cache_hit'] = cache_hit
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                extracted_entities = _extract_ner_from_response(real_response)
                unique_entities = list(dict.fromkeys(extracted_entities))
                result = NerRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    unique_entities=unique_entities,
                    metadata=metadata
                )
                break
            except Exception as e:
                # For any other unexpected exceptions, log them and return with the error message
                logger.warning(f"Exception for ner extract chunk {chunk_key}, response: {raw_response}, error: {e}")
                metadata.update({'error': str(e)})
                result = NerRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,  # Store the error message in metadata
                    unique_entities=[],
                    metadata=metadata  # Store the error message in metadata
                )
                self.llm_model.del_cache(self.llm_model.gen_cache_key_hash(ner_input_message))
        return result

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )
        self.extend_prompt(messages)

        raw_response = ""
        metadata = {}
        result = None
        temperature = 0.7
        for _ in range(3):
            try:
                # LLM INFERENCE
                raw_response, metadata, cache_hit = self.llm_model.infer(messages=messages, temperature=temperature)
                metadata['cache_hit'] = cache_hit
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                extracted_triples = _extract_triples_from_response(real_response)
                triplets = filter_invalid_triples(triples=extracted_triples)
                result = TripleRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    metadata=metadata,
                    triples=triplets
                )
                break
            except Exception as e:
                logger.warning(f"Exception for triple extract chunk {chunk_key}, response: {raw_response}, error: {e}")
                metadata.update({'error': str(e)})
                result = TripleRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    metadata=metadata,
                    triples=[]
                )
                self.llm_model.del_cache(self.llm_model.gen_cache_key_hash(messages))
                temperature += 0.1

        # Success
        return result

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        logger.info(f"Running OpenIE on {len(chunks)} chunks, max workers: {self.max_workers}")

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                if metadata.get('error'):
                    raise Exception(f"Error in NER: {metadata.get('error')}")
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                if metadata.get('error'):
                    raise Exception(f"Error in triple_extraction: {metadata.get('error')}")
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
