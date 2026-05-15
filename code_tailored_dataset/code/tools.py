import re
import json
import multiprocessing
from tqdm import tqdm


def check_reasoning_format(response):
    """
    Validate the XML-style reasoning format required by code_reasoning_system_prompt.

    Valid outputs must start with the four base phases:
    decomposition -> pseudocode -> code -> reflection.
    If the first reflection finds an error, the response may then regenerate from
    the earliest wrong phase onward:
    code -> reflection,
    pseudocode -> code -> reflection, or
    decomposition -> pseudocode -> code -> reflection.
    """
    if not isinstance(response, str) or not response.strip():
        return False

    blocks = _parse_reasoning_blocks(response)
    if blocks is None:
        return False

    labels = [label for label, _ in blocks]
    base = ['decomposition', 'pseudocode', 'code', 'reflection']
    valid_sequences = [
        base,
        base + ['code', 'reflection'],
        base + ['pseudocode', 'code', 'reflection'],
        base + ['decomposition', 'pseudocode', 'code', 'reflection'],
    ]

    if labels not in valid_sequences:
        return False

    for label, content in blocks:
        if not content.strip():
            return False
        if label == 'code' and _looks_like_markdown_code_block(content):
            return False

    final_label, final_content = blocks[-1]
    if final_label != 'reflection':
        return False

    # return 'All phases verified. Solution is complete.' in final_content
    return True


def _parse_reasoning_blocks(response):
    tag_pattern = re.compile(
        r'\s*<(decomposition|pseudocode|code|reflection)>\s*(.*?)\s*</\1>\s*',
        re.DOTALL,
    )
    blocks = []
    pos = 0

    while pos < len(response):
        match = tag_pattern.match(response, pos)
        if not match:
            return None
        blocks.append((match.group(1), match.group(2).strip()))
        pos = match.end()

    return blocks


def _looks_like_markdown_code_block(content):
    stripped = content.strip()
    return stripped.startswith('```') or stripped.endswith('```')


def extract_field(response: str, label: str = 'code', last: bool = True):
    """
    从响应文本中提取指定标签的内容，默认返回最后一个匹配项
    
    Args:
        response: 包含标签的文本
        label: 要提取的标签名（如 'code', 'think', 'sql'）
        last: True=返回最后一个，False=返回第一个
    
    Returns:
        提取到的内容（已strip），如果没找到则返回 None
    """
    pattern = rf'<{label}>(.*?)</{label}>'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if not matches:
        return None
        
    return matches[-1].strip() if last else matches[0].strip()

def extract_code_from_markdown(text):
    pattern = r'```(\w*)\n(.*?)\n```'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2)
    
    return text

def get_jsonline(path):
    return [json.loads(line) for line in open(path)]

def save_jsonline(datas, path):
    with open(path, 'w') as fw:
        for data in datas:
            print(json.dumps(data, ensure_ascii=False), file=fw)

def get_json(path):
    return json.load(open(path))

def save_json(datas, path):
    with open(path, 'w') as fw:
        print(json.dumps(datas, ensure_ascii=False, indent=4), file=fw)

def parallelize_with_multiprocessing(data_list, func, max_workers=4):
    # 适用于cpu密集性任务
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = []
        with tqdm(total=len(data_list)) as pbar:
            for result in pool.imap(func, data_list):
                results.append(result)
                pbar.update(1)
    return results


def parallelize(data_list, func, max_workers=4, on_result=None):
    """Thread-based parallel execution for IO-bound tasks (e.g., LLM API calls).

    Args:
        data_list: list of items to process
        func: function to apply to each item
        max_workers: number of concurrent threads
        on_result: optional callback(result) called when each result is ready,
                   useful for streaming writes (e.g., append to file per completion)
    Returns:
        list of results in original order
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(data_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): i for i, item in enumerate(data_list)}
        with tqdm(total=len(data_list)) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    if on_result and result is not None:
                        on_result(result)
                except Exception as e:
                    print(f"[Error] index {idx}: {e}")
                    results[idx] = None
                pbar.update(1)
    return results

def get_message_str(messages):
    message_str = ''
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if index != len(messages) - 1:
            message_str += f'<|im_start|>{role}\n{content}<|im_end|>\n'
        else:
            if message['role'] == 'user':
                message_str += f'<|im_start|>{role}\n{content}<|im_end|>\n|im_start|>assistant\n'
            elif message['role'] == 'assistant':
                message_str += f'<|im_start|>{role}\n{content}'

    return message_str
