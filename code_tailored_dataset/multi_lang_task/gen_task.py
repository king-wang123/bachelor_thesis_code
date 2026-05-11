import argparse
import ast
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


ROOT = '/data/250010072/zlh/king/code_tailored_dataset'
sys.path.append(os.path.join(ROOT, 'code'))

from qwen_gen import QwenGen
from tools import get_jsonline, save_jsonline
from task_prompts import (
    extract_question,
    sample_transform,
    task_generation_prompt,
    task_generation_system_prompt,
)


thread_local = threading.local()
args = None


def get_model():
    if not hasattr(thread_local, 'model'):
        thread_local.model = QwenGen(
            ip=args.ip,
            port=args.port,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=task_generation_system_prompt(),
        )
    return thread_local.model


def convert_item(item):
    transform = sample_transform()
    prompt = task_generation_prompt(item, transform)
    response = get_model().response(prompt)
    question = extract_question(response)

    return {
        'id': f"{item['id']}-{transform['name']}",
        'src_question': item['question'],
        'question': question,
        'test_in': item.get('test_in', []),
        'test_out': item.get('test_out', []),
        'test_code': '',
    }


def has_io_tests(item):
    return _has_nonempty_list(item.get('test_in')) and _has_nonempty_list(item.get('test_out'))


def _has_nonempty_list(value):
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        text = value.strip()
        if not text or text == '[]':
            return False
        try:
            parsed = ast.literal_eval(text)
            return isinstance(parsed, list) and len(parsed) > 0
        except (SyntaxError, ValueError):
            return True
    return bool(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=os.path.join(ROOT, 'data/kodcode.jsonl'))
    parser.add_argument('--save_file', type=str, default=os.path.join(ROOT, 'multi_lang_task/multi_task_data.jsonl'))
    parser.add_argument('--ratio', type=float, default=0.15)
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--ip', type=str, default='10.120.0.102')
    parser.add_argument('--port', type=int, default=35000)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.7)
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    random.seed(args.seed)

    data = [item for item in get_jsonline(args.data_file) if has_io_tests(item)]
    sample_num = args.sample_num or int(len(data) * args.ratio)
    sample_num = min(len(data), max(1, sample_num))
    sampled_data = random.sample(data, sample_num)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        gen_data = list(tqdm(executor.map(convert_item, sampled_data), total=len(sampled_data)))

    save_jsonline(gen_data, args.save_file)


if __name__ == '__main__':
    main()
