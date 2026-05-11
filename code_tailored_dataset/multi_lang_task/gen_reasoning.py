import argparse
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


ROOT = '/data/250010072/zlh/king/code_tailored_dataset'
sys.path.append(os.path.join(ROOT, 'code'))

from prompt_template import code_reasoning_system_prompt
from qwen_gen import QwenGen
from tools import extract_field, get_jsonline, save_jsonline
from task_prompts import infer_target_lang, reasoning_prompt


thread_local = threading.local()
args = None


def get_model():
    if not hasattr(thread_local, 'model'):
        thread_local.model = QwenGen(
            ip=args.ip,
            port=args.port,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=code_reasoning_system_prompt(),
        )
    return thread_local.model


def inference(item):
    target_lang = infer_target_lang(item['id'], item['question'])
    prompt = reasoning_prompt(item['question'], target_lang)

    item['target_lang'] = target_lang
    item['responses'] = []
    item['gen_codes'] = []

    for _ in range(args.gen_num):
        response = get_model().response(prompt)
        item['responses'].append(response)
        item['gen_codes'].append(extract_field(response, 'code') or '')

    return item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=os.path.join(ROOT, 'multi_lang_task/multi_task_data.jsonl'))
    parser.add_argument('--save_file', type=str, default=os.path.join(ROOT, 'multi_lang_task/multi_task_reasoning.jsonl'))
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--gen_num', type=int, default=1)
    parser.add_argument('--ip', type=str, default='10.120.0.102')
    parser.add_argument('--port', type=int, default=35000)
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--temperature', type=float, default=0.9)
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    random.seed(42)

    data = get_jsonline(args.data_file)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        gen_data = list(tqdm(executor.map(inference, data), total=len(data)))

    save_jsonline(gen_data, args.save_file)


if __name__ == '__main__':
    main()
