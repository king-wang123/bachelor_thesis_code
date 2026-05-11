import argparse
import json
import os
import re

from tools import get_jsonline, save_jsonline


ROOT = '/data/250010072/zlh/king/code_tailored_dataset'


def clean_response(response):
    def repl(match):
        code = match.group(1).strip()
        fence = re.match(r'^```[a-zA-Z0-9_+#.-]*\s*\n(.*?)\n?```$', code, re.DOTALL)
        if fence:
            code = fence.group(1).strip()
        return f'<code>\n{code}\n</code>'

    return re.sub(r'<code>\s*(.*?)\s*</code>', repl, response.strip(), flags=re.DOTALL)


def base_item(item, response, source):
    return {
        'id': item['id'],
        'question': item['question'],
        'response': clean_response(response),
        'test_in': item.get('test_in', []),
        'test_out': item.get('test_out', []),
        'test_code': item.get('test_code', ''),
        'src_question': item.get('src_question', ''),
        'target_lang': item.get('target_lang', ''),
        'source': source,
    }


def merge_valid(path):
    merged = []
    for item in get_jsonline(path):
        responses = item.get('responses') or []
        if responses:
            merged.append(base_item(item, responses[0], 'valid'))
    return merged


def merge_from_wrong(path):
    merged = []
    for item in get_jsonline(path):
        attempts = item.get('from_wrong_attempts') or []
        for attempt in attempts:
            response = attempt.get('repaired_response') or ''
            if response:
                merged.append(base_item(item, response, 'from_wrong'))
                break
    return merged


def merge_multi_task(path):
    merged = []
    for item in get_jsonline(path):
        responses = item.get('responses') or []
        if not responses:
            continue

        merged.append(base_item(item, responses[0], 'multi_task'))
    return merged


def save_dataset_info(data_dir, file_name):
    path = os.path.join(data_dir, 'dataset_info.json')
    info = {}
    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)

    info['kodcode_reasoning_sft'] = {
        'file_name': file_name,
        'columns': {
            'prompt': 'question',
            'response': 'response',
        },
    }

    with open(path, 'w') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_file', type=str, default=os.path.join(ROOT, 'data/kodcode_reasoning_valid.jsonl'))
    parser.add_argument('--wrong_file', type=str, default=os.path.join(ROOT, 'data/kodcode_reasoning_from_wrong.jsonl'))
    parser.add_argument('--multi_file', type=str, default=os.path.join(ROOT, 'multi_lang_task/multi_task_reasoning.jsonl'))
    parser.add_argument('--save_file', type=str, default=os.path.join(ROOT, 'data/kodcode_reasoning_sft.jsonl'))
    return parser.parse_args()


def main():
    args = parse_args()

    data = []
    data.extend(merge_valid(args.valid_file))
    data.extend(merge_from_wrong(args.wrong_file))
    data.extend(merge_multi_task(args.multi_file))

    save_jsonline(data, args.save_file)
    save_dataset_info(os.path.dirname(args.save_file), os.path.basename(args.save_file))

    print(f'total: {len(data)}')
    print(f'save_file: {args.save_file}')


if __name__ == '__main__':
    main()
