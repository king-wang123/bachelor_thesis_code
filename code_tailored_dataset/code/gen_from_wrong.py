import argparse
import random

from prompt_template import wrong_reasoning_prompt, wrong_reasoning_system_prompt
from qwen_gen import QwenGen
from tools import (
    check_reasoning_format,
    extract_field,
    get_jsonline,
    parallelize_with_multiprocessing,
    save_jsonline,
)


args = None
models = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--ips', nargs='+', default=['10.120.0.102'])
    parser.add_argument('--ports', nargs='+', type=int, default=[35000])
    parser.add_argument('--gen_num', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--lang', type=str, default='python')
    parser.add_argument(
        '--data_file',
        type=str,
        default='/data/250010072/zlh/king/code_tailored_dataset/data/kodcode_reasoning_invalid.jsonl',
    )
    parser.add_argument(
        '--save_file',
        type=str,
        default='/data/250010072/zlh/king/code_tailored_dataset/data/kodcode_reasoning_from_wrong.jsonl',
    )
    parser.add_argument(
        '--skip_malformed',
        action='store_true',
        help='Skip attempts that do not contain decomposition, pseudocode, and code.',
    )
    return parser.parse_args()


def build_models(parsed_args):
    system_prompt = wrong_reasoning_system_prompt()
    built_models = []
    for ip in parsed_args.ips:
        for port in parsed_args.ports:
            model = QwenGen(
                ip=ip,
                port=port,
                temperature=parsed_args.temperature,
                system_prompt=system_prompt,
                max_tokens=parsed_args.max_tokens,
            )
            built_models.append(model)
    return built_models


def wrap_xml(label, content):
    return f'<{label}>\n{content.strip()}\n</{label}>'


def build_wrong_context(response, fallback_code=''):
    decomposition = extract_field(response, 'decomposition', last=False)
    pseudocode = extract_field(response, 'pseudocode', last=False)
    code = extract_field(response, 'code', last=False) or fallback_code

    missing = []
    if not decomposition:
        missing.append('decomposition')
    if not pseudocode:
        missing.append('pseudocode')
    if not code or not code.strip():
        missing.append('code')

    return {
        'decomposition': decomposition or '',
        'pseudocode': pseudocode or '',
        'code': (code or '').strip(),
        'missing_fields': missing,
    }


def compose_repaired_response(context, continuation):
    continuation = continuation.strip()
    prefix = '\n'.join([
        wrap_xml('decomposition', context['decomposition']),
        wrap_xml('pseudocode', context['pseudocode']),
        wrap_xml('code', context['code']),
    ])

    if continuation.startswith('<reflection>'):
        return f'{prefix}\n{continuation}'

    # Occasionally the model may ignore the continuation instruction and emit
    # a full response. Preserve it if it already satisfies the target format.
    if continuation.startswith('<decomposition>') and check_reasoning_format(continuation):
        return continuation

    return f'{prefix}\n{continuation}'


def infer_attempt(item, response, code):
    context = build_wrong_context(response, code)
    if context['missing_fields']:
        return {
            'wrong_context': context,
            'reflection_response': '',
            'repaired_response': '',
            'repaired_code': '',
            'format_valid': False,
            'skip_reason': 'missing_' + ','.join(context['missing_fields']),
        }

    prompt = wrong_reasoning_prompt(
        question=item['question'],
        decomposition=context['decomposition'],
        pseudocode=context['pseudocode'],
        code=context['code'],
        solution=item['solution'],
        tgt_lang=args.lang,
    )

    continuation = random.choice(models).response(prompt)
    repaired_response = compose_repaired_response(context, continuation)
    repaired_code = extract_field(repaired_response, 'code') or ''

    return {
        'wrong_context': context,
        'reflection_response': continuation,
        'repaired_response': repaired_response,
        'repaired_code': repaired_code,
        'format_valid': check_reasoning_format(repaired_response),
        'skip_reason': '',
    }


def inference(item):
    attempts = []
    responses = item.get('responses', [])
    gen_codes = item.get('gen_codes', [])

    for response, code in zip(responses, gen_codes):
        context = build_wrong_context(response, code)
        if context['missing_fields']:
            result = {
                'wrong_context': context,
                'reflection_response': '',
                'repaired_response': '',
                'repaired_code': '',
                'format_valid': False,
                'skip_reason': 'missing_' + ','.join(context['missing_fields']),
            }
            if not args.skip_malformed:
                attempts.append(result)
            continue

        for _ in range(args.gen_num):
            attempts.append(infer_attempt(item, response, code))

    item['from_wrong_attempts'] = attempts
    item['responses_from_wrong'] = [x['repaired_response'] for x in attempts if x['repaired_response']]
    item['gen_codes_from_wrong'] = [x['repaired_code'] for x in attempts if x['repaired_code']]
    item['format_valid_from_wrong'] = [x['format_valid'] for x in attempts]
    return item


def main():
    global args, models
    args = parse_args()
    models = build_models(args)

    data = get_jsonline(args.data_file)
    worker_num = max(1, args.bsz * max(1, len(args.ports)))
    gen_data = parallelize_with_multiprocessing(data, inference, worker_num)
    save_jsonline(gen_data, args.save_file)


if __name__ == '__main__':
    main()

# python /data/250010072/zlh/king/code_tailored_dataset/code/gen_from_wrong.py