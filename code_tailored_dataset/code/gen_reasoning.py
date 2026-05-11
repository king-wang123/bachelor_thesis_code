from qwen_gen import QwenGen
from tools import get_jsonline, save_jsonline, parallelize_with_multiprocessing, extract_field
from prompt_template import code_reasoning_system_prompt, basic_prompt
import random
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--ips', type=list, default=['10.120.6.217', '10.120.7.123'])
parser.add_argument('--ports', type=list, default=[35000, 35001])
parser.add_argument('--gen_num', type=int, default=1)
parser.add_argument('--max_tokens', type=int, default=16384)
parser.add_argument('--temperature', type=float, default=0.9)
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--data_file', type=str, default='/data/250010072/zlh/king/code_tailored_dataset/data/kodcode.jsonl')
parser.add_argument('--save_file', type=str)
args = parser.parse_args()

system_prompt = code_reasoning_system_prompt()

models = []
for ip in args.ips:
    for port in args.ports:
        model = QwenGen(ip=ip, port=port, temperature=args.temperature, system_prompt=system_prompt, max_tokens=args.max_tokens)
        models.append(model)

def inference(item):
    question = item['question']
    prompt = basic_prompt(question, args.lang)
    
    item['responses'] = []
    item['gen_codes'] = []

    for _ in range(args.gen_num):
        response = random.choice(models).response(prompt)
        # print(response)

        item['responses'].append(response)
        
        code = extract_field(response, 'code')
        if code is None:
            item['gen_codes'].append("")
        else:
            item['gen_codes'].append(code)

    return item


data = get_jsonline(args.data_file)
# for item in data:
#     inference(item)
#     input()
gen_data = parallelize_with_multiprocessing(data, inference, args.bsz * len(args.ports))
save_jsonline(gen_data, args.save_file)

# python /data/250010072/zlh/king/code_tailored_dataset/code/gen_reasoning.py --save_file 
