# 后处理，筛选
# 1. 过滤 code中 markdown 语法和空格
# 2. 过滤掉输出不符合四段式要求的数据
# 运行代码，确保正确性，合格的数据保留，不合格的单独留下

from tools import get_jsonline, save_jsonline, parallelize_with_multiprocessing, check_reasoning_format, extract_code_from_markdown
from kodecode_exec import kodcode_exec

data = get_jsonline('/data/250010072/zlh/king/code_tailored_dataset/data/kodcode_reasoning.jsonl')

# 把response和code中可能存在的markdown语法去掉，提取出纯代码
for item in data:
    responses = []
    for response in item['responses']:
        response = response.replace("```python", "").replace("```", "").strip()
    item['gen_codes'] = [extract_code_from_markdown(code) for code in item.get('gen_codes', [])]

def filter_item(item):
    for response, code in zip(item['responses'], item['gen_codes']):
        # 判断格式
        if code.strip() == "" or not check_reasoning_format(response):
            item['valid'] = False
            continue
        test_code = item['test_code']
        test_ins = item['test_in']
        test_outs = item['test_out']

        code = extract_code_from_markdown(code)
        passed = kodcode_exec(code, test_code, test_ins, test_outs)
        if passed:
            item['valid'] = True
        else:
            item['valid'] = False
    return item

# for item in data:
#     item = filter_item(item)
#     input()  # 手动检查每条数据，确认过滤结果是否合理

filtered_data = parallelize_with_multiprocessing(data, filter_item, 32)
valid_data = [item for item in filtered_data if item.get('valid', False)]
new_valid_data = []
for item in valid_data:
    new_valid_data.append({
        'id': item['id'],
        'question': item['question'],
        'test_code': item['test_code'],
        'test_in': item['test_in'],
        'test_out': item['test_out'],
        'response': item['responses'][0],  # 只保留第一条生成的结果
        'gen_code': item['gen_codes'][0],  # 只保留第一条生成的代码
    })
save_jsonline(valid_data, '/data/250010072/zlh/king/code_tailored_dataset/data/kodcode_reasoning_valid.jsonl')

invalid_data = [item for item in filtered_data if not item.get('valid', False)]
for item in invalid_data:
    del item['valid']
save_jsonline(invalid_data, '/data/250010072/zlh/king/code_tailored_dataset/data/kodcode_reasoning_invalid.jsonl')