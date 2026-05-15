from tools import get_jsonline, get_json
import random

################### reasoning data ##################
data = get_jsonline('/youtu-tuling-csp-llm/kodewang/bachelor_thesis_code/code_tailored_dataset/data/kodcode_reasoning.jsonl')
data = random.sample(data, min(20, len(data)))
for item in data:
    print("\n####   question  #####")
    print(item['question'])
    print("\n####   reasoning_response  #####")
    for trajectory in item['trajectory']:
        print(f'round : {trajectory["round"]}')
        print(f'role : {trajectory["role"]}')
        if trajectory["role"] == 'execution':
            print(f'feedback : {trajectory["feedback"]}')
        else:
            print(f'response : {trajectory["response"]}')
        input()
    print(f'final result: {item["final_passed"]}')
    input()
    print("***********************************************")