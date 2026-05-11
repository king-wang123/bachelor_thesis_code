from tools import get_jsonline, get_json
import random

################### reasoning data ##################
data = get_jsonline('/data/250010072/zlh/king/code_tailored_dataset/data/kodcode_reasoning_100.jsonl')
# data = random.sample(data, min(100, len(data)))
for item in data:
    print("\n####   question  #####")
    print(item['question'])
    print("\n####   reasoning_response  #####")
    for reasoning_response in item['responses']:
        print(reasoning_response)
        input()
    print("***********************************************")