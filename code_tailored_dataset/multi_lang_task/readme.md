为提升框架的泛化能力，

按一定比例掺入多语言与多任务数据，使模型在学习推理路径的同时获得跨语言、跨任务的泛化能力。

多语言：C/CPP, java, go
多任务：代码翻译，代码优化，其他任务（补充1-2个代码任务）


从/data/250010072/zlh/king/code_tailored_dataset/data/kodcode.jsonl 采样源数据，
设计prompt，引导模型把源任务进行转化，生成新的任务。
得到一批新的数据集，只保留id, 源id-转化类型，如 Code_Contests-Code_Contests_36239_OJ-java, 或者Code_Contests-Code_Contests_36239_OJ-translation
src_question, question（新生成的question），测试样例字段，test_in,test_out, test_code
确保数据干净，可溯源



总的比例是总数据集的15%，
设计一个转化类型的集合（要有对应的提示词等），
从源数据 random.sample 15% 的数据，然后对于每一个item，随机采样一个转化类型（语言，或者任务），
调用大模型（目前在10.120.0.102 35000端口部署了一个qwen3-next模型，可以直接使用）生成任务，保存到本地jsonl文件中



得到任务数据后，参考之前的代码实现，/data/250010072/zlh/king/code_tailored_dataset/code/gen_reasoning.py
按照同样的格式，生成新的推理数据，保存下来。prompt可能需要对应的调整


注意，这部分的数据和代码放到/data/250010072/zlh/king/code_tailored_dataset/multi_lang_task文件夹中，不要干扰之前实现的内容



参考我之前的代码，按照同样的风格写代码，确保简洁凝练，注重逻辑实现，不要有过多不必要的内容。生成数据调用模型时候要多线程并行

先写好代码，然后运行小批量数据进行测试，确保逻辑正确后，把任务挂到后台执行，记录好log日志（两个任务都要运行，确保任务生成完，直接进行推理生成）