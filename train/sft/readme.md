现在需要进行 SFT 训练

首先把之前得到的数据合并起来，实现 /data/250010072/zlh/king/code_tailored_dataset/code/merge_data.py，
得到一个完整的 reasoning jsonl文件，放到/data/250010072/zlh/king/code_tailored_dataset/data目录下
保留id，question，(一个，字符串不要列表) 四段式的response， testcase 等必要字段（用于SFT和后续RL），丢掉无关的数据
包括：kodcode_reasoning_valid.jsonl，kodcode_reasoning_from_wrong.jsonl，和新生成的multi_task_data

我们主要训练 Qwen2.5-Coder-1.5B-instruct 和 Qwen2.5-Coder-3B-instruct 这两个模型

模型权重在 /data/share 目录下
训练环境使用llama-factory， 目录在/data/250010072/zlh/king/LLaMA-Factory，
我记得之前配置的有conda环境，但是找不到了
你检查下我的conda有没有llama-factory环境，如果没有，新创建一个sft环境，安装配置

配置好环境后，准备SFT数据集，和训练代码
代码参考train/sft/train.sh，这是别的项目的训练脚本，直接在这上面修改
超参数等，自行设置,针对任务、数据、和模型，选择合适的超参数
然后运行脚本，分别训练两个模型，保存到train/sft/models目录下

你需要等模型开始训练后，监控一段时间日志，确保训练正常，估算下大致时间，然后结束任务
