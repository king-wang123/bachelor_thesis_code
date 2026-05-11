# Code Tailored 模型评测脚本

此目录包含用于对训练好的模型进行编码任务基准测试的评测脚本。

## 数据集

以下数据集位于 `benchmark/` 目录中：
- `mbpp.jsonl`: MBPP 数据集
- `codeforces.jsonl`: HumanEval 数据集（重命名为 codeforces）
- `livecodebench_v6.jsonl`: LiveCodeBench V6 数据集
- `code_contests.jsonl`: CodeContests 数据集

每个数据集均为 JSONL 格式，具有以下结构：
- `question`: 编程问题描述
- `test_code`: 用于 pytest 的测试代码（针对 MBPP）或空字符串
- `test_ins`: 测试用例的输入字符串列表
- `test_outs`: 测试用例的预期输出字符串列表

## 脚本

- `evaluate.py`: 主评测脚本，使用模型生成代码并针对数据集进行评测。
- `qwen_gen.py`: 通过 OpenAI API 调用训练模型的模型推理类。
- `tools.py`: 从响应中提取代码的实用函数。
- `kodecode_exec.py`: 使用 pytest 或直接执行的代码执行函数。
- `run_code.py`: 用于输入/输出测试的直接代码执行。

## 设置

安装依赖项：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保训练好的模型在指定端口上运行（默认 35000）。
2. 运行评测：
   ```bash
   python evaluate.py
   ```

脚本将输出每个数据集的准确率。

## 注意事项

- 使用的 prompt 与训练 prompt 匹配，包括四阶段推理。
- 从模型响应的 `<code>` 标签中提取代码。
- 对于有 `test_code` 的数据集，使用 pytest；否则，使用直接执行并比较输入输出。
