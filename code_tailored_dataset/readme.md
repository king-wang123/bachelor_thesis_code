# Code Tailored Dataset — 数据构造模块

## 概述

基于 **Decomposition → Pseudocode → Code → [Execution Feedback → Reflection]** 的迭代式代码生成范式，构建高质量 SFT 训练数据。

核心创新：**模型不感知工具调用**，pipeline 层自动提取代码 → 语法检查 → 执行测试 → 格式化反馈 → 模型进行 grounded reflection。

## Pipeline 架构

```
编程问题 → 模型生成 (decomposition + pseudocode + code)
                ↓
         提取 code → 语法检查 + 执行测试
                ↓
         格式化执行反馈 (通过/失败/错误信息)
                ↓
         模型基于反馈生成 reflection (+ 修复代码)
                ↓
         (迭代最多3轮，直到通过或达到上限)
                ↓
         保存完整推理轨迹
```

## 文件说明

### 核心 Pipeline
| 文件 | 功能 |
|------|------|
| `code/gen_reasoning.py` | **主入口** — 迭代式推理生成 pipeline |
| `code/prompt_template.py` | Prompt 模板（生成/反馈/反思） |
| `code/execution_feedback.py` | 语法检查 + 代码执行 → 结构化反馈 |
| `code/qwen_gen.py` | 模型客户端（支持多轮对话，自动绕过代理） |
| `code/tools.py` | 工具函数（JSONL 读写、XML 标签提取、格式校验） |
| `code/kodecode_exec.py` | KodCode 数据集专用执行框架 |
| `code/run_code.py` | 多语言代码执行器（Python/C/C++/Java/Go） |

### 后处理
| 文件 | 功能 |
|------|------|
| `code/select_reasoning.py` | 格式校验 + 执行验证，分为 valid/invalid |
| `code/gen_from_wrong.py` | 对 invalid 数据进行 reflection 修复 |
| `code/merge_data.py` | 合并所有数据为最终 SFT 数据集 |

### 多任务扩展
| 文件 | 功能 |
|------|------|
| `multi_lang_task/` | 15% 多语言多任务数据（翻译/优化/调试/边界修复） |

## 使用方式

```bash
# 生成推理数据（含执行反馈迭代）
python code/gen_reasoning.py \
    --data_file data/kodcode.jsonl \
    --save_file data/kodcode_reasoning.jsonl \
    --ports 35000 35001 \
    --max_iterations 3 \
    --temperature 0.7

# 参数说明
# --max_iterations: 最大执行-反馈-修复迭代轮次（默认3）
# --num_samples: 处理条数（0=全部）
# --ports: 模型服务端口列表
```

## 输出数据格式

每条数据包含：
- `response`: 完整的推理轨迹（用于 SFT 训练）
- `trajectory`: 结构化的每步记录（生成/执行/反思）
- `final_passed`: 最终代码是否通过所有测试
- `iterations_used`: 使用的迭代轮次

## 实验结果（20条测试）

| 指标 | 数值 |
|------|------|
| 最终通过率 | 80.0% (16/20) |
| 一次通过率 | 70.0% (14/20) |
| 平均迭代轮次 | 1.55 |
| 迭代修复成功 | 2/6 未一次通过的样本被修复 |
