import random
import re


TRANSFORM_TYPES = [
    {
        'name': 'c',
        'target_lang': 'C',
        'task_kind': 'multi_language',
        'instruction': 'Rewrite the source programming problem as a standard competitive-programming task that asks for a complete C solution.',
    },
    {
        'name': 'cpp',
        'target_lang': 'C++17',
        'task_kind': 'multi_language',
        'instruction': 'Rewrite the source programming problem as a standard competitive-programming task that asks for a complete C++17 solution.',
    },
    {
        'name': 'java',
        'target_lang': 'Java 17',
        'task_kind': 'multi_language',
        'instruction': 'Rewrite the source programming problem as a standard competitive-programming task that asks for a complete Java 17 solution.',
    },
    {
        'name': 'go',
        'target_lang': 'Go',
        'task_kind': 'multi_language',
        'instruction': 'Rewrite the source programming problem as a standard competitive-programming task that asks for a complete Go solution.',
    },
    {
        'name': 'translation',
        'target_lang': 'C++17',
        'task_kind': 'code_translation',
        'instruction': 'Create a code translation task: give the Python reference program and ask the solver to translate it into C++17 while preserving stdin/stdout behavior.',
    },
    {
        'name': 'optimization',
        'target_lang': 'Python 3',
        'task_kind': 'code_optimization',
        'instruction': 'Create a code optimization task: describe the original problem and ask the solver to write a faster Python 3 program with the same input/output behavior.',
    },
    {
        'name': 'debugging',
        'target_lang': 'Python 3',
        'task_kind': 'code_debugging',
        'instruction': 'Create a debugging task: include a short buggy Python solution derived from the reference idea and ask the solver to fix it.',
    },
    {
        'name': 'edge_cases',
        'target_lang': 'Python 3',
        'task_kind': 'edge_case_repair',
        'instruction': 'Create an edge-case repair task: emphasize robust handling of boundary cases while preserving the original input/output behavior.',
    },
]


def sample_transform():
    return random.choice(TRANSFORM_TYPES)


def task_generation_system_prompt():
    return """You create clean coding-task data for supervised fine-tuning.

Return only one XML block:
<question>
...
</question>

Rules:
- The new task must be self-contained.
- Preserve the original input/output semantics exactly, so the original tests remain valid.
- Do not mention dataset construction, source id, hidden tests, ground truth, or that the task was transformed.
- Include a clear line near the top: Target Language: <language>.
- Keep examples and constraints if they are present in the source task.
- For translation, optimization, debugging, and edge-case tasks, make the task instruction natural and specific.
- Do not output a solution outside the question block.
"""


def task_generation_prompt(item, transform):
    return f"""Transformation Type: {transform['name']}
Task Kind: {transform['task_kind']}
Target Language: {transform['target_lang']}

Transformation Instruction:
{transform['instruction']}

Source Programming Problem:
{item['question'].strip()}

Reference Python Program:
{str(item.get('solution', '')).strip()}

Generate the transformed task now. Output only <question>...</question>.
"""


def reasoning_prompt(question, target_lang):
    return f"""**Programming Task:**
{question.strip()}

**Target Language:** {target_lang}

---

Please generate a complete response following the system template format.
For translation, optimization, debugging, or edge-case repair tasks, solve the requested task as stated and put the final complete executable program in the <code> block.
"""


def extract_question(response):
    match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def infer_target_lang(item_id, question):
    match = re.search(r'Target Language:\s*([^\n\r]+)', question, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    suffix = item_id.rsplit('-', 1)[-1].lower()
    if suffix == 'c':
        return 'C'
    if suffix == 'cpp':
        return 'C++17'
    if suffix == 'java':
        return 'Java 17'
    if suffix == 'go':
        return 'Go'
    if suffix == 'translation':
        return 'C++17'
    return 'Python 3'
