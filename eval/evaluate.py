import json
import os
from qwen_gen import QwenGen
from tools import extract_code_from_markdown
from kodecode_exec import kodcode_exec
from run_code import run_code

# Prompt template
def code_reasoning_system_prompt():
    return f"""
You are an expert software engineer. When solving any coding problem, you MUST strictly follow the four-phase structured reasoning pipeline below. Do NOT skip any phase, do NOT merge phases, and do NOT produce code before completing the prior phases.

---

## OUTPUT FORMAT (Mandatory)

Your entire response must be wrapped in the following four XML tags, in this exact order:

<decomposition>
{{Problem Analysis and Breakdown}}
</decomposition>
<pseudocode>
{{Pseudocode Representation}}
</pseudocode>
<code>
{{Final Executable Code}}
</code>
<reflection>
{{Reflection and Verification}}
</reflection>

---

## PHASE INSTRUCTIONS

### Phase 1 — <decomposition>
Before writing any code or pseudocode, perform a structured breakdown of the problem:
- **Restate the problem** in your own words to confirm understanding
- **Identify constraints**: data types, edge cases, input/output requirements, performance constraints
- **Outline the solution strategy**: describe your high-level approach without any code
- **Estimate complexity**: provide time and space complexity estimates for your intended approach

### Phase 2 — <pseudocode>
Translate your decomposition into structured pseudocode that serves as the reasoning bridge to implementation:
- Use clear, language-agnostic pseudocode
- Do NOT use actual programming language syntax — pseudocode only

### Phase 3 — <code>
Implement the final, directly executable code:
- The logic must exactly mirror the pseudocode from Phase 2
- Use the programming language specified by the user
- The code must be complete and runnable

### Phase 4 — <reflection>
Critically evaluate each of the three prior phases in sequence:
- **Decomposition check**: Was the problem correctly understood? Were all constraints and edge cases captured?
- **Pseudocode check**: Does the pseudocode correctly represent the intended algorithm?
- **Code check**: Does the code faithfully implement the pseudocode? Are there bugs, off-by-one errors, unhandled edge cases, or inefficiencies?

**Decision rule (strictly enforced)**:
- If ALL three phases are correct → state "All phases verified. Solution is complete." and stop.
- If ANY phase contains an error → explicitly state which phase has the error and why, then regenerate from that phase onward (re-output the corrected phase and all subsequent phases inside their respective XML tags).
"""

def get_prompt(question):
    system_prompt = code_reasoning_system_prompt()
    user_prompt = f"**Programming Problem:**\n{question.strip()}\n\n**Target Language:** python\n\n---\n\nPlease generate a complete response following the system template format:"
    return f"{system_prompt}\n\n{user_prompt}"

def load_dataset(dataset_path):
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def generate_code(generator, question):
    prompt = get_prompt(question)
    response = generator.response(prompt)
    code = extract_code_from_markdown(response)
    return code

def evaluate_code(code, test_code, test_ins, test_outs):
    if test_code.strip():
        # Use pytest
        passed = kodcode_exec(code, test_code, [], [])
    else:
        # Use run_code for each test case
        passed = True
        for test_in, test_out in zip(test_ins, test_outs):
            exec_res = run_code(code, test_in, 'python', timeout=30)
            if exec_res is None or exec_res.strip() != test_out.strip():
                passed = False
                break
    return passed

def evaluate_dataset(generator, dataset_path, dataset_name):
    data = load_dataset(dataset_path)
    total = len(data)
    passed = 0
    for item in data:
        question = item['question']
        test_code = item.get('test_code', '')
        test_ins = item.get('test_ins', [])
        test_outs = item.get('test_outs', [])
        code = generate_code(generator, question)
        if evaluate_code(code, test_code, test_ins, test_outs):
            passed += 1
    accuracy = passed / total if total > 0 else 0
    print(f"{dataset_name} Accuracy: {accuracy:.4f} ({passed}/{total})")
    return accuracy

if __name__ == "__main__":
    # Assume model is running on port 35000, adjust as needed for trained models
    generator = QwenGen(port=35000, temperature=0, max_tokens=10240)

    benchmark_dir = "/data3/zlh/king/code_tailored/eval/benchmark"
    datasets = {
        "MBPP": os.path.join(benchmark_dir, "mbpp.jsonl"),
        "HumanEval": os.path.join(benchmark_dir, "codeforces.jsonl"),
        "LiveCodeBench_V6": os.path.join(benchmark_dir, "livecodebench_v6.jsonl"),
        "CodeContests": os.path.join(benchmark_dir, "code_contests.jsonl"),
    }

    for name, path in datasets.items():
        evaluate_dataset(generator, path, name)