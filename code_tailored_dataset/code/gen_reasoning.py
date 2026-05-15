"""
Code Tailored — Iterative Reasoning Pipeline with Execution Feedback

Flow for each problem:
  1. Model generates: decomposition → pseudocode → code
  2. We extract code → syntax check → execute against test cases
  3. Format execution feedback → feed back to model for reflection (+fix)
  4. Repeat up to max_iterations (default 3) or until all tests pass

The model never sees "tool calling" — it only sees structured execution feedback
as a user message, and responds with reflection + optional corrected phases.

Output schema per item:
  - id, question, test_in, test_out, test_code, solution
  - response: final assembled full response string
  - trajectory: list of dicts recording every iteration step
  - final_passed: bool — whether the final code passes all tests
  - iterations_used: int — how many iterations were used
"""

import random
import argparse
import json
import os
import sys

from qwen_gen import QwenGen
from tools import get_jsonline, save_jsonline, extract_field, parallelize
from prompt_template import (
    code_reasoning_system_prompt,
    reflection_system_prompt,
    basic_prompt,
    reflection_prompt,
    format_execution_feedback,
)
from execution_feedback import syntax_check, execute_and_feedback


# ========================== Args ==========================

parser = argparse.ArgumentParser()
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--ips', nargs='+', default=['127.0.0.1'])
parser.add_argument('--ports', nargs='+', type=int, default=[35000, 35001])
parser.add_argument('--max_tokens', type=int, default=32768)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--max_iterations', type=int, default=3, help='Max feedback-reflection iterations')
parser.add_argument('--data_file', type=str,
                    default='/youtu-tuling-csp-llm/kodewang/bachelor_thesis_code/code_tailored_dataset/data/kodcode.jsonl')
parser.add_argument('--save_file', type=str,
                    default='/youtu-tuling-csp-llm/kodewang/bachelor_thesis_code/code_tailored_dataset/data/kodcode_reasoning.jsonl')
parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process (0=all)')
args = parser.parse_args()


# ========================== Models ==========================

gen_system_prompt = code_reasoning_system_prompt()
ref_system_prompt = reflection_system_prompt()

gen_models = []
ref_models = []
for ip in args.ips:
    for port in args.ports:
        gen_models.append(QwenGen(
            ip=ip, port=port,
            temperature=args.temperature,
            system_prompt=gen_system_prompt,
            max_tokens=args.max_tokens,
        ))
        ref_models.append(QwenGen(
            ip=ip, port=port,
            temperature=args.temperature,
            system_prompt=ref_system_prompt,
            max_tokens=args.max_tokens,
        ))


# ========================== Helpers ==========================

def clean_code(code: str) -> str:
    """Remove markdown fences if present."""
    if code is None:
        return ""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split('\n')
        # Remove first and last lines (fences)
        if lines[-1].strip() == '```':
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        code = '\n'.join(lines)
    return code.strip()


def assemble_prev_response(decomposition, pseudocode, code):
    """Assemble decomposition + pseudocode + code into XML string."""
    parts = []
    if decomposition:
        parts.append(f"<decomposition>\n{decomposition.strip()}\n</decomposition>")
    if pseudocode:
        parts.append(f"<pseudocode>\n{pseudocode.strip()}\n</pseudocode>")
    if code:
        parts.append(f"<code>\n{code.strip()}\n</code>")
    return '\n'.join(parts)


def run_tests(code: str, item: dict, lang: str) -> tuple:
    """
    Run syntax check + execution tests.

    Returns:
        (syntax_ok: bool, exec_result: dict, feedback_text: str)
    """
    # Step 1: Syntax check
    syn = syntax_check(code, lang)
    if not syn["ok"]:
        exec_result = {"status": "error", "error_message": syn["error_message"],
                       "passed_count": 0, "total_count": 0, "first_failure": {}}
        feedback_text = format_execution_feedback(False, exec_result)
        return False, exec_result, feedback_text

    # Step 2: Execute against test cases
    test_code = item.get("test_code", "")
    test_ins = item.get("test_in", [])
    test_outs = item.get("test_out", [])

    if test_code.strip():
        # pytest mode
        from execution_feedback import execute_with_pytest
        exec_result = execute_with_pytest(code, test_code)
    else:
        # stdin/stdout mode
        exec_result = execute_and_feedback(code, test_ins, test_outs, lang)

    feedback_text = format_execution_feedback(True, exec_result)
    return True, exec_result, feedback_text


# ========================== Core Pipeline ==========================

def inference(item):
    """
    Process one item through the iterative pipeline:
      Round 0: generate decomposition + pseudocode + code
      Round 1-N: execute → feedback → reflection (+ fix) — up to max_iterations
    """
    question = item['question']
    lang = args.lang

    trajectory = []

    # ---- Round 0: Initial generation (decomposition + pseudocode + code) ----
    prompt = basic_prompt(question, lang)
    gen_response = random.choice(gen_models).response(prompt)

    if not gen_response:
        item['response'] = ''
        item['trajectory'] = []
        item['final_passed'] = False
        item['iterations_used'] = 0
        return item

    # Extract the three phases
    decomposition = extract_field(gen_response, 'decomposition', last=False) or ''
    pseudocode = extract_field(gen_response, 'pseudocode', last=False) or ''
    code = clean_code(extract_field(gen_response, 'code', last=False) or '')

    # Record round 0
    trajectory.append({
        "round": 0,
        "role": "generation",
        "response": gen_response,
        "code": code,
    })

    # If extraction failed badly, save and return
    if not code.strip():
        item['response'] = gen_response
        item['trajectory'] = trajectory
        item['final_passed'] = False
        item['iterations_used'] = 0
        return item

    # ---- Iterative execution-feedback-reflection loop ----
    current_response_parts = gen_response  # Full text so far
    current_code = code
    current_decomposition = decomposition
    current_pseudocode = pseudocode
    final_passed = False
    iterations_used = 0

    for iteration in range(1, args.max_iterations + 1):
        iterations_used = iteration

        # Execute and get feedback
        syntax_ok, exec_result, feedback_text = run_tests(current_code, item, lang)

        trajectory.append({
            "round": iteration,
            "role": "execution",
            "syntax_ok": syntax_ok,
            "exec_result": exec_result,
            "feedback": feedback_text,
        })

        # Check if passed
        if exec_result.get("status") == "passed":
            final_passed = True
            # Still get reflection for the passing case
            prev_response = assemble_prev_response(current_decomposition, current_pseudocode, current_code)
            ref_prompt = reflection_prompt(question, prev_response, feedback_text, lang)
            ref_response = random.choice(ref_models).response(ref_prompt)

            trajectory.append({
                "round": iteration,
                "role": "reflection",
                "response": ref_response,
                "code": current_code,
            })
            break

        # Code failed — ask model for reflection + fix
        prev_response = assemble_prev_response(current_decomposition, current_pseudocode, current_code)
        ref_prompt = reflection_prompt(question, prev_response, feedback_text, lang)
        ref_response = random.choice(ref_models).response(ref_prompt)

        if not ref_response:
            trajectory.append({
                "round": iteration,
                "role": "reflection",
                "response": "",
                "code": current_code,
            })
            break

        # Extract updated phases from reflection response
        new_code = clean_code(extract_field(ref_response, 'code', last=True) or '')

        # Safety net: if tests failed but model didn't output a <code> block,
        # give it one more chance with an explicit nudge
        if not new_code:
            nudge = (
                "Your reflection above did not include a corrected <code> block, "
                "but the execution feedback clearly shows the code FAILED. "
                "The test results are ground truth — the code was actually executed and it did not pass. "
                "You MUST output a corrected <code> block now. "
                "Do not argue the tests are wrong. Find and fix the bug."
            )
            retry_response = random.choice(ref_models).response(ref_prompt + "\n\n" + nudge)
            if retry_response:
                retry_code = clean_code(extract_field(retry_response, 'code', last=True) or '')
                if retry_code:
                    # Use the retry response instead
                    ref_response = retry_response
                    new_code = retry_code

        new_decomposition = extract_field(ref_response, 'decomposition', last=True)
        new_pseudocode = extract_field(ref_response, 'pseudocode', last=True)

        # Update current state with any new phases
        if new_decomposition:
            current_decomposition = new_decomposition
        if new_pseudocode:
            current_pseudocode = new_pseudocode
        if new_code:
            current_code = new_code

        trajectory.append({
            "round": iteration,
            "role": "reflection",
            "response": ref_response,
            "code": current_code,
        })

    # If we exhausted iterations without passing, do a final check
    if not final_passed and iterations_used > 0:
        _, final_exec_result, _ = run_tests(current_code, item, lang)
        if final_exec_result.get("status") == "passed":
            final_passed = True

    # ---- Assemble final response ----
    # The final response is the complete trajectory assembled for SFT:
    #   Turn 1 (assistant): decomposition + pseudocode + code
    #   Turn 2 (user): feedback
    #   Turn 2 (assistant): reflection (+ possible fix phases)
    #   ... more rounds if needed ...
    final_response = _assemble_sft_response(trajectory)

    item['response'] = final_response
    item['trajectory'] = trajectory
    item['final_passed'] = final_passed
    item['iterations_used'] = iterations_used

    return item


def _assemble_sft_response(trajectory):
    """
    Assemble the trajectory into a structured SFT training string.

    Format:
      <decomposition>...</decomposition>
      <pseudocode>...</pseudocode>
      <code>...</code>
      [EXECUTION_FEEDBACK]
      ...feedback...
      [/EXECUTION_FEEDBACK]
      <reflection>...</reflection>
      <code>...</code>  (if fixed)
      ... (more rounds)
    """
    parts = []

    for step in trajectory:
        if step['role'] == 'generation':
            parts.append(step['response'])
        elif step['role'] == 'execution':
            parts.append(f"\n[EXECUTION_FEEDBACK]\n{step['feedback']}\n[/EXECUTION_FEEDBACK]\n")
        elif step['role'] == 'reflection':
            if step.get('response'):
                parts.append(step['response'])

    return '\n'.join(parts)


# ========================== Main ==========================

def main():
    data = get_jsonline(args.data_file)

    if args.num_samples > 0:
        data = data[:args.num_samples]

    num_workers = args.bsz * len(args.ports)
    print(f"Processing {len(data)} items with max {args.max_iterations} iterations each...")
    print(f"Models: {len(gen_models)} gen + {len(ref_models)} ref on ports {args.ports}")
    print(f"Concurrency: {num_workers} threads")

    # Streaming write — append each result as it completes, so no data loss on crash
    import json
    from threading import Lock
    write_lock = Lock()
    counters = {"passed": 0, "failed": 0, "total": 0}
    fout = open(args.save_file, 'w')

    def on_result(result):
        with write_lock:
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()
            counters["total"] += 1
            if result.get('final_passed'):
                counters["passed"] += 1
            else:
                counters["failed"] += 1

    results = parallelize(data, inference, max_workers=num_workers, on_result=on_result)
    fout.close()

    # Rewrite in original order (parallelize returns ordered results, streaming write is unordered)
    ordered_results = [r for r in results if r is not None]
    save_jsonline(ordered_results, args.save_file)

    # Print statistics
    total = len(ordered_results)
    passed = sum(1 for r in ordered_results if r.get('final_passed'))
    iter_counts = [r.get('iterations_used', 0) for r in ordered_results]
    avg_iter = sum(iter_counts) / max(len(iter_counts), 1)

    print(f"\n{'='*50}")
    print(f"Total: {total}")
    print(f"Final passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Avg iterations: {avg_iter:.2f}")

    # Breakdown by iteration count
    for i in range(args.max_iterations + 1):
        count = sum(1 for r in ordered_results if r.get('iterations_used') == i)
        p = sum(1 for r in ordered_results if r.get('iterations_used') == i and r.get('final_passed'))
        if count > 0:
            print(f"  Iteration {i}: {count} items, {p} passed ({p/count*100:.1f}%)")

    print(f"Saved to: {args.save_file}")


if __name__ == '__main__':
    main()
