def gen_code_system_prompt():
    return f'Based on the provided programming problem, deliver a clear, concise, and structured solution, and output the code in the target language as required. The format is:\n```lang\ncode\n```'


def basic_prompt(question, tgt_lang):
    return f"""**Programming Problem:**
{question.strip()}

**Target Language:** {tgt_lang}

---

Please generate a complete, well-structured response following the system template format.
"""


def code_reasoning_system_prompt():
    return """You are an expert software engineer. When solving any coding problem, you MUST strictly follow the three-phase structured reasoning pipeline below. Do NOT skip any phase, do NOT merge phases, and do NOT produce code before completing the prior phases.

---

## OUTPUT FORMAT (Mandatory)

Your entire response must be wrapped in the following three XML tags, in this exact order:

<decomposition>
{Problem Analysis and Breakdown}
</decomposition>
<pseudocode>
{Pseudocode Representation}
</pseudocode>
<code>
{Final Executable Code}
</code>

---

## PHASE INSTRUCTIONS

### Phase 1 — <decomposition>
Before writing any code or pseudocode, perform a structured breakdown of the problem:
- **Restate the problem** in your own words to confirm understanding
- **Identify inputs and outputs**: data types, formats, edge cases, constraints
- **Outline the solution strategy**: describe your high-level approach without any code
- **Estimate complexity**: provide time and space complexity estimates for your intended approach

### Phase 2 — <pseudocode>
Translate your decomposition into structured pseudocode that serves as the reasoning bridge to implementation:
- Use clear, language-agnostic pseudocode with proper indentation
- Cover all branches, edge cases, and the main algorithm flow
- Do NOT use actual programming language syntax — pseudocode only

### Phase 3 — <code>
Implement the final, directly executable code:
- The logic must faithfully mirror the pseudocode from Phase 2
- Use the programming language specified by the user
- The code must be complete, self-contained, and directly runnable
- Do NOT wrap code in Markdown fences — raw executable code only
"""


def reflection_system_prompt():
    return """You are an expert software engineer performing code review and debugging.

You will receive a programming problem, a previous attempt (with `<decomposition>`, `<pseudocode>`, and `<code>` phases), and execution feedback from running the code against test cases.

---

## GROUND TRUTH PRINCIPLE

The execution feedback is obtained by actually running your code. It is objective, deterministic, and **absolutely correct**.
- If the feedback says the code errors or fails a test, your code IS wrong. No exceptions.
- Do NOT question, doubt, or rationalize away the execution feedback.
- Do NOT speculate that the test cases might be wrong.
- Do NOT argue that your code "should" work. It was run and it did not.
- Your ONLY job when tests fail is to find and fix the bug.

---

## OUTPUT FORMAT

### When ALL tests passed:

Output a `<reflection>` that briefly confirms correctness. The LAST line INSIDE the `<reflection>` block (before `</reflection>`) must be exactly:
`All phases verified. Solution is complete.`
Do NOT place this line outside or after the `</reflection>` tag.

### When ANY test failed, errored, or timed out:

You MUST output:

1. A `<reflection>` block: briefly diagnose the bug (keep it concise, under 300 words). Identify which phase has the earliest error.

2. Then IMMEDIATELY output corrected phases as XML blocks OUTSIDE the reflection:
   - If the decomposition is wrong → output `<decomposition>`, `<pseudocode>`, `<code>`
   - If the pseudocode is wrong → output `<pseudocode>`, `<code>`
   - If only the code is wrong → output `<code>`

The corrected `<code>` block is MANDATORY when tests fail. You MUST always provide one.

---

## RULES

- Keep the reflection concise. Do not ramble or second-guess.
- The corrected code must be complete, self-contained, and directly executable.
- Do NOT wrap code in Markdown fences. Raw executable code only inside `<code>` tags.
- Do NOT put corrected `<code>` blocks inside `<reflection>` — they must come AFTER `</reflection>`.
"""


def reflection_prompt(question, prev_response, feedback, tgt_lang):
    return f"""**Programming Problem:**
{question.strip()}

**Target Language:** {tgt_lang}

---

**Previous Attempt:**

{prev_response.strip()}

---

**Execution Feedback (ground truth — the code was actually run):**

{feedback.strip()}

---

Analyze the execution feedback. If all tests passed, confirm with a brief reflection. If any test failed or errored, diagnose the bug in `<reflection>`, then output the corrected phase(s) with a fixed `<code>` block.
"""


def format_execution_feedback(syntax_ok, exec_result):
    """Format execution results into structured feedback text.

    Args:
        syntax_ok: bool — whether the code passed syntax check
        exec_result: dict with keys:
            - status: 'passed' | 'failed' | 'error' | 'timeout'
            - passed_count: int
            - total_count: int
            - error_message: str (for 'error' status)
            - first_failure: dict with input/expected/actual (for 'failed' status)
    """
    lines = []
    lines.append(f"[Syntax Check]: {'✅ Passed' if syntax_ok else '❌ Failed'}")

    if not syntax_ok:
        if exec_result.get('error_message'):
            lines.append(f"[Error]: {exec_result['error_message']}")
        return '\n'.join(lines)

    status = exec_result.get('status', 'error')

    if status == 'passed':
        total = exec_result.get('total_count', 0)
        lines.append(f"[Test Execution]: ✅ All {total} test(s) passed")
    elif status == 'failed':
        passed = exec_result.get('passed_count', 0)
        total = exec_result.get('total_count', 0)
        lines.append(f"[Test Execution]: ❌ {passed}/{total} test(s) passed")
        ff = exec_result.get('first_failure', {})
        if ff:
            lines.append(f"[First Failing Test]:")
            inp = ff.get('input', '')
            if len(inp) > 300:
                inp = inp[:300] + '... (truncated)'
            lines.append(f"  Input: {inp}")
            exp = ff.get('expected', '')
            if len(exp) > 200:
                exp = exp[:200] + '... (truncated)'
            lines.append(f"  Expected Output: {exp}")
            act = ff.get('actual', '')
            if len(act) > 200:
                act = act[:200] + '... (truncated)'
            lines.append(f"  Actual Output: {act}")
    elif status == 'timeout':
        lines.append(f"[Test Execution]: ⏰ Execution timed out (>30s)")
    elif status == 'error':
        lines.append(f"[Test Execution]: ❌ Runtime Error")
        err_msg = exec_result.get('error_message', '')
        if err_msg:
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + '... (truncated)'
            lines.append(f"[Error]: {err_msg}")

    return '\n'.join(lines)


# ==================== Legacy prompts (kept for gen_from_wrong / multi_task) ====================

def wrong_reasoning_system_prompt():
    return """
You are an expert software engineer creating high-quality supervised fine-tuning data for coding reflection.

You will receive a programming problem, a previous model attempt with three phases
(`<decomposition>`, `<pseudocode>`, and `<code>`), and a private reference implementation.
Your task is to continue the previous attempt by writing a rigorous `<reflection>` and, when needed,
regenerating the solution from the earliest incorrect phase onward.

The private reference implementation is only an internal calibration signal. Never mention, quote, or imply
that you saw a reference implementation, ground truth, official answer, hidden tests, or evaluator feedback.
The reflection must read like the model is self-checking its own reasoning against the problem statement,
input/output contract, edge cases, and algorithmic invariants.

---

## OUTPUT FORMAT (Mandatory)

Your response must start with exactly one `<reflection>` block.
Inside that reflection, evaluate the previous attempt in this order:
- **Decomposition check**: whether the problem understanding, constraints, edge cases, and strategy are correct.
- **Pseudocode check**: whether the pseudocode follows from the decomposition and handles the required cases.
- **Code check**: whether the code faithfully implements the pseudocode and the problem's input/output format.
- **Decision**: identify the earliest incorrect phase, or state that all phases are correct.

After the reflection:
- If the previous attempt is fully correct, end the reflection with exactly:
  `All phases verified. Solution is complete.`
- If the decomposition is the earliest incorrect phase, output corrected
  `<decomposition>`, `<pseudocode>`, `<code>`, and a final `<reflection>`.
- If the pseudocode is the earliest incorrect phase, output corrected
  `<pseudocode>`, `<code>`, and a final `<reflection>`.
- If only the code is incorrect, output corrected `<code>` and a final `<reflection>`.

The final reflection after any regenerated phase must again check the corrected phases and end with exactly:
`All phases verified. Solution is complete.`

---

## QUALITY RULES

- Keep the critique aligned with the existing decomposition, pseudocode, and code. Do not invent unrelated issues.
- Prefer precise failure analysis: wrong recurrence, missed edge case, incorrect parsing, output mismatch,
  off-by-one indexing, complexity issue, or mismatch between pseudocode and code.
- The corrected code must be complete, executable, and written in the requested language.
- Do not copy the private reference implementation verbatim. Use it to understand the intended behavior, then
  write an independent implementation with natural variable names and structure.
- Do not wrap code in Markdown fences. The `<code>` block must contain raw executable code only.
- Use only the XML tags required above; do not add extra top-level prose.
"""


def wrong_reasoning_prompt(question, decomposition, pseudocode, code, solution, tgt_lang):
    return f"""**Programming Problem:**
{question.strip()}

**Target Language:** {tgt_lang}

---

**Previous Attempt To Reflect On**

<decomposition>
{decomposition.strip()}
</decomposition>

<pseudocode>
{pseudocode.strip()}
</pseudocode>

<code>
{code.strip()}
</code>

---

**Private Reference Implementation For Calibration**

Use this only to infer the intended behavior and locate mistakes. Do not mention it, quote it, or copy it.

<reference_solution>
{str(solution).strip()}
</reference_solution>

---

Continue the previous attempt. Output only the required XML continuation beginning with `<reflection>`.
"""
