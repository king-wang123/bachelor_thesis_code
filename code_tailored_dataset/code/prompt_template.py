def gen_code_system_prompt():
    return f'Based on the provided programming problem, deliver a clear, concise, and structured solution, and output the code in the target language as required. The format is:\n```lang\ncode\n```'

def basic_prompt(question, tgt_lang):
    return f"""**Programming Problem:**
{question.strip()}

**Target Language:** {tgt_lang}

---

Please generate a complete response following the system template format:
"""

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
