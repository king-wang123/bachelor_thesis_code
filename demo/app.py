from flask import Flask, render_template, request, Response
from qwen_gen import QwenGen
import re

app = Flask(__name__)

# Prompt template
def code_reasoning_system_prompt():
    return """
You are an expert software engineer. When solving any coding problem, you MUST strictly follow the four-phase structured reasoning pipeline below. Do NOT skip any phase, do NOT merge phases, and do NOT produce code before completing the prior phases.

---

## OUTPUT FORMAT (Mandatory)

Your entire response must be wrapped in the following four XML tags, in this exact order:

<decomposition>
{Problem Analysis and Breakdown}
</decomposition>
<pseudocode>
{Pseudocode Representation}
</pseudocode>
<code>
{Final Executable Code}
</code>
<reflection>
{Reflection and Verification}
</reflection>

---

## PHASE INSTRUCTIONS

### Phase 1 — <decomposition>
Before writing any code or pseudocode, perform a structured breakdown of the problem:
- Restate the problem in your own words to confirm understanding
- Identify constraints: data types, edge cases, input/output requirements, performance constraints
- Outline the solution strategy: describe your high-level approach without any code
- Estimate complexity: provide time and space complexity estimates for your intended approach

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
- Decomposition check: Was the problem correctly understood? Were all constraints and edge cases captured?
- Pseudocode check: Does the pseudocode correctly represent the intended algorithm?
- Code check: Does the code faithfully implement the pseudocode? Are there bugs, off-by-one errors, unhandled edge cases, or inefficiencies?

Decision rule (strictly enforced):
- If ALL three phases are correct → state "All phases verified. Solution is complete." and stop.
- If ANY phase contains an error → explicitly state which phase has the error and why, then regenerate from that phase onward (re-output the corrected phase and all subsequent phases inside their respective XML tags).
"""

def get_prompt(question):
    system_prompt = code_reasoning_system_prompt()
    user_prompt = f"**Programming Problem:**\n{question.strip()}\n\n**Target Language:** python\n\n---\n\nPlease generate a complete response following the system template format:"
    return f"{system_prompt}\n\n{user_prompt}"

generator = QwenGen(port=35000, temperature=0, max_tokens=4096)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    question = request.args.get('question')
    if not question:
        return Response("Error: No question provided", mimetype='text/plain')
    
    prompt = get_prompt(question)
    
    def generate_stream():
        try:
            for chunk in generator.stream_response(prompt):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return Response(generate_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)