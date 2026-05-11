import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple


ROOT = "/data/250010072/zlh/king/code_tailored_dataset"
CODE_DIR = os.path.join(ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

try:
    from kodecode_exec import auto_detect_and_run
except Exception:
    auto_detect_and_run = None


LANG_ALIASES = {
    "py": "python",
    "python3": "python",
    "python": "python",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "c/cpp": "cpp",
    "c++17": "cpp",
    "java": "java",
    "golang": "go",
    "go": "go",
}


def normalize_lang(lang: str) -> str:
    text = (lang or "").strip().lower()
    return LANG_ALIASES.get(text, text if text in {"python", "c", "cpp", "java", "go"} else "python")


def infer_lang(item: Dict[str, Any]) -> str:
    for key in ("target_lang", "lang", "language"):
        if item.get(key):
            return normalize_lang(str(item[key]))

    text = " ".join(str(item.get(k, "")) for k in ("id", "question", "src_question")).lower()
    patterns = [
        (r"target language:\s*(c\+\+|cpp|c\+\+17)", "cpp"),
        (r"target language:\s*java", "java"),
        (r"target language:\s*go", "go"),
        (r"target language:\s*c\b", "c"),
        (r"target language:\s*python", "python"),
        (r"-(java|go|cpp|c)(?:$|[-_])", None),
    ]
    for pattern, lang in patterns:
        match = re.search(pattern, text)
        if match:
            return lang or normalize_lang(match.group(1))
    return "python"


def parse_tests(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return ["" if x is None else str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return ["" if x is None else str(x) for x in parsed]
        except Exception:
            pass
        return [value]
    return [str(value)]


def extract_tag(text: str, tag: str, last: bool = False) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    matches = re.findall(pattern, text or "", flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return ""
    return matches[-1 if last else 0].strip()


def strip_code_fence(code: str) -> str:
    code = (code or "").strip()
    match = re.match(r"^```[a-zA-Z0-9_+\-#]*\s*(.*?)\s*```$", code, flags=re.DOTALL)
    return match.group(1).strip() if match else code


def extract_code(response: str) -> str:
    return strip_code_fence(extract_tag(response, "code", last=True))


def check_format(response: str) -> Tuple[bool, Dict[str, str]]:
    tags = ["decomposition", "pseudocode", "code", "reflection"]
    blocks = {tag: extract_tag(response, tag, last=(tag == "code")) for tag in tags}
    if any(not blocks[tag].strip() for tag in tags):
        return False, blocks

    pos = []
    for tag in tags:
        match = re.search(rf"<{tag}>.*?</{tag}>", response or "", flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return False, blocks
        pos.append(match.start())
    if pos != sorted(pos):
        return False, blocks
    if len(blocks["code"].strip()) < 20:
        return False, blocks
    return True, blocks


def normalize_output(text: Optional[str]) -> List[str]:
    if text is None:
        return ["<runtime_error>"]
    return str(text).replace("\r\n", "\n").strip().split()


def output_equal(pred: Optional[str], gold: str) -> bool:
    return normalize_output(pred) == normalize_output(gold)


def _run(cmd: List[str], stdin: str = "", timeout: int = 20, cwd: Optional[str] = None):
    try:
        res = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return res.stdout, res.stderr, res.returncode, False
    except subprocess.TimeoutExpired:
        return "", "timeout", -1, True
    except FileNotFoundError as exc:
        return "", str(exc), 127, False


def run_program(code: str, test_input: str, lang: str, timeout: int = 20) -> Optional[str]:
    lang = normalize_lang(lang)
    if not code.strip():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        if lang == "python":
            path = os.path.join(tmpdir, "main.py")
            open(path, "w", encoding="utf-8").write(code)
            out, _, rc, timed = _run([sys.executable, path], test_input, timeout=timeout)
            return None if timed or rc != 0 else out

        if lang in {"c", "cpp"}:
            ext = "c" if lang == "c" else "cpp"
            path = os.path.join(tmpdir, f"main.{ext}")
            exe = os.path.join(tmpdir, "main")
            open(path, "w", encoding="utf-8").write(code)
            if lang == "c":
                compile_cmd = ["gcc", path, "-O2", "-pipe", "-lm", "-o", exe]
            else:
                compile_cmd = ["g++", path, "-O2", "-pipe", "-std=c++17", "-o", exe]
            _, _, rc, timed = _run(compile_cmd, timeout=timeout)
            if timed or rc != 0:
                return None
            out, _, rc, timed = _run([exe], test_input, timeout=timeout)
            return None if timed or rc != 0 else out

        if lang == "java":
            class_name = "Main"
            src = code
            public_match = re.search(r"public\s+class\s+([A-Za-z_$][\w$]*)", src)
            if public_match and public_match.group(1) != class_name:
                src = re.sub(r"public\s+class\s+[A-Za-z_$][\w$]*", f"public class {class_name}", src, count=1)
            elif not public_match:
                src = re.sub(r"\bclass\s+([A-Za-z_$][\w$]*)", f"class {class_name}", src, count=1)
            path = os.path.join(tmpdir, f"{class_name}.java")
            open(path, "w", encoding="utf-8").write(src)
            _, _, rc, timed = _run(["javac", path], timeout=timeout)
            if timed or rc != 0:
                return None
            out, _, rc, timed = _run(["java", class_name], test_input, timeout=timeout, cwd=tmpdir)
            return None if timed or rc != 0 else out

        if lang == "go":
            path = os.path.join(tmpdir, "main.go")
            exe = os.path.join(tmpdir, f"main_{uuid.uuid4().hex[:8]}")
            open(path, "w", encoding="utf-8").write(code)
            _, _, rc, timed = _run(["go", "build", "-o", exe, path], timeout=timeout)
            if timed or rc != 0:
                return None
            out, _, rc, timed = _run([exe], test_input, timeout=timeout)
            return None if timed or rc != 0 else out

    return None


def reason_reward(blocks: Dict[str, str]) -> float:
    score = 0.0
    if len(blocks.get("decomposition", "").split()) >= 30:
        score += 0.25
    if len(blocks.get("pseudocode", "").splitlines()) >= 3:
        score += 0.25
    reflection = blocks.get("reflection", "").lower()
    checks = ["decomposition", "pseudocode", "code"]
    score += 0.25 * (sum(1 for x in checks if x in reflection) / len(checks))
    if "all phases verified" in reflection or "earliest incorrect phase" in reflection or "error" in reflection:
        score += 0.25
    return min(score, 1.0)


def evaluate_response(response: str, item: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    format_ok, blocks = check_format(response)
    code = extract_code(response)
    lang = infer_lang(item)

    if not format_ok:
        return {
            "score": 0.0,
            "acc": 0.0,
            "format": 0.0,
            "syntax": 0.0,
            "exec": 0.0,
            "reason": 0.0,
            "all_pass": False,
            "pass_count": 0,
            "test_count": 0,
            "lang": lang,
        }

    test_code = str(item.get("test_code") or "").strip()
    test_ins = parse_tests(item.get("test_in"))
    test_outs = parse_tests(item.get("test_out"))

    if test_code and lang == "python" and auto_detect_and_run is not None:
        passed = auto_detect_and_run(code, test_code) == 0
        exec_score = 1.0 if passed else 0.0
        syntax_score = 1.0 if passed else 0.0
        test_count = 1
        pass_count = int(passed)
    else:
        pairs = list(zip(test_ins, test_outs))
        pass_count = 0
        first_runnable = False
        for test_in, test_out in pairs:
            out = run_program(code, test_in, lang, timeout=timeout)
            if out is not None:
                first_runnable = True
            if output_equal(out, test_out):
                pass_count += 1
        test_count = len(pairs)
        exec_score = pass_count / test_count if test_count else 0.0
        syntax_score = 1.0 if first_runnable or pass_count > 0 else 0.0

    all_pass = test_count > 0 and pass_count == test_count
    r_score = reason_reward(blocks)
    score = 0.10 * syntax_score + 0.75 * exec_score + 0.15 * r_score
    if all_pass:
        score += 0.20

    return {
        "score": round(float(score), 6),
        "acc": 1.0 if all_pass else 0.0,
        "format": 1.0,
        "syntax": float(syntax_score),
        "exec": round(float(exec_score), 6),
        "reason": round(float(r_score), 6),
        "all_pass": bool(all_pass),
        "pass_count": int(pass_count),
        "test_count": int(test_count),
        "lang": lang,
    }
