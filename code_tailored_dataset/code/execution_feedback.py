"""
Execution feedback module: syntax check + code execution → structured feedback.

Two tools:
1. syntax_check: compile/parse the code to catch syntax errors before execution
2. execute_and_feedback: run the code against test cases and produce a structured result dict
"""

import ast
import subprocess
import tempfile
import os

from run_code import run_code


def syntax_check(code: str, lang: str = "python") -> dict:
    """
    Check code for syntax errors without executing it.

    Returns:
        dict with keys:
            - ok: bool
            - error_message: str (empty if ok)
    """
    lang = lang.lower().strip()

    if lang == "python":
        try:
            ast.parse(code)
            return {"ok": True, "error_message": ""}
        except SyntaxError as e:
            msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            if e.text:
                msg += f"\n  {e.text.rstrip()}"
            return {"ok": False, "error_message": msg}

    elif lang in ("c", "c++", "cpp"):
        return _compile_check(code, lang)

    elif lang == "java":
        return _compile_check(code, lang)

    elif lang == "go":
        return _compile_check(code, lang)

    else:
        return {"ok": True, "error_message": ""}


def _compile_check(code: str, lang: str) -> dict:
    """Try to compile without linking to check syntax."""
    ext_map = {"c": ".c", "c++": ".cpp", "cpp": ".cpp", "java": ".java", "go": ".go"}
    ext = ext_map.get(lang, ".txt")

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8") as f:
        f.write(code)
        src_path = f.name

    try:
        if lang in ("c", "c++", "cpp"):
            compiler = "g++" if lang in ("c++", "cpp") else "gcc"
            cmd = [compiler, "-fsyntax-only", src_path]
        elif lang == "java":
            cmd = ["javac", "-Xlint:none", src_path]
        elif lang == "go":
            # go vet for syntax check
            cmd = ["go", "vet", src_path]
        else:
            return {"ok": True, "error_message": ""}

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return {"ok": True, "error_message": ""}
        else:
            err = (result.stderr or result.stdout or "").strip()
            # Remove temp file paths from error messages to keep them clean
            err = err.replace(src_path, "<source>")
            if len(err) > 500:
                err = err[:500] + "... (truncated)"
            return {"ok": False, "error_message": err}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_message": "Compilation timed out"}
    except Exception as e:
        return {"ok": False, "error_message": str(e)}
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass
        # Clean up Java .class files
        if lang == "java":
            import glob
            for f in glob.glob(os.path.join(os.path.dirname(src_path), "*.class")):
                try:
                    os.remove(f)
                except OSError:
                    pass


def execute_and_feedback(code: str, test_ins: list, test_outs: list,
                         lang: str = "python", timeout: int = 30) -> dict:
    """
    Execute code against stdin/stdout test cases and produce structured feedback.

    Returns:
        dict with keys:
            - status: 'passed' | 'failed' | 'error' | 'timeout'
            - passed_count: int
            - total_count: int
            - error_message: str
            - first_failure: dict with input/expected/actual (only for 'failed')
    """
    total = len(test_ins)
    if total == 0:
        return {
            "status": "passed",
            "passed_count": 0,
            "total_count": 0,
            "error_message": "",
            "first_failure": {},
        }

    passed_count = 0
    first_failure = {}

    for i, (test_in, test_out) in enumerate(zip(test_ins, test_outs)):
        try:
            actual = run_code(code, test_in, lang, timeout=timeout)
        except Exception as e:
            return {
                "status": "error",
                "passed_count": passed_count,
                "total_count": total,
                "error_message": str(e),
                "first_failure": {},
            }

        if actual is None:
            # Runtime error or timeout — try to get error details
            err_detail = _get_runtime_error(code, test_in, lang, timeout)
            if not first_failure:
                first_failure = {
                    "input": test_in,
                    "expected": test_out,
                    "actual": "(runtime error)",
                }
            return {
                "status": "error",
                "passed_count": passed_count,
                "total_count": total,
                "error_message": err_detail,
                "first_failure": first_failure,
            }

        # Compare output (strip whitespace for robustness)
        if actual.strip() == test_out.strip():
            passed_count += 1
        else:
            if not first_failure:
                first_failure = {
                    "input": test_in,
                    "expected": test_out,
                    "actual": actual,
                }

    if passed_count == total:
        return {
            "status": "passed",
            "passed_count": passed_count,
            "total_count": total,
            "error_message": "",
            "first_failure": {},
        }
    else:
        return {
            "status": "failed",
            "passed_count": passed_count,
            "total_count": total,
            "error_message": "",
            "first_failure": first_failure,
        }


def _get_runtime_error(code: str, test_input: str, lang: str, timeout: int = 30) -> str:
    """Re-run the code to capture stderr for error diagnosis."""
    if lang != "python":
        return "Runtime error or timeout during execution"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        src_path = f.name

    try:
        result = subprocess.run(
            ["python", src_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        err = result.stderr.strip()
        if err:
            # Extract the last meaningful part of the traceback
            lines = err.split('\n')
            # Keep only the last few lines (error type + message)
            relevant = []
            for line in reversed(lines):
                relevant.insert(0, line)
                if line and not line.startswith(' ') and not line.startswith('Traceback'):
                    break
            err = '\n'.join(relevant[-5:])
            err = err.replace(src_path, "<source>")
            if len(err) > 500:
                err = err[:500] + "... (truncated)"
            return err
        return "Runtime error (no stderr captured)"
    except subprocess.TimeoutExpired:
        return "Execution timed out (>30s)"
    except Exception as e:
        return str(e)
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def execute_with_pytest(code: str, test_code: str, timeout: int = 30) -> dict:
    """
    Execute code with pytest test cases and produce structured feedback.
    For datasets that use pytest-style tests instead of stdin/stdout.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sol_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "test_solution.py")

        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(code + "\n")

        # Ensure test imports work
        if "from solution import" not in test_code:
            test_code = f"from solution import *\n{test_code}"

        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code + "\n")

        try:
            result = subprocess.run(
                ["pytest", "-q", "--tb=short", test_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "passed_count": 0,
                "total_count": 0,
                "error_message": "Pytest execution timed out",
                "first_failure": {},
            }
        except Exception as e:
            return {
                "status": "error",
                "passed_count": 0,
                "total_count": 0,
                "error_message": str(e),
                "first_failure": {},
            }

        if result.returncode == 0:
            return {
                "status": "passed",
                "passed_count": 0,
                "total_count": 0,
                "error_message": "",
                "first_failure": {},
            }
        else:
            err = result.stdout.strip() or result.stderr.strip()
            if len(err) > 500:
                err = err[:500] + "... (truncated)"
            return {
                "status": "failed",
                "passed_count": 0,
                "total_count": 0,
                "error_message": err,
                "first_failure": {},
            }
