import tempfile
import os
import subprocess
import re
from run_code import run_code

def extract_function_name(code: str) -> str:
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
    return match.group(1) if match else "unknown_func"

def run_with_solution_module(sol_code: str, test_code: str, timeout=30) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        # solution.py
        with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as f:
            f.write(sol_code + "\n")
        # test_solution.py
        with open(os.path.join(tmpdir, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write(test_code + "\n")
        # run pytest
        try:
            result = subprocess.run(
                ["pytest", "-q", "--tb=no", "test_solution.py"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except:
            return 2
        return 0 if result.returncode == 0 else 1

def run_in_single_file(sol_code: str, test_code: str, timeout=30) -> int:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(sol_code.strip() + "\n\n" + test_code.strip() + "\n")
        temp_path = f.name

    try:
        result = subprocess.run(
            ["pytest", "-q", "--tb=no", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return 0 if result.returncode == 0 else 1
    except:
        return 2
    finally:
        os.unlink(temp_path)

def auto_detect_and_run(solution, test_code) -> int:
    # 策略1：是否包含 "from solution import"
    if "from solution import" in test_code:
        return run_with_solution_module(solution, test_code)

    # 策略2：是否直接调用函数名（启发式）
    func_name = extract_function_name(solution)
    if func_name != "unknown_func" and func_name in test_code:
        return run_in_single_file(solution, test_code)

    # 策略3：兜底用单文件
    return run_in_single_file(solution, test_code)


def kodcode_exec(code, test_code, test_ins, test_outs):
    if test_code != '':
        # 运行 pytest
        passed = auto_detect_and_run(code, test_code)
    else:
        passed = 0
        for test_in, test_out in zip(test_ins, test_outs):
            exec_res = run_code(code, test_in, 'python', timeout=30)
            if exec_res != test_out:
                passed = 1
            break
    
    if passed == 0:
        return True
    else:
        return False
    