import uuid
import os
import subprocess
import tempfile
import re
from typing import Optional

# ------------------- 语言 → 后缀 映射 -------------------
_LANG_EXT = {
    "c": "c",
    "c++": "cpp",
    "cpp": "cpp",
    "python": "py",
    "java": "java",
    "go": "go",
}

# ------------------- 统一的运行函数 -------------------
def _run_with_timeout(cmd, stdin: str = "", timeout: int = 30, cwd: Optional[str] = None):
    """统一执行 subprocess.run，返回 (stdout, returncode, timedout)"""
    try:
        result = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout.rstrip(), result.returncode, False
    except subprocess.TimeoutExpired:
        return "", -1, True


def run_code(
    code: str,
    test_input: str = "",
    lang: str = "python",
    timeout: int = 30,
) -> Optional[str]:
    """
    执行代码并返回标准输出。

    Returns
    -------
    str  : 代码成功执行且有非空 stdout
    ""   : 代码成功执行但 stdout 为空
    None : 编译/运行错误、超时等
    """
    lang = lang.lower().strip()
    assert lang in _LANG_EXT

    ext = _LANG_EXT[lang]
    # ---------- 1. 创建临时文件 ----------
    with tempfile.NamedTemporaryFile(mode="w", suffix=f".{ext}", delete=True) as src_file:
        src_path = src_file.name
        src_file.write(code)
        src_file.flush()

        # ---------- 2. 语言专用处理 ----------
        try:
            if lang in ("c", "c++", "cpp"):
                # 编译
                exe_path = src_path.rsplit(".", 1)[0]  # 去掉后缀
                if lang in ("c", "c++", "cpp"):
                    compiler = "g++" if lang in ("c++", "cpp") else "gcc"
                    compile_cmd = [compiler, src_path, "-o", exe_path, "-std=c++11" if compiler == "g++" else "-lm"]
                compile_out, compile_rc, _ = _run_with_timeout(compile_cmd, timeout=timeout)
                if compile_rc != 0:
                    return None

                # 运行
                run_out, run_rc, timedout = _run_with_timeout([exe_path], test_input, timeout=timeout)
                if timedout or run_rc != 0:
                    return None

            elif lang == "python":
                run_out, run_rc, timedout = _run_with_timeout(["python", src_path], test_input, timeout=timeout)
                if timedout or run_rc != 0:
                    return None

            elif lang == "java":
                # Java 需要类名与文件名一致，且类名不能以数字开头
                class_name = f"Main_{uuid.uuid4().hex[:8]}"
                java_code = re.sub(r"(public\s+class\s+)[\w$]+", rf"\1{class_name}", code, count=1)
                # 重新写入（覆盖原文件）
                with open(src_path, "w") as f:
                    f.write(java_code)

                # 编译
                compile_out, compile_rc, _ = _run_with_timeout(["javac", src_path], timeout=timeout)
                if compile_rc != 0:
                    return None

                # 运行（工作目录设为临时文件所在目录）
                work_dir = os.path.dirname(src_path)
                run_out, run_rc, timedout = _run_with_timeout(
                    ["java", class_name], test_input, timeout=timeout, cwd=work_dir
                )
                if timedout or run_rc != 0:
                    return None

            elif lang == "go":
                exe_path = src_path.rsplit(".", 1)[0]
                # 编译
                compile_out, compile_rc, _ = _run_with_timeout(
                    ["go", "build", "-o", exe_path, src_path], timeout=timeout
                )
                if compile_rc != 0:
                    return None

                # 运行
                run_out, run_rc, timedout = _run_with_timeout([exe_path], test_input, timeout=timeout)
                if timedout or run_rc != 0:
                    return None

            else:  # 不可能走到这里
                return None

            # ---------- 3. 返回结果 ----------
            return run_out if run_out else ""

        finally:
            # 清理编译产物（C/CPP/Go 可执行文件、Java .class 文件）
            import glob
            exe_candidates = [
                src_path.rsplit(".", 1)[0],                     # C/CPP/Go 可执行文件
                f"{os.path.dirname(src_path)}/*.class",         # Java .class 文件
            ]
            for pattern in exe_candidates:
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
    # with 结束，src_file 自动删除