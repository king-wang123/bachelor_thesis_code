import sys

sys.path.insert(0, "/data/250010072/zlh/king/code_tailored_dataset/train/rl")

from reward_utils import output_equal, run_program


CASES = [
    ("python", "a,b=map(int,input().split())\nprint(a+b)\n"),
    ("c", "#include <stdio.h>\nint main(){int a,b; scanf(\"%d%d\",&a,&b); printf(\"%d\\n\",a+b); return 0;}\n"),
    ("cpp", "#include <bits/stdc++.h>\nusing namespace std; int main(){int a,b; cin>>a>>b; cout<<a+b<<'\\n';}\n"),
    ("java", "import java.util.*; public class Main { public static void main(String[] args){ Scanner sc=new Scanner(System.in); int a=sc.nextInt(), b=sc.nextInt(); System.out.println(a+b); }}\n"),
    ("go", "package main\nimport \"fmt\"\nfunc main(){var a,b int; fmt.Scan(&a,&b); fmt.Println(a+b)}\n"),
]


if __name__ == "__main__":
    ok = True
    for lang, code in CASES:
        out = run_program(code, "2 5\n", lang, timeout=10)
        passed = output_equal(out, "7")
        ok = ok and passed
        print(lang, "PASS" if passed else "FAIL", repr(out))
    raise SystemExit(0 if ok else 1)

