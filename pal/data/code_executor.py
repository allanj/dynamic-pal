


from pal.core.runtime import GenericRuntime
from pal.core.interface import timeout
from typing import Optional, List
from collections import Counter

def process_generation_to_code(gens: str):
    return [g.split('\n') for g in gens]

def run_code(runtime, code_gen: str, answer_expr, time_out:float = 10):
    code_snippets = process_generation_to_code(code_gen)
    results = []
    for code in code_snippets:
        with timeout(time_out):
            try:
                exec_result = execute(runtime, code, answer_expr)
            except Exception as e:
                print(e)
                continue
            results.append(exec_result)
    counter = Counter(results)
    return code_snippets, counter.most_common(1)[0][0]

def execute(runtime: GenericRuntime, code: Optional[List[str]] = None, answer_expr: str = "solution()"):
    code = code if code else code
    runtime.exec_code('\n'.join(code))
    return runtime.eval_code(answer_expr)


