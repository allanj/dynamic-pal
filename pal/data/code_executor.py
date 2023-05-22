


from pal.core.runtime import GenericRuntime
from pal.core.interface import timeout


def run_code(runtime, code_gen: str, answer_expr, time_out:float = 10):
    snippet = code_gen.split('\n')
    ## post process the code
    updated_code_snippet = ['import math', 'import sympy']
    for snippet_line in snippet:
        if snippet_line.startswith('def solution'):
            updated_code_snippet.append(snippet_line)
            continue
        if snippet_line.strip() == "":
            break
        updated_code_snippet.append(snippet_line)
    updated_code_gen = '\n'.join(updated_code_snippet)
    with timeout(time_out):
        try:
            runtime.exec_code(updated_code_gen)
            return snippet, runtime.eval_code(answer_expr)
        except Exception as e:
            print(e, flush=True)
            return snippet, None


if __name__ == '__main__':
    runtime = GenericRuntime()
    code = 'def solution():\n    \"\"\"There are 10000 gallons of water in a pool. Using a water pump, Anthony and his father fill a tank with half the amount of water in the pool. They use water from the tank to water their vegetable garden. If the tank is emptied at a rate of 500 gallons of water per day, how many gallons of water will be remaining in the tank after 6 days?\"\"\"\n    water_initial = 10000\n    water_per_day = 500\n    num_days = 6\n    water_added = water_per_day * num_days\n    water_left = water_initial - water_added\n    result = water_left\n    return result\n\n\n# 5 more gallons of water in a pool\nresult = solution()\nprint(result)\n\n# gallon of water in a pool with only 50% of it filled in the tank\nresult = solution(water_initial=20000)\nprint(result)\n\n# gallon of water in a pool with 50% of it filled in the tank\nresult = solution(water_initial=20000,'
    # print(code.split('\n'))
    print(run_code(runtime=runtime, code_gen=code, answer_expr="solution()"))

