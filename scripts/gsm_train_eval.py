# Copyright 2022 PAL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import json
import argparse
import tqdm
import os

from pal import interface
from pal.prompt import math_prompts

def read_data(file: str):
    with open(file, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--append', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dataset', default='gsm', type=str)
parser.add_argument('--majority_at', default=None, type=int)
parser.add_argument('--temperature', default=0.0, type=float)
parser.add_argument('--top_p', default=1.0, type=float)
parser.add_argument('--max_tokens', default=256, type=int)
args = parser.parse_args()

DATA_PATH = f'datasets/gsm8k_train_sent_split.json'
OUTPUT_PATH = f'eval_results/gsm8k_train_sent_split_results.jsonl'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

examples = read_data(DATA_PATH)

itf = interface.ProgramInterface(
    stop='\n\n\n',
    get_answer_expr='solution()',
    verbose=args.verbose
)

if args.append:
    lines = open(OUTPUT_PATH).readlines()
    num_skip_exps = len(lines)
    scores = [x['score'] for x in map(json.loads, lines)]
else:
    num_skip_exps = 0
    scores = []

def write_data(file: str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


all_data = []
with open(OUTPUT_PATH, 'a' if args.append else 'w') as f:
    pbar = tqdm.tqdm(examples[num_skip_exps:], initial=num_skip_exps, total=len(examples))
    for x in pbar:
        question = x['question']
        result = copy.copy(x)
        
        solved = False
        temperature = args.temperature
        run_count = 0
        while not solved:
            try:
                code, ans = itf.run(math_prompts.MATH_PROMPT.format(
                    question=question), majority_at=args.majority_at, 
                    temperature=temperature, top_p=args.top_p,
                    max_tokens=args.max_tokens)
                ans = float(ans)
                score = 1 if abs(ans - float(x['extracted_answer'])) < 1e-3 else 0
                if score == 1:
                    solved = True
                    break
            except Exception as e:
                print(e)
                code = ''
                ans = ''
                score = 0
            temperature = 0.5
            run_count += 1
            if run_count == 5:
                break
        scores.append(score)
        
        result['prediction'] = ans
        result['score'] = score
        result['code'] = code
        result['generation'] = itf.history
        all_data.append(result)
        # f.write(json.dumps(result) + '\n')
        
        itf.clear_history()
        f.flush()
        if len(all_data) % 20 == 0:
            write_data(data=all_data, file="gsm8k_train_eval_result.json")
write_data(data=all_data, file="gsm8k_train_eval_result.json")
print(f'Accuracy - {sum(scores) / len(scores)}')
