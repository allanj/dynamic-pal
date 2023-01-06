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
from pal.prompt import gsm8k_expanded_prompts
from pal.utils import read_data, write_data

parser = argparse.ArgumentParser()
parser.add_argument('--append', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dataset', default='gsm8k', type=str)
parser.add_argument('--majority_at', default=None, type=int)
parser.add_argument('--temperature', default=0.0, type=float)
parser.add_argument('--top_p', default=1.0, type=float)
parser.add_argument('--max_tokens', default=600, type=int)
args = parser.parse_args()

# expand_times = None
# DATA_PATH = f'datasets/gsm8k/gsm8k_train_sent_split.json'
# output_file_name = f"datasets/gsm8k/gsm8k_train_eval_result.json"
# prompt_to_use = math_prompts

DATA_PATH = f'datasets/gsm8k/gsm8k_train_eval_result_with_id.json'
expand_times = 1
output_file_name = f"datasets/gsm8k/gsm8k_train_eval_result_expanded_{expand_times}.json"
prompt_to_use = gsm8k_expanded_prompts

examples = read_data(DATA_PATH)

itf = interface.ProgramInterface(
    stop='\n\n\n',
    get_answer_expr='solution()',
    verbose=args.verbose
)



all_data = []
scores = []
pbar = tqdm.tqdm(examples, initial=0, total=len(examples))
for x in pbar:
    question = x['question']
    result = copy.copy(x)

    if "score" in x and x["score"] == 1:
        all_data.append(result)
        ## only evaluate the ones that are not correct
        if len(all_data) % 20 == 0:
            write_data(data=all_data, file=output_file_name)
        continue

    solved = False
    temperature = args.temperature
    run_count = 0
    while not solved:
        try:
            code, ans = itf.run(prompt_to_use.MATH_PROMPT.format(
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
    if expand_times is not None:
        result[f'expand_{expand_times}'] = score
    result['code'] = code
    result['generation'] = itf.history
    all_data.append(result)
    # f.write(json.dumps(result) + '\n')

    itf.clear_history()
    if len(all_data) % 20 == 0:
        write_data(data=all_data, file=output_file_name)
write_data(data=all_data, file=output_file_name)
print(f'Accuracy - {sum(scores) / len(scores)}')
