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
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--majority_at', default=None, type=int)
parser.add_argument('--temperature', default=0.0, type=float)
parser.add_argument('--top_p', default=1.0, type=float)
parser.add_argument('--max_tokens', default=600, type=int)
parser.add_argument('--dataset_folder', default="svamp", choices=["svamp", "MathQA"], type=str)
args = parser.parse_args()

dataset_folder = args.dataset_folder

DATA_PATH = f'datasets/{dataset_folder}/trainset_nodup.json'
if dataset_folder == "MathQA":
    DATA_PATH = f'datasets/{dataset_folder}/mathqa_train_nodup_our_filtered.json'

result_folder = "results"
os.makedirs(result_folder, exist_ok=True)

examples = read_data(DATA_PATH)

itf = interface.ProgramInterface(
    stop='\n\n\n',
    get_answer_expr='solution()',
    verbose=args.verbose
)


def write_data(file: str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


all_data = []
scores = []
for x in tqdm.tqdm(examples, total=len(examples), desc="prompting"):
    question = x['Problem'].strip()
    result = copy.copy(x)

    solved = False
    temperature = args.temperature
    run_count = 0
    while not solved:
        try:
            code, ans = itf.run(math_prompts.MATH_PROMPT.format(
                question=question.strip()), majority_at=args.majority_at,
                temperature=temperature, top_p=args.top_p,
                max_tokens=args.max_tokens)
            ans = float(ans)
            score = 1 if abs(ans - float(x['answer'])) < 1e-2 else 0
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

    itf.clear_history()
    if len(all_data) % 20 == 0:
        write_data(data=all_data, file=f"{result_folder}/{dataset_folder}_trainset_res_part_2.json")
write_data(data=all_data, file=f"{result_folder}/{dataset_folder}_trainset_res_part_2.json")
print(f'Accuracy - {sum(scores) / len(scores)}')