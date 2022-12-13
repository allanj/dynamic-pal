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
import numpy as np

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
parser.add_argument('--max_tokens', default=600, type=int)
parser.add_argument('--similarity_order', default='most_similar', type=str, choices=['most_similar', 'least_similar', 'random'])
args = parser.parse_args()

DATA_PATH = f'datasets/gsm8k_test_sent_split.json'
OUTPUT_PATH = f'eval_results/gsm8k_test_sent_split_results.jsonl'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

examples = read_data(DATA_PATH)
training_data = read_data('datasets/gsm8k_train_eval_result.json')
train_sent_embs = np.load('datasets/gsm8k_train_sent_emb.npy')
test_sent_embs = np.load('datasets/gsm8k_test_sent_emb.npy')
os.makedirs("results", exist_ok=True)
output_name = f"results/gsm8k_eval_{args.similarity_order}_result.json"

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

def find_top_k_prompt(test_sent_embs: np.array, test_question_idx: int, train_sent_embs: np.array, training_data, k= 8,
                      similarity_order='most_similar'):
    """
    find the top k training question idx in the training data based on the cosine similarity
    :param test_sent_embs:
    :param test_question_idx:
    :param train_sent_embs:
    :param k:
    :return:
    """
    if similarity_order == 'random':
        candidate_idx = np.random.choice(len(training_data), len(training_data), replace=False).tolist()
        top_k_idx = []
        cursor = 0
        while len(top_k_idx) < k:
            ## only accept the data that have score == 1
            if training_data[candidate_idx[cursor]]['score'] == 1:
                # print(sim[sorted_idx[cursor]])
                top_k_idx.append(candidate_idx[cursor])
            cursor += 1
    else:
        test_sent_emb = test_sent_embs[test_question_idx]
        sim = np.dot(train_sent_embs, test_sent_emb)
        sorted_idx = np.argsort(sim)
        if similarity_order == 'most_similar':
            sorted_idx = sorted_idx[::-1]
        else:
            pass
        top_k_idx = []
        cursor = 0
        while len(top_k_idx) < k:
            ## only accept the data that have score == 1
            if training_data[sorted_idx[cursor]]['score'] == 1:
                # print(sim[sorted_idx[cursor]])
                top_k_idx.append(sorted_idx[cursor])
            cursor += 1
    return top_k_idx

def construct_prompt_based_on_top_k_prompt(training_data, top_k_idx, test_question_idx, test_data):
    """
    construct the prompt based on the top k training question idx
    :param training_data:
    :param top_k_idx:
    :param test_question_idx:
    :return:
    """
    prompt = ""
    for idx in top_k_idx:
        prompt += "Q: " + training_data[idx]['question'] + "\n\n" + "# solution in Python:\n\n\n"
        prompt +=  training_data[idx]['generation'][-1][0] + "\n\n\n\n\n\n"

    prompt += "Q: " + test_data[test_question_idx]['question'] + "\n\n" + "# solution in Python:\n\n\n"
    return prompt



all_data = []
with open(OUTPUT_PATH, 'a' if args.append else 'w') as f:
    pbar = tqdm.tqdm(enumerate(examples[num_skip_exps:]), initial=num_skip_exps, total=len(examples))
    for idx, x in pbar:
        question = x['question']
        result = copy.copy(x)
        top_k_idx = find_top_k_prompt(test_sent_embs=test_sent_embs,
                                      test_question_idx = idx,
                                      train_sent_embs = train_sent_embs,
                                      training_data=training_data, k=8,
                                      similarity_order=args.similarity_order)
        current_prompt = construct_prompt_based_on_top_k_prompt(training_data=training_data, top_k_idx=top_k_idx, test_question_idx=idx, test_data=examples)
        temperature = args.temperature
        try:
            # debug info is commented
            # print(current_prompt)
            # exit(0)
            code, ans = itf.run(current_prompt, majority_at=args.majority_at,
                                temperature=temperature, top_p=args.top_p,
                                max_tokens=args.max_tokens)
            ans = float(ans)
            score = 1 if abs(ans - float(x['extracted_answer'])) < 1e-3 else 0
        except Exception as e:
            print(e)
            code = ''
            ans = ''
            score = 0
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
            write_data(data=all_data, file=output_name)
write_data(data=all_data, file=output_name)
print(f'Accuracy - {sum(scores) / len(scores)}')
