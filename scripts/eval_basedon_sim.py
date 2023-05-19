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
from typing import List, Dict, Tuple, Union

from pal import interface
from pal.prompt import math_prompts
import numpy as np

def read_data(file: str):
    with open(file, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data

def write_data(file: str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)



parser = argparse.ArgumentParser()
parser.add_argument('--append', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dataset', default='ssat', type=str)
parser.add_argument('--majority_at', default=1, type=int)
parser.add_argument('--temperature', default=0.0, type=float)
parser.add_argument('--top_p', default=1.0, type=float)
parser.add_argument('--max_tokens', default=600, type=int)
parser.add_argument('--similarity_order', default='most_similar', type=str, choices=['most_similar', 'least_similar', 'random', 'no_similarity'])
parser.add_argument('--emb_model', default='text-embedding-ada-002', type=str)
parser.add_argument('--top_k_prompt', default=12, type=int)


def find_top_k_prompt(test_sent_embs: np.array, test_question_idx: int, train_sent_embs: np.array, training_data, k= 8,
                      similarity_order='most_similar') -> Union[List[int], None]:
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
    elif similarity_order in ['most_similar', 'least_similar']:
        test_sent_emb = test_sent_embs[test_question_idx]
        sim = np.dot(train_sent_embs, test_sent_emb)
        sorted_idx = np.argsort(sim)
        if similarity_order == 'most_similar':
            sorted_idx = sorted_idx[::-1]
        else:
            pass
        if "score" in training_data[0]:
            top_k_idx = []
            cursor = 0
            have_score_not_1 = False
            while len(top_k_idx) < k:
                ## only accept the data that have score == 1
                if 'score' in training_data[sorted_idx[cursor]] and training_data[sorted_idx[cursor]]['score'] == 1:
                    # print(sim[sorted_idx[cursor]])
                    top_k_idx.append(sorted_idx[cursor])
                else:
                    have_score_not_1 = True
                cursor += 1
        else:
            top_k_idx = sorted_idx[:k]
    else:
        ## normal prompts
        return None
    return top_k_idx

def construct_prompt_based_on_top_k_prompt(training_data, top_k_idx, test_question_idx, test_data, test_question_key):
    """
    construct the prompt based on the top k training question idx
    :param training_data:
    :param top_k_idx:
    :param test_question_idx:
    :return:
    """
    prompt = ""
    for idx in top_k_idx:

        if test_question_key not in training_data[idx]:
            train_question = training_data[idx]['sQuestion'].strip()
        else:
            train_question = training_data[idx][test_question_key].strip()
        prompt += "Q: " + train_question + "\n\n" + "# solution in Python:\n\n\n"
        prompt +=  training_data[idx]['generation'][-1][0] + "\n\n\n\n\n\n"

    prompt += "Q: " + test_data[test_question_idx][test_question_key].strip() + "\n\n" + "# solution in Python:\n\n\n"
    return prompt

def construct_formal_prompt_based_on_top_k_prompt(training_data, top_k_idx, test_question_idx, test_data, test_question_key):
    """
    construct the prompt based on the top k training question idx
    :param training_data:
    :param top_k_idx:
    :param test_question_idx:
    :return:
    """
    prompt = ""
    for idx in top_k_idx:

        train_question = training_data[idx][test_question_key].strip()
        prompt += "Q: " + train_question + "\n" + "# Parsing results for the above question:\n"
        prompt +=  training_data[idx]['formal'].strip() + "\n# End of parsing\n\n\n"

    prompt += "Q: " + test_data[test_question_idx][test_question_key].strip() + "\n" + "# Parsing results for the above question:\n"
    return prompt


def inference(args,
              examples,
              training_data,
              train_sent_embs,
              test_sent_embs,
              output_name,
              test_question_key: str,
              answer_key: str):
    scores = []
    all_data = []
    for idx, x in tqdm.tqdm(enumerate(examples), total=len(examples)):
        result = copy.copy(x)
        top_k_idx = find_top_k_prompt(test_sent_embs=test_sent_embs,
                                      test_question_idx = idx,
                                      train_sent_embs = train_sent_embs,
                                      training_data=training_data, k=args.top_k_prompt,
                                      similarity_order=args.similarity_order)

        if top_k_idx is None:
            ## means using normal index.
            question = x[test_question_key]
            current_prompt = math_prompts.MATH_PROMPT.format(question=question)
        else:
            result["similar_questions"] = []
            for tkidx in top_k_idx:
                train_question = training_data[tkidx][test_question_key].strip()
                result["similar_questions"].append([train_question, training_data[tkidx]['formal'].strip()])
            if answer_key is None:
                current_prompt = construct_formal_prompt_based_on_top_k_prompt(training_data=training_data,
                                                                        top_k_idx=top_k_idx,
                                                                        test_question_idx=idx,
                                                                        test_data=examples, test_question_key=test_question_key)
            else:
                current_prompt = construct_prompt_based_on_top_k_prompt(training_data=training_data,
                                                                        top_k_idx=top_k_idx,
                                                                        test_question_idx=idx,
                                                                        test_data=examples, test_question_key=test_question_key)
        try:
            if answer_key is None:
                code = itf.run_formal(current_prompt, majority_at=args.majority_at,
                                    temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens)
                ans = ''
                score = 0
            else:
                code, ans = itf.run(current_prompt, majority_at=args.majority_at,
                                    temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens)
                ans = float(ans)
                score = 1 if abs(ans - float(x[answer_key])) < 1e-2 else 0
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
        if len(all_data) % 20 == 0:
            write_data(data=all_data, file=output_name)
    write_data(data=all_data, file=output_name)
    print(f'Accuracy - {sum(scores) / len(scores)}')


if __name__ == '__main__':
    args = parser.parse_args()

    training_data = None
    train_sent_embs = None
    test_sent_embs = None
    dataset_folder = args.dataset
    emb_suffix = args.emb_model
    if dataset_folder == "gsm8k":
        DATA_PATH = f'datasets/{dataset_folder}/gsm8k_test_sent_split.json'
        if args.similarity_order != 'no_similarity':
            training_data = read_data(f'datasets/{dataset_folder}/gsm8k_train_eval_result_expanded_1.json')
            train_sent_embs = np.load(f'datasets/{dataset_folder}/gsm8k_train_sent_emb_{emb_suffix}.npy')
            test_sent_embs = np.load(f'datasets/{dataset_folder}/gsm8k_test_sent_emb_{emb_suffix}.npy')
        test_question_key = "question"
        answer_key = 'extracted_answer'
    elif dataset_folder == "svamp":
        DATA_PATH = f'datasets/{dataset_folder}/testset_nodup.json'
        if args.similarity_order != 'no_similarity':
            training_data = read_data(f'datasets/{dataset_folder}/train_eval_result.json')
            train_sent_embs = np.load(f'datasets/{dataset_folder}/trainset_{emb_suffix}.npy')
            test_sent_embs = np.load(f'datasets/{dataset_folder}/testset_{emb_suffix}.npy')
        test_question_key = "question"
        answer_key = "answer"
    elif dataset_folder == "MathQA":
        DATA_PATH = f'datasets/{dataset_folder}/mathqa_test_nodup_our_filtered.json'
        if args.similarity_order != 'no_similarity':
            training_data = read_data(f'datasets/{dataset_folder}/MathQA_trainset_prompted_combined.json')
            train_sent_embs = np.load(f'datasets/{dataset_folder}/mathqa_train_emb_{emb_suffix}.npy')
            test_sent_embs = np.load(f'datasets/{dataset_folder}/mathqa_test_emb_{emb_suffix}.npy')
        test_question_key = "Problem"
        answer_key = "answer"
    elif dataset_folder == "ssat":
        DATA_PATH = f'datasets/{dataset_folder}/sat_prelabel.json'
        if args.similarity_order != 'no_similarity':
            training_data = read_data(f'datasets/{dataset_folder}/parsing_samples.json')
            train_sent_embs = np.load(f'datasets/{dataset_folder}/parsing_samples.npy')
            assert len(train_sent_embs) == len(training_data)
            test_sent_embs = np.load(f'datasets/{dataset_folder}/sat_prelabel.npy')
            assert len(test_sent_embs) == len(read_data(DATA_PATH))
        test_question_key = "question"
        answer_key = None
    else:
        raise ValueError("dataset not found")

    examples = read_data(DATA_PATH)
    os.makedirs("results", exist_ok=True)
    output_name = f"results/{dataset_folder}_test_{args.similarity_order}_{emb_suffix}_majority_{args.majority_at}_result.json"

    itf = interface.ProgramInterface(
        stop='\n\n\n',
        get_answer_expr='solution()',
        verbose=args.verbose
    )
    inference(args=args, examples=examples,
              training_data= training_data,
              train_sent_embs= train_sent_embs,
              test_sent_embs= test_sent_embs,
              output_name=output_name,
              test_question_key = test_question_key,
              answer_key=answer_key)