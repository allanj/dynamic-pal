
from pal.utils import read_data
from tqdm import tqdm
import openai
import time
import numpy as np

"""
Prompting the sentence embeddings
"""

def get_embedding(text, model="text-embedding-ada-002"):
    return np.array(openai.Embedding.create(input=[text], model=model)['data'][0]['embedding'])

def prompt_emb(input_file, output_file, dataset_name, model_name= "text-embedding-ada-002"):
    data = read_data(input_file)
    all_reps = []
    for i in tqdm(range(len(data)), desc="Processing", total=len(data)):
        obj = data[i]
        if dataset_name == "gsm8k":
            question = obj['question']
        elif dataset_name == "svamp":
            question = obj['question'] if 'question' in obj else obj['sQuestion'].strip() ## for svamp
        elif dataset_name == "MathQA":
            question = obj["Problem"].strip()
        else:
            raise NotImplementedError
        success = False
        count = 1
        while not success:
            try:
                current_rep = get_embedding(question, model=model_name)
                success = True
            except:
                print(f"Not success, sleeping for {count} then retrying", flush=True)
                time.sleep(count)
                count *= 2
                continue
        all_reps.append(current_rep)
        time.sleep(1)
        if len(all_reps) % 50 == 0:
            res = np.stack(all_reps)
            np.save(output_file, res)
    res = np.stack(all_reps)
    np.save(output_file, res)

if __name__ == '__main__':
    openai.api_key = "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"
    dataset = 'MathQA'
    embedding_model_name = "text-embedding-ada-002"
    if dataset == 'gsm8k':
        prompt_emb(input_file="data/gsm8k/train_sent_split.json",
                   output_file="data/gsm8k/train_sent_emb.npy", dataset_name=dataset,
                   model_name= embedding_model_name)
        prompt_emb(input_file="data/gsm8k/test_sent_split.json",
                   output_file="data/gsm8k/test_sent_emb.npy", dataset_name=dataset,
                   model_name= embedding_model_name)
    elif dataset == "svamp":
        prompt_emb(input_file="datasets/svamp/trainset_nodup.json",
                   output_file="datasets/svamp/trainset.npy", dataset_name=dataset,
                   model_name= embedding_model_name)
        prompt_emb(input_file="datasets/svamp/testset_nodup.json",
                   output_file="datasets/svamp/testset.npy", dataset_name=dataset,
                   model_name= embedding_model_name)
    elif dataset == "MathQA":
        prompt_emb(input_file="datasets/MathQA/mathqa_train_nodup_our_filtered.json",
                   output_file="datasets/MathQA/mathqa_train_emb.npy", dataset_name=dataset,
                   model_name= embedding_model_name)
        prompt_emb(input_file="datasets/MathQA/mathqa_test_nodup_our_filtered.json",
                   output_file="datasets/MathQA/mathqa_test_emb.npy", dataset_name=dataset,
                   model_name= embedding_model_name)
