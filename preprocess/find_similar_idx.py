from typing import Union, List
import numpy as np
from pal.utils import read_data, write_data


"""
Experiments to find similar questions in the training set for each train/test question.
So that we can imitate the prompting process in the trianing.
"""

def find_top_k_train_idx(test_sent_embs: np.array, test_question_idx: int, train_sent_embs: np.array, training_data, k= 8,
                      similarity_order='most_similar', except_idx:int = -1) -> Union[List[int], None]:
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
            if training_data[candidate_idx[cursor]]['score'] == 1 and candidate_idx[cursor] != except_idx:
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
        top_k_idx = []
        cursor = 0
        sorted_idx = sorted_idx.tolist()
        while len(top_k_idx) < k:
            ## only accept the data that have score == 1
            if training_data[sorted_idx[cursor]]['score'] == 1 and sorted_idx[cursor] != except_idx:
                # print(sim[sorted_idx[cursor]])
                top_k_idx.append(sorted_idx[cursor])
            cursor += 1
    else:
        ## normal prompts
        return None
    return top_k_idx

def find_similar_idx_for_train(train_data, train_sent_embs, similarity_order):
    """
    Find the top k similar training instances for each training sample.
    :param train_data:
    :param train_sent_embs:
    :return:
    """
    new_data = []
    for idx, obj in enumerate(train_data):
        top_k_idxs = find_top_k_train_idx(test_sent_embs=train_sent_embs,
                             test_question_idx=idx,
                             train_sent_embs=train_sent_embs,
                             training_data=train_data,
                             k=8,
                             similarity_order=similarity_order,
                             except_idx=idx)
        obj["top_k_idxs"] = top_k_idxs
        new_data.append(obj)
    return new_data

def find_similar_idx_for_test(test_sent_embs, test_data, train_data, train_sent_embs, similarity_order):
    """
    Find the top k similar training instances for each training sample.
    :param train_data:
    :param train_sent_embs:
    :return:
    """
    new_data = []
    for idx, obj in enumerate(test_data):
        top_k_idxs = find_top_k_train_idx(test_sent_embs=test_sent_embs,
                             test_question_idx=idx,
                             train_sent_embs=train_sent_embs,
                             training_data=train_data,
                             k=8,
                             similarity_order=similarity_order,
                             except_idx=-1)
        obj["top_k_idxs"] = top_k_idxs
        new_data.append(obj)
    return new_data


if __name__ == '__main__':
    dataset_folder= "gsm8k"
    training_data = read_data(f'datasets/{dataset_folder}/gsm8k_train_eval_result.json')
    test_data = read_data(f'datasets/{dataset_folder}/gsm8k_test_sent_split.json')
    train_sent_embs = np.load(f'datasets/{dataset_folder}/gsm8k_train_sent_emb.npy')
    test_sent_embs = np.load(f'datasets/{dataset_folder}/gsm8k_test_sent_emb.npy')

    new_train_data = find_similar_idx_for_train(
        train_data=training_data,
        train_sent_embs=train_sent_embs,
        similarity_order='most_similar'
    )
    new_test_data = find_similar_idx_for_test(
        test_sent_embs=test_sent_embs,
        test_data=test_data,
        train_data=training_data,
        train_sent_embs=train_sent_embs,
        similarity_order='most_similar')

    write_data(file=f'datasets/{dataset_folder}/gsm8k_train_with_similar_idx.json', data=new_train_data)
    write_data(file=f'datasets/{dataset_folder}/gsm8k_test_with_similar_idx.json', data=new_test_data)