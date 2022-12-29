from pal.utils import read_data
from collections import Counter

def from_category_to_acc(res_file:str):
    mathqa_res = read_data(res_file)
    corr_counter = Counter()
    total_counter = Counter()
    for obj in mathqa_res:
        if obj["score"] == 1:
            corr_counter[obj["category"]] += 1
        total_counter[obj["category"]] += 1

    for category in corr_counter:
        print(category, corr_counter[category] / total_counter[category] * 100)


if __name__ == '__main__':
    from_category_to_acc('results/MathQA_test_no_similarity_result.json')