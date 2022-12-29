
from pal.utils import read_data,write_data

def combine_mathqa_prompted_train():

    """Combine MathQA train and prompted train sets."""

    part_1 = read_data('results/MathQA_trainset_res_bak.json')
    part_2 = read_data('results/MathQA_trainset_res_part_2.json')

    combined = part_1 + part_2
    write_data(file='datasets/MathQA/MathQA_trainset_prompted_combined.json', data=combined)

if __name__ == '__main__':
    combine_mathqa_prompted_train()