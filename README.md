# Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning
Repo for the paper [Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning](https://arxiv.org/abs/2305.18170) appear in ACL 2023.

The main idea is to iteratively perform prompting to generate more training data for numerical reasoning. 
We simply use a similarity-based method to select the most similar questions from existing training set.

![procedure](/process.png)

Our codebase is adapted from the [PaL: Program-Aided Language Models](https://github.com/reasoning-machines/pal).

## Requirements
```bash
pip3 install accelerate # for distributed training
pip3 install openai # for chatgpt
```
**NOTE**: you also need the `OPENAI KEY` if you want to obtain the annotations yourself. 

## Dynamic Program Prompting

```bash
python3 -m scripts.eval_basedon_sim --dataset_folder=gsm8k
```

## Program Distillation
You also need to install the `accelerate` for distributed training
```bash
accelerate launch gsm8k_gen.py --train_file=datasets/gsm8k/gsm8k_train_eval_result.json \
                              --dev_file=datasets/gsm8k/test_sent_split.json 
```


## Citation
```
@inproceedings{jie2023leveraging,
  title={Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning},
  author={Jie, Zhanming and Lu, Wei},
  booktitle={Proceedings of ACL},
  year={2023}
}
```
