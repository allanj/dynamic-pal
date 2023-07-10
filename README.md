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
pip3 install sentence-transformers # optional for sentence embedding
```
**NOTE**: you also need the `OPENAI KEY` if you want to obtain the annotations yourself. 

## Datasets

### (Optional) Similarity-based training data
If you need the sentence embeddings for the data, I have already processed them and available in Google Drive.
(In our experiments, we also tried the sentence transformer, which is also as good as the OpenAI embedings. )

| Dataset | Link                                                                                             |
|---------|--------------------------------------------------------------------------------------------------|
 | GSM8K   | [Download](https://drive.google.com/drive/folders/1srjGLa5Ers_9eTBO3X97Lg5mbfp2sw76?usp=sharing) |
|MathQA|[Download](https://drive.google.com/drive/folders/19otXTswUMlvsY2dvyiMl404j02zeT2oC?usp=sharing) |
| SVAMP   | [Download](https://drive.google.com/drive/folders/1224AT6hAzSw2cSG8ZQ4aPHvhCb2dVkcv?usp=sharing) |

Alternatively, you can obtain the embeddings using this script:
```bash
python3 -m preprocess.sent_embedding --dataset_folder=gsm8k
```

### Program-distilled training data

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
