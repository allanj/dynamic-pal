# Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning
Repo for the paper [Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning](https://arxiv.org/abs/2305.18170) appear in Findings of ACL 2023.

The main idea is to iteratively perform prompting to generate more training data for numerical reasoning. 
We simply use a similarity-based method to select the most similar questions from existing training set.

![procedure](/process.png)

Our codebase is adapted from the [PaL: Program-Aided Language Models](https://github.com/reasoning-machines/pal).

## Requirements
```bash
pip3 install accelerate # for distributed training
pip3 install openai # optional for gpt-3.5-turbo, if you need to annotate the data again
pip3 install sentence-transformers # optional for sentence embedding
```
**NOTE**: you also need the `OPENAI KEY` if you want to obtain the annotations yourself. 

## Datasets

### (Optional) Similarity-based training data
If you need the sentence embeddings for the data, I have already processed them and available in Google Drive.
(In our experiments, we also tried the `sentence transformer`, which is also as good as the OpenAI embedings. )

| Dataset | Link                                                                                             |
|---------|--------------------------------------------------------------------------------------------------|
 | GSM8K   | [Download](https://drive.google.com/drive/folders/1srjGLa5Ers_9eTBO3X97Lg5mbfp2sw76?usp=sharing) |
|MathQA|[Download](https://drive.google.com/drive/folders/19otXTswUMlvsY2dvyiMl404j02zeT2oC?usp=sharing) |
| SVAMP   | [Download](https://drive.google.com/drive/folders/1224AT6hAzSw2cSG8ZQ4aPHvhCb2dVkcv?usp=sharing) |
After you downloaded the sentence representation, put them to the corresponding dataset folder under `datasets`.

Alternatively, you can obtain the embeddings using this script:
```bash
python3 -m preprocess.sent_embedding --openai_key=xxx --dataset_folder=gsm8k --embedding_model_name=text-embedding-ada-002
```

`openai_key` is required if we use the OpenAI embeddings. `dataset_folder` should be within `[gsm8k, svamp, MathQA]`, the last one is can be "`text-embedding-ada-002`", "`sentence-transformers`" or "`princeton-nlp`". The last one use SimCSE to obtain the sentence representations.

### Program-distilled Training data
All data has been placed under the `datasets` folder.

## Dynamic Program Prompting

```bash
python3 -m scripts.eval_basedon_sim --dataset_folder=gsm8k
```

You can specify other arguments such as `similarity_order` or `top_k_prompt` (number of exemplars as prompt).

## Program Distillation
You also need to install the `accelerate` for distributed training
```bash
accelerate launch gsm8k_gen.py --train_file=datasets/gsm8k/gsm8k_train_eval_result.json \
                              --dev_file=datasets/gsm8k/test_sent_split.json --model_folder=gsm8k_program_sft
```

The script will automatically download the `Salesforce/codegen-350M-mono` from HuggingFace.

## TODO
- [ ] Revise the code for prompting and Codex no longer available

## Citation
```
@inproceedings{jie2023leveraging,
  title={Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning},
  author={Jie, Zhanming and Lu, Wei},
  booktitle={Proceedings of ACL},
  year={2023}
}
```
