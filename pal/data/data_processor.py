
from pal.utils import read_data
import os
from datasets import Dataset, load_from_disk
from typing import Dict, List
from transformers import PreTrainedTokenizer
import torch
import logging

logger = logging.getLogger(__name__)

def read_from_dataset(dataset_file_path: str, split:str):
    """
    Read the dataset from the dataset file path
    :param dataset_file_path:
    :param split: "train", "validation", "test"
    :return:
    """
    hf_data_cached_file_name = dataset_file_path.replace(".json","_cached")
    if os.path.exists(hf_data_cached_file_name):
        logger.info("Loading cached file")
        hf_data = load_from_disk(hf_data_cached_file_name)
    else:
        logger.info(f"reading from: {dataset_file_path}")
        data = read_data(dataset_file_path)
        logger.info(f"length of data: {len(data)} for split: {split}")
        new_data = []
        for obj in data:
            if split == "train" and obj["score"] == 0:
                ## we only use those correct one as training data.
                continue
            if split == "train":
                assert str(obj['question']) != ""
                assert str(obj['extracted_answer']) != ""
                assert str(obj['raw_answer']) != ""
                assert str(obj['chains']) != ""
                assert str(obj["code"][0]) != ""
                assert str(obj["generation"][-1][0]) != ""

                new_obj = {'question': obj['question'],
                             'answer': str(obj['extracted_answer']),
                             "raw_answer": obj['raw_answer'],
                             "chains": obj['chains'],
                             "formated_code": obj["code"][0],
                             "generated_code_string": obj["generation"][-1][0]}

            else:
                new_obj = {'question': obj['question'],
                           'answer': obj['extracted_answer'],
                           "raw_answer": obj['raw_answer'],
                           "chains": obj['chains'],
                           "generated_code_string": ""}
            if "top_k_idxs" in obj:
                new_obj["top_k_idxs"] = obj["top_k_idxs"]
                assert len(obj["top_k_idxs"]) == 8
            new_data.append(new_obj)
        hf_data = Dataset.from_list(new_data)
        hf_data.save_to_disk(hf_data_cached_file_name)
    return hf_data


def tokenize_data(example_dict: Dict, tokenizer: PreTrainedTokenizer):
    features = {"input_ids": [], "attention_mask": [], "metadata":[]}

    for question, code in zip(example_dict["question"], example_dict["generated_code_string"]):
        if code != "":
            res = tokenizer(question + " The resulting python solution: " +  code, truncation=True, max_length=512, return_attention_mask=True)
        else:
            res = tokenizer(question, truncation=True, max_length=512, return_attention_mask=True)
        features["input_ids"].append(res["input_ids"])
        features["attention_mask"].append(res["attention_mask"])

    for answer, question in zip(example_dict["answer"], example_dict["question"]):
        metadata = {"question": question, "answer": answer}
        features["metadata"].append(metadata)
    return features

def tokenize_data_with_prompt(example_dict: Dict, tokenizer: PreTrainedTokenizer, all_training_data: List[Dict]):
    features = {"input_ids": [], "attention_mask": [], "metadata":[]}

    for question, code, top_k_idxs in zip(example_dict["question"], example_dict["generated_code_string"], example_dict["top_k_idxs"]):
        all_input_seq = ""
        for idx in top_k_idxs:
            top_k_question = all_training_data[idx]["question"]
            top_k_code = all_training_data[idx]["generation"][-1][0]
            current_sample = top_k_question + "\n\n# python solution:\n\n" + top_k_code + "\n\n"
            all_input_seq = all_input_seq + current_sample
        if code != "":
            # for training
            res = tokenizer(all_input_seq + question + "\n\n# python solution:\n\n" +  code, truncation=True, max_length=2048, return_attention_mask=True)
        else:
            # for testing.
            res = tokenizer(all_input_seq + question, truncation=True, max_length=2048, return_attention_mask=True)
        features["input_ids"].append(res["input_ids"])
        features["attention_mask"].append(res["attention_mask"])

    for answer, question in zip(example_dict["answer"], example_dict["question"]):
        metadata = {"question": question, "answer": answer}
        features["metadata"].append(metadata)
    return features


from dataclasses import dataclass

@dataclass
class PaddedDataCollator:
    tokenizer: PreTrainedTokenizer
    def __call__(self, features):
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        max_input_length = max(len(x["input_ids"]) for x in features)
        for feature in features:
            # change to left padding
            input_ids = [self.tokenizer.eos_token_id] * (max_input_length - len(feature["input_ids"])) + feature["input_ids"]
            attention_mask = [0] * (max_input_length - len(feature["attention_mask"])) + feature["attention_mask"]
            labels = [-100] * (max_input_length - len(feature["input_ids"])) + feature["input_ids"]
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch["labels"] = torch.tensor(batch["labels"])
        return batch