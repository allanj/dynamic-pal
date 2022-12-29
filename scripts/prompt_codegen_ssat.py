

from pal.prompt import ssat_parsing_prompt
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from typing import Dict
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import torch
from pal.utils import read_data, write_data
from pal.core.runtime import GenericRuntime
import logging
import argparse
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def parse_arguments(parser:argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lm_model_name', type=str, default="codegen-6B-mono")

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        logger.info(f"{k} = {args.__dict__[k]}")
    return args

args = parse_arguments(argparse.ArgumentParser())

def tokenize(example_dict: Dict, tokenizer: PreTrainedTokenizerFast):
    features = {"input_ids": [], "attention_mask": []}
    for question in example_dict["question"]:
        question = question.replace("\n", " ")
        prompt = ssat_parsing_prompt.MATH_PROMPT.format(question=question)
        dict = tokenizer(prompt, truncation=True, max_length=2048, return_attention_mask=True, return_tensors="pt")
        features["input_ids"].append(dict["input_ids"][0])
        features["attention_mask"].append(dict["attention_mask"][0])
    return features


from dataclasses import dataclass

@dataclass
class PaddedDataCollator:
    tokenizer: PreTrainedTokenizerFast
    def __call__(self, features):
        batch = {"input_ids": [], "attention_mask": [], "input_length": []}
        max_input_length = max(len(x["input_ids"]) for x in features)
        for feature in features:
            # change to left padding
            input_ids = [self.tokenizer.eos_token_id] * (max_input_length - len(feature["input_ids"])) + feature["input_ids"]
            attention_mask = [0] * (max_input_length - len(feature["attention_mask"])) + feature["attention_mask"]
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["input_length"].append(max_input_length)
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch["input_length"] = torch.tensor(batch["input_length"])
        return batch



fp16=True #TODO: remove this

tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not accelerator.is_local_main_process)

dataset_file = "datasets/ssat/ssat_test_questions_no_image.json"
sub_model_name = args.lm_model_name
lm_model_name = f"Salesforce/{sub_model_name}"
batch_size = args.batch_size

tokenizer = AutoTokenizer.from_pretrained(lm_model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(lm_model_name, pad_token_id=tokenizer.eos_token_id, low_cpu_mem_usage=True)
collator = PaddedDataCollator(tokenizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) ## dummy optimizer for deepspeed purpose

data =  read_data(dataset_file)

hf_data = Dataset.from_list(data)

processed_data = hf_data.map(
    function=tokenize,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
    load_from_cache_file=False,
    num_proc=1,
    remove_columns=hf_data.column_names
)

loader = DataLoader(processed_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)

model, loader, optimizer = accelerator.prepare(model, loader, optimizer)
runtime = GenericRuntime()



model.eval()
predictions = []
# labels = []
correct = 0
res = []
total = 0

all_code = []
for index, feature in tqdm(enumerate(loader), desc="--validation", total=len(loader)):
    with torch.cuda.amp.autocast(enabled=fp16), torch.no_grad():
        module = accelerator.unwrap_model(model)
        generated_ids = module.generate(input_ids=feature["input_ids"],
                                        attention_mask=feature["attention_mask"],
                                        num_beams=1,
                                        max_new_tokens=400,
                                        return_dict_in_generate=True).sequences
        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.eos_token_id)
        generated_ids = accelerator.gather_for_metrics(generated_ids)
        input_length = accelerator.gather_for_metrics(feature["input_length"])
        all_ids = []
        generated_ids = generated_ids.tolist()
        for gen_ids, input_id_length in zip(generated_ids, input_length.tolist()):
            gen_ids = gen_ids[input_id_length:]
            all_ids.append(gen_ids)
        # generated_ids = generated_ids[0, len(feature["input_ids"][0]):]
        # prediction = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prediction_code_gen = tokenizer.batch_decode(all_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        all_code.extend(prediction_code_gen)

new_data = []
for idx, current_predicted_code in enumerate(all_code):
    code = current_predicted_code.split("\n")
    new_data.append({"question": data[idx]["question"], "extracted_answer": data[idx]["extracted_answer"],
                     "code": code, "predicted_code": current_predicted_code})
write_data(data = new_data, file=f"results/ssat_test_parsing_prompt_{sub_model_name}.json")
logger.info(f"accuracy : {correct/total * 100:.4f}%")
