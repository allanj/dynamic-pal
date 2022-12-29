

from pal.prompt import math_prompts
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
from transformers.data.data_collator import default_data_collator

def tokenize(example_dict: Dict, tokenizer: PreTrainedTokenizerFast):
    features = {"input_ids": [], "attention_mask": []}
    for question in example_dict["question"]:
        prompt = math_prompts.MATH_PROMPT.format(question=question)
        dict = tokenizer(prompt, truncation=True, max_length=512, return_attention_mask=True, return_tensors="pt")
        features["input_ids"].append(dict["input_ids"][0])
        features["attention_mask"].append(dict["attention_mask"][0])
    return features


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
fp16=True #TODO: remove this

tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not accelerator.is_local_main_process)


lm_model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(lm_model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(lm_model_name, pad_token_id=tokenizer.eos_token_id, low_cpu_mem_usage=True)


data = [
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
    }
]

hf_data = Dataset.from_list(data)

processed_data = hf_data.map(
    function=tokenize,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
    load_from_cache_file=False,
    num_proc=1,
    remove_columns=hf_data.column_names
)

loader = DataLoader(processed_data, batch_size=1, shuffle=False, num_workers=0, collate_fn=default_data_collator)

model, loader = accelerator.prepare(model, loader)

model.eval()
predictions = []
# labels = []
with torch.no_grad():
    for index, feature in tqdm(enumerate(loader), desc="--validation", total=len(loader)):
        with torch.cuda.amp.autocast(enabled=fp16):
            module = accelerator.unwrap_model(model)
            ## Note: need to check if the underlying model has revised the "prepare_inputs_for_generation" method
            generated_ids = module.generate(input_ids=feature["input_ids"],
                                            attention_mask=feature["attention_mask"],
                                            num_beams=1,
                                            max_new_tokens=300,
                                            return_dict_in_generate=True).sequences
            generated_ids = accelerator.gather_for_metrics(generated_ids)
            generated_ids = generated_ids[0, len(feature["input_ids"][0]):]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(prediction)

