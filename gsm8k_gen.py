import math

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from pal.utils import get_optimizers, write_data, read_data
import torch
import torch.nn as nn
import os
import logging
from transformers import set_seed
import logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import pad_across_processes
from accelerate import DistributedDataParallelKwargs
from pal.data.data_processor import read_from_dataset, tokenize_data, PaddedDataCollator, read_from_svamp, read_from_mathqa
from pal.core.runtime import GenericRuntime
from pal.data.code_executor import run_code
from functools import partial

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not accelerator.is_local_main_process)

def parse_arguments(parser:argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")
    parser.add_argument('--test_num', type=int, default=-1, help="The number of development data, -1 means all data")
    parser.add_argument('--num_proc', type=int, default=8, help="The number of development data, -1 means all data")
    parser.add_argument('--max_length', type=int, default=600, help="max generation length")

    parser.add_argument('--train_file', type=str, default="datasets/svamp/train_eval_results.json")
    parser.add_argument('--dev_file', type=str, default="datasets/svamp/testset_nodup.json")
    parser.add_argument('--test_file', type=str, default="none")
    parser.add_argument('--dft', type=int, default=0, help="direct finetune without CoT/PAL")
    parser.add_argument('--cot_ft', type=int, default=0, help="use chain-of-thought prompt to finetune, need to change train file name")

    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="svamp_program_sft", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="Salesforce", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="codegen-350M-mono", help="The bert model name to used")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="learning rate of the AdamW optimizer")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=40, help="The number of epochs to run")
    parser.add_argument('--fp16', type=int, default=1, choices=[0,1], help="using fp16 to train the model")


    # testing a pretrained model
    parser.add_argument('--error_file', type=str, default="results/error.json", help="The file to print the errors")
    parser.add_argument('--result_file', type=str, default="results/res.json", help="The file to print the errors")

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        logger.info(f"{k} = {args.__dict__[k]}")
    return args


def train(args, model, train_dataloader: DataLoader, num_epochs: int,
          tokenizer: PreTrainedTokenizerFast, valid_dataloader: DataLoader = None, all_metadata= None, test_dataloader: DataLoader = None,
          res_file:str = None):

    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)

    runtime = GenericRuntime()

    optimizer, scheduler = get_optimizers(model, args.learning_rate, t_total)
    model.zero_grad()

    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, scheduler)
    if test_dataloader is not None:
        test_dataloader = accelerator.prepare(test_dataloader)


    best_accuracy = -1
    os.makedirs(f"model_files/{args.model_folder}", exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for iter, feature in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                loss = model(**feature).loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            if iter % 1000 == 0:
                logger.info(f"epoch: {epoch}, iteration: {iter}, current mean loss: {total_loss/iter:.2f}")
        logger.info(f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss/len(train_dataloader):.2f}")
        if valid_dataloader is not None:
            val_accuracy = evaluate(args, runtime, valid_dataloader, model, fp16=bool(args.fp16), tokenizer=tokenizer, res_file=res_file, all_metadata= all_metadata)
            test_accuracy = -1
            if test_dataloader is not None:
                test_accuracy = evaluate(args, runtime, test_dataloader, model, fp16=bool(args.fp16), tokenizer=tokenizer, res_file=res_file, all_metadata= all_metadata)
            if val_accuracy > best_accuracy:
                logger.info(f"[Model Info] Saving the best model with best valid accuracy {val_accuracy:.6f} at epoch {epoch} ("
                            f"valid accuracy: {val_accuracy:.6f} test accuracy: {test_accuracy:.6f}")
                best_accuracy = val_accuracy
                unwraped_model = accelerator.unwrap_model(model)
                unwraped_model.save_pretrained(f"model_files/{args.model_folder}")
                tokenizer.save_pretrained(f"model_files/{args.model_folder}")
    logger.info(f"[Model Info] Best validation performance: {best_accuracy}")


def evaluate(args, runtime:GenericRuntime, valid_dataloader: DataLoader, model: nn.Module, fp16:bool, tokenizer: PreTrainedTokenizerFast, all_metadata,
             res_file: str= None, ) -> float:
    model.eval()
    predictions = []
    # labels = []
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                module = accelerator.unwrap_model(model)
                ## Note: need to check if the underlying model has revised the "prepare_inputs_for_generation" method
                generated_ids = module.generate(input_ids=feature["input_ids"],
                                                attention_mask=feature["attention_mask"],
                                                num_beams=1,
                                                max_new_tokens=args.max_length,
                                                return_dict_in_generate=True).sequences
                ## do not do this line after the pad and gather
                generated_ids = generated_ids[:, feature["input_ids"].size(1):].contiguous()
                generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.eos_token_id)
                generated_ids = accelerator.gather_for_metrics(generated_ids)
                prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predictions.extend(prediction)

    correct = 0
    all_data = []
    for idx, (prediction, metadata) in enumerate(zip(predictions, all_metadata)):
        try:
            if args.dft:
                code = ""
                ans = prediction
            else:
                if args.cot_ft:
                    code = prediction.split("\nA: ")[1]
                    ans = prediction.split("The answer is ")[1]
                    if ans.endswith("."):
                        ans = ans[:-1]
                else:
                    predicted_code = prediction
                    code, ans = run_code(runtime=runtime, code_gen=predicted_code, answer_expr="solution()")
        except:
            predicted_code = "<split failed>"
            code = ""
            ans = None
        score = 0
        try:
            numeric_ans = float(ans)
        except:
            numeric_ans = None
        if numeric_ans is not None and math.fabs(numeric_ans - float(metadata["answer"])) < 1e-2:
            correct += 1
            score = 1
        all_data.append({"question": metadata["question"],
                         "answer": metadata["answer"],
                         "predicted_answer": numeric_ans,
                         "score": score,
                         "code": code,
                         "generation": prediction})
    accuracy = correct / len(predictions)
    logger.info(f"Validation accuracy: {accuracy:.6f}")
    if res_file is not None:
        write_data(file=res_file, data=all_data)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="T5 for gsm8k experiments")
    args = parse_arguments(parser)
    set_seed(args.seed)
    os.makedirs("results", exist_ok=True)
    lm_model_name = args.bert_model_name if args.bert_folder == "" or args.bert_folder=="none" else f"{args.bert_folder}/{args.bert_model_name}"

    tokenizer = AutoTokenizer.from_pretrained(lm_model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(lm_model_name, pad_token_id=tokenizer.eos_token_id, low_cpu_mem_usage=True)

    logger.info("[Data Info] Reading all data")
    if "svamp" in args.train_file:
        dataset = read_from_svamp(dataset_file_path=args.train_file, split="train", direct_finetune=bool(args.dft))
        eval_dataset = read_from_svamp(dataset_file_path=args.dev_file, split="dev", direct_finetune=bool(args.dft))
    elif "gsm8k" in args.train_file:
        dataset = read_from_dataset(dataset_file_path=args.train_file, split="train", direct_finetune=bool(args.dft))
        eval_dataset = read_from_dataset(dataset_file_path=args.dev_file, split="dev", direct_finetune=bool(args.dft))
    elif "MathQA" in args.train_file:
        dataset = read_from_mathqa(dataset_file_path=args.train_file, split="train", direct_finetune=bool(args.dft))
        eval_dataset = read_from_mathqa(dataset_file_path=args.dev_file, split="dev", direct_finetune=bool(args.dft))
    if args.train_num > 0:
        dataset = dataset.select(range(args.train_num))
    if args.dev_num > 0:
        eval_dataset = eval_dataset.select(range(args.dev_num))
    logger.info(f"[Data Info] Training instances: {len(dataset)}, Validation instances: {len(eval_dataset)}")
    logger.info(f"[Data Info] Tokenizzing the dataset")
    with accelerator.main_process_first():
        train_tokenized_data = dataset.map(
            lambda x: tokenize_data(x, tokenizer=tokenizer, is_train=True,
                                    direct_finetune=bool(args.dft),
                                    cot_finetune=bool(args.cot_ft),
                                    max_length=args.max_length),
            batched=True,
            load_from_cache_file=False,
            num_proc=args.num_proc,
            remove_columns=dataset.column_names
        )
        eval_tokenized_dataset = eval_dataset.map(
            lambda x: tokenize_data(x, tokenizer=tokenizer, is_train=False,
                                    direct_finetune=bool(args.dft),
                                    cot_finetune=bool(args.cot_ft),
                                    max_length=args.max_length),
            batched=True,
            load_from_cache_file=False,
            num_proc=args.num_proc,
            remove_columns=eval_dataset.column_names
        )
    # Prepare data loader
    logger.info("[Data Info] Loading data")
    collator = PaddedDataCollator(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_tokenized_data.remove_columns("metadata"), batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collator)
    valid_dataloader = DataLoader(eval_tokenized_dataset.remove_columns("metadata"), batch_size=args.batch_size, shuffle=False, num_workers=8,
                                  collate_fn=collator)
    res_file = f"results/{args.model_folder}.res.json"
    err_file = f"results/{args.model_folder}.err.json"
    # Read dataset
    if args.mode == "train":
        # Train the model
        train(args, model, train_dataloader,
                  num_epochs= args.num_epochs,
                  valid_dataloader = valid_dataloader,
                  tokenizer=tokenizer, all_metadata=eval_tokenized_dataset["metadata"],
                  res_file=res_file)
    else:
        runtime = GenericRuntime()
        model = AutoModelForCausalLM.from_pretrained(f"model_files/{args.model_folder}")
        model, valid_dataloader = accelerator.prepare(model, valid_dataloader)
        test_accuracy = evaluate(args, runtime, valid_dataloader, model, fp16=bool(args.fp16), tokenizer=tokenizer, res_file=res_file, all_metadata=eval_tokenized_dataset["metadata"])

if __name__ == "__main__":
    # logger.addHandler(logging.StreamHandler())
    main()

