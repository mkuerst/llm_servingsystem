import asyncio
import itertools
import gc
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import List, AsyncGenerator, Tuple
import numpy as np
import os
import json
import datasets
from functools import partial
import random
from my_project.engine.async_llm_engine import AsyncLLMEngine

RESET = "\033[0m"
COLORS = {
    'BLUE': "\033[94m",  # Blue
    'SUCCESS': "\033[92m",   # Green
    'FAIL': "\033[91m",  # Red
    'YELLOW': "\033[93m",# Yellow
    'MAGENTA': "\033[95m",# Magenta
    'ORANGE': "\033[1m\033[38;5;208m",   # Orange
}


class TestUtils():
    def __init__(self,
                 model_name="meta-llama/Meta-Llama-3-8B",
                 cache_dir="/scratch/kumichae",
                 prompt_len=1024,
                 gen_len=1024,
                 engine=None,
                 seed=42,
                 batch_size=60,
                 pin_memory=0,
                 n_reqs=10,
                 max_seq_len=2048,
                 dataset='liyucheng/ShareGPT90K'
        ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.gen_len = gen_len
        self.prompt_len = prompt_len
        self.engine = engine
        self.seed = seed 
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.n_reqs = n_reqs
        self.max_seq_len=max_seq_len
        self.dataset = dataset
        torch.manual_seed(seed)
        random.seed(seed)
        self.device = torch.cuda.current_device()

    def print_important(self, msg: str):
        print(COLORS["YELLOW"]+"-"*80)
        print(msg)
        print(COLORS["YELLOW"]+"-"*80+RESET)

    def print_colored_strings(self, str1, str2):
        min_len = min(len(str1), len(str2))
        match_index = min_len
        for i in range(min_len):
            if str1[i] != str2[i]:
                match_index = i
                break

        print(f"Async LLM Engine:\n")
        print(COLORS["SUCCESS"]+str1[:match_index] + COLORS["FAIL"]+str1[match_index:])
        print("-"*60)
        print(f"HF Result:\n")
        print(COLORS["SUCCESS"]+str2[:match_index] + COLORS["FAIL"]+str2[match_index:]+RESET)

    def validate_shareGPT_outputs(self, tid, outputs, expected, reqs):
        self.print_important(f"VALIDATING OUTPUTS OF TEST shareGPT test | TID: {tid}")
        failed = 0
        succeeded = 0

        processed_reqs = 0
        expected_processed_reqs = sum([len(x) for x in expected])

        CUDA_oom = "CUDA out of memory"
        eos_token_id = self.eos_token_id
        check_list = []
        for row in expected:
            x = len(row)
            check_list.append([0]*x)

        for output in outputs:
            out_ids, batch_meta_ids, error = output[0], output[1], output[-1]
            processed_reqs += len(out_ids)
            for out_id, (b_id, r_id) in zip(out_ids, batch_meta_ids):
                out_id = out_id[out_id != eos_token_id]
                result = expected[b_id][r_id]
                result = result[result != eos_token_id]
                check_list[b_id][r_id] = 1
                try:
                    assert torch.equal(out_id, result) 
                    succeeded += 1
                except Exception as e:
                    error_msg = f"\nException: {error}" if error != None else None
                    print("\n"+COLORS["FAIL"]+"*"*100)
                    print(f"***FAILED: ({b_id},{r_id}) | LENS (Actual, Expected): {len(out_id), len(result)} | Error: {error_msg}")
                    print("-"*60)
                    print(COLORS["ORANGE"]+f"Original prompt:\n{reqs[b_id][r_id]}"+COLORS["FAIL"])
                    print("-"*60)
                    actual_str = self.tokenizer.decode(out_id, skip_special_tokens=True)
                    result_str = self.tokenizer.decode(result, skip_special_tokens=True)
                    self.print_colored_strings(actual_str, result_str)
                    failed += 1
                    if CUDA_oom in str(error):
                        print(COLORS["FAIL"]+"STOPPING VALIDATION: GPU ran out of memory!")
                        break

        print(COLORS['YELLOW']+"#"*60)
        color = COLORS["SUCCESS"] if (failed == 0 and processed_reqs == expected_processed_reqs) else COLORS["FAIL"]
        print(color+f'{succeeded}/{expected_processed_reqs} requests passed')
        if processed_reqs != expected_processed_reqs:
            print(f"{expected_processed_reqs-processed_reqs}/{expected_processed_reqs} didn't finish.")
            missing_ids = [] 
            for i,row in enumerate(check_list):
                for j,r_id in enumerate(row):
                    if r_id == 0:
                        missing_ids.append((i, j))

            print(f"Missing Reqs:\n{missing_ids}")
        print(COLORS["YELLOW"]+"#"*60+RESET)

    def prepare_shareGPT(self, sample_raw):
        sample_raw = sample_raw['conversations']
        prompt = sample_raw["value"][0] if sample_raw["from"][0] == "human" else None
        return dict(prompt=prompt)

    def prepare_alpaca(self, sample_raw):
        template = {
            "description": "A shorter template to experiment with.",
            "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}",
            "prompt_no_input": "### Instruction:\n{instruction}",
        }
        if len(sample_raw["input"]):
            sample_text = template["prompt_input"].format(
                instruction=sample_raw["instruction"], input=sample_raw["input"]
            )
        else:
            sample_text = template["prompt_no_input"].format(
                instruction=sample_raw["instruction"]
            )
        return dict(prompt=sample_text)

    def custom_collate_fn(self, batch, max_batch_size):
        batch = random.sample(batch, random.randint(1, max_batch_size))
        return batch
    
    def init_data_iterator(self):
        prep_func = None
        if self.dataset == 'liyucheng/ShareGPT90K':
            prep_func = self.prepare_shareGPT
        elif self.dataset == 'yahma/alpaca-cleaned':
            prep_func = self.prepare_alpaca
        else:
            raise RuntimeError("Tried to prepare unknown dataset")

        dataset = datasets.load_dataset(self.dataset, cache_dir=self.cache_dir)
        dataset = dataset.map(lambda sample_raw: prep_func(sample_raw), remove_columns=dataset["train"].column_names)
        dataset = dataset.filter(lambda x: x["prompt"] is not None)
        dataloader = torch.utils.data.DataLoader(
            dataset["train"], shuffle=True, collate_fn=partial(self.custom_collate_fn, max_batch_size=self.batch_size),
            batch_size=self.batch_size, pin_memory=self.pin_memory,
        )
        self.data_iterator = itertools.cycle(iter(dataloader))

    def dump_results(self, 
                     name, 
                     reqs_per_s=[], 
                     timings=[], 
                     GTs=[], 
                     req_timings=[], 
                     req_fwd_timings=[], 
                     bs=[], 
                     token_nums=[], 
                     is_avg_sz=False):
        print(f"Saving results for {name} test on {self.dataset} dataset")
        try:
            dataset_name = self.dataset.split('/')[1]
        except:
            raise RuntimeError("Invalid dataset")

        if "ShareGPT90K" in dataset_name:
            dataset_name = "ShareGPT"
        if "alpaca-cleaned" in dataset_name:
            dataset_name = "Alpaca"

        res_filename = f"{name}_{dataset_name}_BS{self.batch_size}_PL{self.prompt_len}_GL{self.gen_len}"
        res_dir = f"my_project/tests/avg_sz_results/" if is_avg_sz  else f"my_project/tests/dataset_results/{dataset_name}/"
        out_filename = res_dir+res_filename+".txt"
        with open(out_filename, "a") as file:
            file.write("name BS PL GL dataset\n")
            file.write(f"{name} {self.batch_size} {self.prompt_len} {self.gen_len} {dataset_name}\n")
            if not is_avg_sz:
                file.write("reqs_per_s\n")
                reqs_per_s_string = ' '.join(f"{num:.3f}" for num in reqs_per_s)
                file.write(f"{reqs_per_s_string}\n")
                file.write("total_timings\n")
                total_timings_string = ' '.join(f"{num:.3f}" for num in timings)
                file.write(f"{total_timings_string}\n")

                file.write("GTs\n")
                for gts in GTs:
                    GTs_string = ' '.join(f"{num}" for num in gts)
                    file.write(f"{GTs_string}\n")

                file.write("req_timings\n")
                for timings in req_timings:
                    req_timings_string = ' '.join(f"{num}" for num in timings)
                    file.write(f"{req_timings_string}\n")

                file.write("req_fwd_timings\n")
                for timings in req_fwd_timings:
                    req_fwd_timings_string = ' '.join(f"{num}" for num in timings)
                    file.write(f"{req_fwd_timings_string}\n")

                file.write("secs_per_token\n")
                for timings, gts in zip(req_timings, GTs):
                    timings, gts = np.array(timings), np.array(gts)
                    tps = timings / gts
                    tps_string = ' '.join(f"{num}" for num in tps)
                    file.write(f"{tps_string}\n")

            if is_avg_sz:
                file.write("batch_sizes\n")
                bs_string = " ".join(f"{num}" for num in bs)
                file.write(f"{bs_string}\n")
                file.write("token_nums\n")
                token_nums_string = " ".join(f"{num}" for num in token_nums)
                file.write(f"{token_nums_string}\n")
        print("Saved results")


    def dump_hf_results(self):
        try:
            dataset_name = self.dataset.split('/')[1]
        except:
            raise RuntimeError("Invalid dataset")

        self.print_important("PREPARING HF RESULTS")
        filename = f"{dataset_name}_{self.n_reqs}nreqs_bs{self.batch_size}_{self.prompt_len}PL_{self.gen_len}GL_seed{self.seed}"
        data_dir = "my_project/tests/data/"
        reqs_filename = data_dir+"reqs_"+filename+".txt"
        out_filename = data_dir+"out_ids_"+filename+".pt"

        if os.path.exists(out_filename) and os.path.exists(reqs_filename):
            print("HF Results have already been generated. Loading from files.")
            with open(reqs_filename, 'r') as file:
                reqs = json.load(file)
            out_ids = torch.load(out_filename)
            self.engine = AsyncLLMEngine(prompt_len=self.prompt_len, gen_len=self.gen_len)
            return reqs, out_ids

        model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir, torch_dtype=torch.float16)
        model.to(self.device)
        model = model.eval()

        self.init_data_iterator()
        reqs = []
        hf_output_ids = []

        PLs = []
        GLs = []
        timings = []
        # max_len = self.engine.max_seq_length

        #WARMUP
        # print("HF WARMUP")
        # paris_prompt = "Paris is the capital city of"
        # for _ in range(3):
        #     input_tokens = self.engine.tokenizer.batch_encode_plus([paris_prompt]*3, return_tensors="pt",
        #         padding=False, max_length=self.prompt_len, truncation=True)
        #     input_tokens.to(self.device)
        #     out = self.engine.model.generate(**input_tokens, max_new_tokens=512, do_sample=False).to("cpu")
        #     del input_tokens, out
        #     torch.cuda.empty_cache()
        # print("FINISHED HF WARMUP")

        for _ in range(self.n_reqs):
            batch = next(self.data_iterator)
            batch = [x["prompt"] for x in batch]
            out_ids = []
            n_tokens = []
            for req in batch:
                s1 = time.time()
                input_tokens = self.tokenizer.encode_plus(req, return_tensors="pt",
                    padding=False, max_length=self.prompt_len, truncation=True)
                input_tokens.to(self.device)
                e1 = time.time()-s1
                input_ids = input_tokens["input_ids"]
                len_prompt = input_ids.shape[1]
                final_len = len_prompt + self.gen_len 
                n_tokens = 2048 if final_len > 2048 \
                    else final_len
                # max_new_tokens = max_len - len_prompt if final_len > max_len else self.gen_len

                s2 = time.time()
                ids = model.generate(**input_tokens, max_new_tokens=self.gen_len, do_sample=False).to("cpu")
                assert  ids.shape[1] <= n_tokens, "HF Result has too many tokens"
                out_ids.append(ids)
                e2 = time.time()-s2
                timings.append(e1+e2)
                PLs.append(len_prompt)
                GLs.append(ids.shape[1]-len_prompt)

            hf_output_ids.append(out_ids)
            reqs.append(batch)

            del out_ids, input_tokens, input_ids, n_tokens
            torch.cuda.empty_cache()

        torch.save(hf_output_ids, out_filename)
        with open(reqs_filename, 'w') as file:
            json.dump(reqs, file)

        # res_filename = f"HF_{dataset_name}_BS{self.batch_size}_PL{self.prompt_len}_GL{self.gen_len}"
        # res_dir = "my_project/tests/dataset_results/"
        # out_filename = res_dir+res_filename+".txt"

        # with open(out_filename, "w") as file:
        #     file.write("name BS PL GL\n")
        #     file.write(f"HF {self.batch_size} {self.prompt_len} {self.gen_len}\n")
        #     file.write("total_timings\n")
        #     total_timings_string = ' '.join(f"{num:.3f}" for num in timings)
        #     file.write(f"{total_timings_string}\n")
        #     file.write("PLs\n")
        #     PLs_string = ' '.join(f"{num}" for num in PLs)
        #     file.write(f"{PLs_string}\n")
        #     file.write("GLs\n")
        #     GLs_string = ' '.join(f"{num}" for num in GLs)
        #     file.write(f"{GLs_string}\n")

        self.print_important("HF RESULTS ARE READY")
        self.print_important("PREPARING ASYNC ENGINE")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        self.engine = AsyncLLMEngine(prompt_len=self.prompt_len, gen_len=self.gen_len)
        return reqs, hf_output_ids 
