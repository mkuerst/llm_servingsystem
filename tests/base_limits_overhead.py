# CUDA_VISIBLE_DEVICES=3 python -m my_project.tests.base_limits_overhead
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=./ kernprof -l ./my_project/tests/base_limits_overhead.py
import pstats
import cProfile
import asyncio
import argparse
import gc
import my_project.tests.bench_utils as utils
import torch
from typing import List
import time
import numpy as np

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from my_project.engine.async_llm_engine import AsyncLLMEngine
# import logging
# logging.disable(logging.CRITICAL)

device = torch.cuda.current_device()

MB = 1024 ** 2
GB = 1024 ** 3
paris_short = "Paris is the capital city of"
paris_long = \
"""
Paris is the capital of France and is located in the north central part of the country.  It is the largest city in France and the second largest city in the European Union.  It is also the most visited city in the world, with over 30 million visitors per year.  The city is home to many famous landmarks, including the Eiffel Tower, the Louvre, and Notre Dame Cathedral.  It is also home to a large number of museums, including the Musée dOrsay, the Musée du Louvre, and the Musée dArt Moderne de la Ville de Paris.  The city is also home to a large number of parks, including the Jardin du Luxembourg and the Parc de la Villette.  The city is also home to a large number of universities, including the Sorbonne and the Université ParisSorbonne.  The city is also home to a large number of businesses, including the headquarters of many large French companies.  The city is also home to a large number of restaurants, including many Michelinstarred restaurants.  The city is also home to a large number of hotels, including many luxury hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs
"""
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name or path")
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--trials", type=int, default=3,  help="Number of token generation iterations") #50
parser.add_argument("--prompt_len", type=int, default=1024,  help="prompt length") # DEFAULT: 512
parser.add_argument("--gen_len", type=int, default=1024,  help="number of tokens to generate") # DEFAULT: 32
parser.add_argument("--cache_dir", type=str, default="/scratch/kumichae", help="cache dir for model name")
args = parser.parse_args()
gc.collect()

def memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    memory_info = f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB\nreserved memory: {reserved_memory / (1024 ** 2):.2f} MB\n"
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))
    memory_info += f"Peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
    print(memory_info)

def log_summary(total_timings, forward_timings=[], name="", bs=args.batch_size, gen_len=args.gen_len, prompt_len=args.prompt_len):
    filename = f"{name}_BS{bs}_PL{prompt_len}_GL{gen_len}"
    data_dir = "my_project/tests/overhead_results/"
    out_filename = data_dir+filename+".txt"

    # Log output
    print(f"Summary:")
    print(f"total_timings = {total_timings}")
    total_timings = np.array(total_timings)

    with open(out_filename, "w") as file:
        file.write("name BS PL GL dataset\n")
        file.write(f"{name} {bs} {prompt_len} {gen_len} overhead\n")
        file.write("total_timings\n")
        total_timings_string = ' '.join(f"{num:.3f}" for num in total_timings)
        file.write(f"{total_timings_string}\n")
        if len(forward_timings):
            file.write("forward_timings\n")
            forward_timings_string = ""
            for fwd_timing in forward_timings:
                fwd_timing = fwd_timing[0,1]
                forward_timings_string += f"{fwd_timing:.3f} "
            file.write(f"{forward_timings_string}\n")

def run_hf_lat_bench(bs=args.batch_size, trials=args.trials, gen_len=args.gen_len):
    print(f"\nRUNNING BENCHMARK HF batch_size={bs} gen_len={gen_len}")
    req = paris_long
    total_timings = []
    output_ids = []
    prompts = [req]*bs
    
    for _ in range(3):
        input_tokens = async_llm_engine.tokenizer.batch_encode_plus(prompts, return_tensors="pt",
            padding=False, max_length=args.prompt_len, truncation=True)
        input_tokens.to(device)
        with torch.no_grad():
            async_llm_engine.model.generate(**input_tokens, max_new_tokens=args.gen_len, do_sample=False)
        torch.cuda.empty_cache()

    for _ in range(trials):
        start = time.time()
        input_tokens = async_llm_engine.tokenizer.batch_encode_plus(prompts, return_tensors="pt",
            padding="max_length", max_length=args.prompt_len, truncation=True)
        with torch.no_grad():
            input_tokens.to(device)
            async_llm_engine.model.stage = "prefill"
            output_ids.append(async_llm_engine.model.generate(**input_tokens, max_new_tokens=args.gen_len, do_sample=False))
        end = time.time()
        total_timings.append(end - start)
    
    for out in output_ids:
        assert all([len(x)==args.prompt_len+args.gen_len for x in out]), "HF BENCH: ouput length mismatch"

    log_summary(total_timings, name="HF")
    
async def run_engine_profiling(bs=args.batch_size, trials=args.trials, gen_len=args.gen_len):
    asyncio.create_task(async_llm_engine.start_loops(start_recv=False))
    req = paris_long
    prompts = [req]*bs
    gen_lens = torch.tensor([gen_len]*bs).cpu()
    # WARMUP
    for _ in range(3):
        await async_llm_engine.put(prompts, gen_lens)
        await async_llm_engine.test_done.wait()
        async_llm_engine.outputs = []
        async_llm_engine.test_done.clear()
        # torch.cuda.empty_cache()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(1):
        start = time.time()
        await async_llm_engine.put(prompts, gen_lens)

    await async_llm_engine.test_done.wait()
    end = time.time() - start
    profiler.disable()
    print("PRINTING OVERHEADS FOR PROFILING RUN:")
    for output in async_llm_engine.outputs:
        overhead = end - (output[4][0][1] / 1000)
        print(f"GPU TIME: {output[4][0][1] / 1000}")
        print(f"ASYNC_STEP TOTAL:{output[4][0][2]}")
        print(f"ASYNC STEP OVERHEAD: {output[4][0][2] - (output[4][0][1] / 1000)}")
        print(f"CPU FORWARDING TOTAL: {output[4][0][3]}")
        print(f"CPU FORWARDING OVERHEAD: {output[4][0][3] - (output[4][0][1] / 1000)}")
        print(f"TOTAL OVERHEAD: {overhead}")

    stats = pstats.Stats(profiler)
    include_fcts = '(my_project/engine|my_project/common/req_manager|my_project/common/mem_manager|infer_struct|copy_|infer_utils|asyncio|_update_model_kwargs_for_generation|prepare_inputs_for_generation)'
    # include_fcts = '(get_fitting_batches|remove_reqs|put|add_reqs|wait_for_new_reqs|prepare|alloc|init_req_to_token_indexes|init_some_extra_state|copy_kv_index_to_req|prefill|decode|batch_encode_plus|get_new_reqs|_update_model_kwargs_for_generation|asyncio)'
    stats.sort_stats('cumtime').print_stats()
    return

async def run_llm_engine_lat_bench(bs=args.batch_size, trials=args.trials, gen_len=args.gen_len):
    asyncio.create_task(async_llm_engine.start_loops(start_recv=False))

    req = paris_long
    total_timings = []
    forward_timings = []
    output_ids = []
    prompts = [req]*bs
    gen_lens = torch.tensor([gen_len]*bs).cpu()

    # WARMUP
    for _ in range(3):
        await async_llm_engine.put(prompts, gen_lens)
        await async_llm_engine.test_done.wait()
        async_llm_engine.outputs = []
        async_llm_engine.test_done.clear()
        torch.cuda.empty_cache()

    for _ in range(trials):
        start = time.time()
        await async_llm_engine.put(prompts, gen_lens)
        await async_llm_engine.test_done.wait()
        end = time.time()
        output_ids.append(async_llm_engine.outputs[0][0])
        forward_timings.append(async_llm_engine.outputs[0][4])
        total_timings.append(end-start)
        # total_timings.append(async_llm_engine.outputs[0][4][0][0])

        async_llm_engine.outputs = []
        async_llm_engine.test_done.clear()
        torch.cuda.empty_cache()

    for out in output_ids:
        assert all([len(x)==args.prompt_len+args.gen_len for x in out]), "LLM_ENGINE BENCH: ouput length mismatch"
    log_summary(total_timings=total_timings, forward_timings=forward_timings, name="Engine")

async def run_llm_engine_limits_bench(gen_len=args.gen_len):
    asyncio.create_task(async_llm_engine.start_loops())

    req = paris_long
    for i in range(8,1000):
        with torch.no_grad():
            prompts = [req]*i
            gen_lens = torch.tensor([gen_len]*i)#.to(device)
            await async_llm_engine.put(prompts, gen_lens)
            await async_llm_engine.test_done.wait()
            async_llm_engine.test_done.clear()
            # async_llm_engine.outputs = []
            out = async_llm_engine.outputs[-1][0]
            torch.cuda.empty_cache()
            print(f"Batch_size {i} succeeded: w/ {out.shape[0] * out.shape[1]} tokens")
            memory_usage(torch.cuda.current_device())

async def run_hf_limits_bench():
    req = paris_long
    for i in range(1,1000):
        prompts = [req]*i
        input_tokens = async_llm_engine.tokenizer.batch_encode_plus(prompts, return_tensors="pt",
            padding=False, max_length=args.prompt_len, truncation=True)
        input_tokens.to(device)
        with torch.no_grad():
            output = async_llm_engine.model.generate(**input_tokens, max_new_tokens=args.gen_len, do_sample=False)
        del input_tokens, output
        torch.cuda.empty_cache()
        memory_usage(torch.cuda.current_device())
        print(f"Batch_size {i} succeeded:")


if __name__ == '__main__':
    async_llm_engine = AsyncLLMEngine(prompt_len=args.prompt_len, gen_len=args.gen_len)
    memory_usage(torch.cuda.current_device())

    # Last output: 71 batch_size w/ 256 len
    ### FIND MAX BATCH SIZE @ MAX (prompt_len=128 & gen_len=128)
    # Total of 18176 tokens.

    # HF
    # PL 1024 | GL 1024: max_bs 9 |  tokens

    # ASYNC LLM ENGINE
    # PL 2048 GL 128: max_bs 4 | 8704 tokens
    # PL 2048 GL 32: max_bs 4 | 8320 tokens
    # PL 1024 GL 128: max_bs 8 | 9216 tokens
    # PL 1024 GL 32: max_bs 9 | 9504 tokens
    # PL 1024 GL 1024: max_bs ? | 9504 tokens

    # res = asyncio.run(run_llm_engine_lat_bench())
    # run_hf_lat_bench()
    # res = asyncio.run(run_llm_engine_limits_bench())
    res = asyncio.run(run_engine_profiling())

        
### Paris is the capital city of ### NOPAD
# GL 32: max_bs 300 ...
# PL 2048 GL 32 : max_bs 4 | 8320 tokens
# PL 1024 GL 1024: max_bs 25 | 52000 tokens


