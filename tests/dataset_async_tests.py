# CUDA_VISIBLE_DEVICES=3 python -m my_project.tests.dataset_async_tests
import time
import torch
import asyncio
from typing import List
import numpy as np
import argparse
import gc
from my_project.engine.async_llm_engine import AsyncLLMEngine
from my_project.tests.utils import TestUtils
from transformers import (
    AutoModelForCausalLM,
)

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

paris_short = "Paris is the capital city of"
paris_long = \
"""
Paris is the capital of France and is located in the north central part of the country.  It is the largest city in France and the second largest city in the European Union.  It is also the most visited city in the world, with over 31 million visitors per year.  The city is home to many famous landmarks, including the Eiffel Tower, the Louvre, and Notre Dame Cathedral.  It is also home to a large number of museums, including the Musée dOrsay, the Musée du Louvre, and the Musée dArt Moderne de la Ville de Paris.  The city is also home to a large number of parks, including the Jardin du Luxembourg and the Parc de la Villette.  The city is also home to a large number of universities, including the Sorbonne and the Université ParisSorbonne.  The city is also home to a large number of businesses, including the headquarters of many large French companies.  The city is also home to a large number of restaurants, including many Michelinstarred restaurants.  The city is also home to a large number of hotels, including many luxury hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs.  The city is also home to a large number of theaters, including many famous theaters.  The city is also home to a large number of cinemas, including many famous cinemas.  The city is also home to a large number of concert halls, including many famous concert halls.  The city is also home to a large number of sports stadiums, including many famous sports stadiums.  The city is also home to a large number of shopping malls, including many famous shopping malls.  The city is also home to a large number of parks, including many famous parks.  The city is also home to a large number of museums, including many famous museums.  The city is also home to a large number of universities, including many famous universities.  The city is also home to a large number of businesses, including many famous businesses.  The city is also home to a large number of restaurants, including many famous restaurants.  The city is also home to a large number of hotels, including many famous hotels.  The city is also home to a large number of nightclubs, including many famous nightclubs
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-2-8B", help="model name or path")
parser.add_argument("--batch_size", type=int, default=1) # 71 is max @ 128 PL & 128 GL
parser.add_argument("--prompt_len", type=int, default=1024,  help="prompt length") # DEFAULT: 512
parser.add_argument("--gen_len", type=int, default=1024,  help="number of tokens to generate") # DEFAULT: 32
parser.add_argument("--cache_dir", type=str, default="/scratch/kumichae", help="cache dir for model name")

args = parser.parse_args()
gc.collect()

# Paranoid validation
async def shareGPT_test():
    TU.print_important(f"START TEST: ShareGPT Dataset Test | TID: {test_id}")
    reqs, hf_output_ids = TU.dump_hf_results()
    hf_output_ids3 = []
    for i,req in enumerate(reqs):
        with torch.no_grad():
            batch_size = len(req)
            gen_lens = torch.LongTensor([args.gen_len]*batch_size)
            await engine.put(req, gen_lens)
            await engine.test_done.wait()
            # engine.outputs = []
            input_tokens = engine.tokenizer.batch_encode_plus(req, return_tensors="pt",
                padding="max_length", max_length=args.prompt_len, truncation=True)
            input_tokens.to(TU.device)

            hf_output_ids3.append(engine.model.generate(**input_tokens, max_new_tokens=args.gen_len, do_sample=False).to("cpu"))
            if not torch.equal(hf_output_ids[i], hf_output_ids3[i]):
                raise RuntimeError("HF OUTPUTS NOT THE SAME") 

            engine.test_done.clear()
            
    TU.validate_shareGPT_outputs(test_id, engine.outputs, hf_output_ids, reqs)

async def correctness_dataset_test(reqs_per_s=10):
    TU.print_important(f"START TEST: Async {TU.dataset} Correctness Dataset Test")
    reqs, hf_output_ids = TU.dump_hf_results()
    gen_lens = []
    for batch_in, batch_out in zip(reqs, hf_output_ids):
        prompt_lens = torch.Tensor([(len(x) if len(x) <= args.prompt_len else args.prompt_len) for x in batch_in])
        out_lens = torch.Tensor([len(x) for x in batch_out])
        gen_lens.append(out_lens - prompt_lens)

    engine = TU.engine
    asyncio.create_task(engine.start_loops())
    for batch in reqs:
        asyncio.create_task(engine.put(batch))
        if reqs_per_s == float("inf"):
            continue

        interval = np.random.exponential(1.0 / reqs_per_s)
        engine.test_done.clear()
        await asyncio.sleep(interval)
    TU.print_important(f"WAITING FOR {TU.dataset} Correctness Dataset Test TO FINISH")
    await engine.test_done.wait()
    TU.print_important(f"FINISHED TEST: {TU.dataset} Correctness Dataset Test")
    TU.validate_shareGPT_outputs(0, engine.outputs, hf_output_ids, reqs)

async def avg_size_dataset_test():
    TU.print_important(f"START TEST: {TU.dataset} Avg Size Dataset Test")
    TU.init_data_iterator()
    for _ in range(10000):
        batch = next(TU.data_iterator)
        batch = [x["prompt"] for x in batch]
        asyncio.create_task(engine.put(batch, timestamp=time.time()))

    asyncio.create_task(engine.start_loops())
    TU.print_important(f"\nWAITING FOR {TU.dataset} Avg Size Dataset Test TO FINISH")
    await asyncio.sleep(900)
    engine.outputs = []
    
    TU.print_important(f"FINISHED TEST: {TU.dataset} Avg Size Dataset Test")
    TU.dump_results("Nopad", 
                    bs=engine.bs,
                    token_nums=engine.token_nums,
                    is_avg_sz=True)

async def tp_dataset_test(reqs_per_s=[10], n_reqs=3):
    TU.print_important(f"START TEST: Async {TU.dataset} TP Dataset Test")
    # WARMUP
    asyncio.create_task(engine.start_loops())
    print("WARMING UP")
    for _ in range(3):
        await engine.put([paris_long]*3, torch.tensor([512]*3))
        await engine.test_done.wait()
        engine.outputs = []
        engine.bs = [] 
        engine.token_nums = []
        engine.test_done.clear()
        torch.cuda.empty_cache()
    print("WARMUP DONE")

    TU.init_data_iterator()
    total_timings = []
    req_timings = []
    req_fwd_timings = []
    GTs = []
    for req_rate in reqs_per_s:
        start = time.time()
        for _ in range(n_reqs):
            batch = next(TU.data_iterator)
            batch = [x["prompt"] for x in batch]
            asyncio.create_task(engine.put(batch, timestamp=time.time()))
            if req_rate == float("inf"):
                continue
            interval = np.random.exponential(1.0 / req_rate)
            engine.test_done.clear()
            await asyncio.sleep(interval)
    
        TU.print_important(f"\nWAITING FOR {TU.dataset} Dataset Test w/ req_rate {req_rate} TO FINISH")
        await engine.test_done.wait()
        end = time.time() - start
        total_timings.append(end)
        engine.test_done.clear()
        gts =  []
        timings = []
        fwd_timings = []
        for output in engine.outputs:
            _, _, pls, cls, req_timing, _ = output
            req_total_timing, req_fwd_timing = req_timing[:, 0], req_timing[:, 1]
            for pl, cl, rt, rfwt in zip(pls.tolist(), cls.tolist(), req_total_timing.tolist(), req_fwd_timing.tolist()):
                gts.append((cl-pl))
                timings.append(rt)
                fwd_timings.append(rfwt)
        GTs.append(gts)
        req_timings.append(timings)
        req_fwd_timings.append(fwd_timings)
        engine.outputs = []
    
    TU.print_important(f"FINISHED TEST: {TU.dataset} Performance Dataset Test")
    TU.dump_results("Nopad", 
                    reqs_per_s, 
                    total_timings, 
                    GTs, 
                    req_timings, 
                    req_fwd_timings, 
                    engine.bs, 
                    engine.token_nums,
                    is_avg_sz=False)

if __name__ == "__main__":
    shareGPT = 'liyucheng/ShareGPT90K'
    alpaca = 'yahma/alpaca-cleaned'
    n_reqs = 200  # 250 ShareGPT | 300 alpaca
    seed = 42
    # For deterministic request arrival times
    np.random.seed(seed)

    engine = AsyncLLMEngine(prompt_len=args.prompt_len, gen_len=args.gen_len)
    # engine = None
    TU = TestUtils(engine=engine, 
                   seed=seed, 
                   batch_size=1, 
                   n_reqs=n_reqs, 
                   prompt_len=args.prompt_len, 
                   gen_len=args.gen_len, 
                   dataset=shareGPT)

    test_id = 0
    # shareGPT_reqs_per_s = [0.1, 1.0, 2.0, 3, 4, 6, 8, 12, 16] # FOR KV CACHE
    # alpaca_reqs_per_s = [0.2, 4,8,12,16,20,24,28,32] # KV CACHE
    # shareGPT_reqs_per_s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # SimpleReqManager LARGE
    # alpaca_reqs_per_s = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2] # SimpleReqManager LARGE
    shareGPT_reqs_per_s = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2,0.24,0.28,0.3, 0.4] # BOTH SMALL
    alpaca_reqs_per_s = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] # BOTH SMALL
    # res = asyncio.run(correctness_dataset_test(1000)) 
    res = asyncio.run(tp_dataset_test(shareGPT_reqs_per_s, n_reqs=n_reqs))
    # res = asyncio.run(avg_size_dataset_test())
    TU.print_important("ALL TESTS DONE")