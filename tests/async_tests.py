
# CUDA_VISIBLE_DEVICES=3 python -m my_project.engine.async_tests
import re
import torch
import asyncio
from typing import List, AsyncGenerator, Tuple
import itertools
import numpy as np
import os

device = torch.cuda.current_device()

import argparse
import gc

import os
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name or path")
parser.add_argument("--adapter_size", type=int, default=2, help="lora adapters swapping")
parser.add_argument("--batch_size", type=int, default=60) # 71 is max @ 128 PL & 128 GL
parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--trials", type=int, default=1,  help="Number of token generation iterations")
parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
parser.add_argument("--prompt_len", type=int, default=128,  help="prompt length") # DEFAULT: 512
parser.add_argument("--gen_len", type=int, default=128,  help="number of tokens to generate") # DEFAULT: 32
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
parser.add_argument("--cache_dir", type=str, default="/scratch/kumichae", help="cache dir for model name")
args = parser.parse_args()
gc.collect()

from my_project.engine.async_llm_engine import AsyncLLMEngine

RESET = "\033[0m"
COLORS = {
    'BLUE': "\033[94m",  # Blue
    'SUCCESS': "\033[92m",   # Green
    'FAIL': "\033[91m",  # Red
    'YELLOW': "\033[93m",# Yellow
    'MAGENTA': "\033[95m"# Magenta
}

# len 6
req1 = "Paris is the capital city of"
# len 7
req2 = "Berlin is the capital city of Germany"
simple_reqs = [req1, req2]

gl_8_8 = [2,3,4,5,6,7,8,2]
gl_12_64 = [32,40,64,32,2,24,25,7,9,10,64,36]
gl_32_64 = [64]*32
gl32 = [32]
gl64 = [64]
gl128 = [128]

def print_important(msg: str):
    print(COLORS["YELLOW"]+"-"*80)
    print(msg)
    print(COLORS["YELLOW"]+"-"*80+RESET)

def re_findall(str):
    return re.findall(r"[\w']+|[.,!?;]", str)

def reset():
    global expected
    global test_id
    expected = dict(
        prompt_lens = [],
        gen_lens = [],
        word_lens = []
    )
    async_llm_engine.outputs = []
    async_llm_engine.id_generator.current_id = 0
    async_llm_engine.steps = 0
    test_id += 1

def validate_outputs(tid, outputs, expected, n_reqs):
    print(COLORS['YELLOW']+"#"*60)
    print(f"VALIDATING OUTPUTS OF TEST {tid}")
    print("#"*60+RESET)
    eos_token_id = async_llm_engine.tokenizer.eos_token_id

    failed = 0
    succeeded = 0

    expected_prompt_lens = expected["prompt_lens"]
    expected_gen_lens = expected["gen_lens"]
    expected_word_lens = expected["word_lens"]

    processed_reqs = 0
    expected_processed_reqs = sum([len(x) for x in expected_prompt_lens])

    CUDA_oom = "CUDA out of memory"

    for output in outputs:
        out_ids, batch_meta_ids, error = output
        processed_reqs += len(out_ids)


        for i, (batch_id, req_id) in enumerate(batch_meta_ids):
            word_len = len(out_ids[i][out_ids[i] != eos_token_id][1:])
            expected_word_len = expected_word_lens[batch_id][req_id] + expected_gen_lens[batch_id][req_id]
            try:
                assert word_len == expected_word_len
                succeeded += 1
            except Exception as e:
                error_msg = f"\nException: {error}" if error != None else None
                print(COLORS["FAIL"]+f"FAILED: ({batch_id},{req_id}). Expected length: {expected_word_len}. Actual len: {word_len}{error_msg}"+RESET)
                failed += 1
                if CUDA_oom in str(error):
                    print(COLORS["FAIL"]+"STOPPING VALIDATION: GPU ran out of memory!")
                    break

    print(COLORS['YELLOW']+"#"*60)
    color = COLORS["SUCCESS"] if (failed == 0 and processed_reqs == expected_processed_reqs) else COLORS["FAIL"]
    print(color+f'{succeeded}/{expected_processed_reqs} requests passed')
    if processed_reqs != expected_processed_reqs:
        print(f"{expected_processed_reqs-processed_reqs}/{expected_processed_reqs} didn't finish.")
    print(COLORS["YELLOW"]+"#"*60+RESET)
            
async def get_simple_reqs(reqs: List[str], n_reqs: int, req_rate: float, gen_lens: List[int]) -> AsyncGenerator[Tuple[List[str]], torch.Tensor]:
    inputs = itertools.cycle(reqs)
    for req in inputs:
        if n_reqs <= 0:
            break
        n_reqs -= 1
        
        batch_size = len(gen_lens)
        # TODO prompt_len after mem. manager is implemented
        gl = torch.tensor(gen_lens)
        yield [req]*batch_size, gl, [args.prompt_len]*batch_size

        if req_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / req_rate)
        await asyncio.sleep(interval)

async def test_simple_reqs(reqs: List[str], gen_lens: List[str], test_name="test_simple_reqs", n_reqs=1, req_rate=10):
    print(COLORS["YELLOW"]+f"START TEST: {test_name} | TID: {test_id}"+RESET)
    tasks: List[asyncio.Task] = []

    with torch.no_grad():
        async for req in get_simple_reqs(reqs, n_reqs, req_rate, gen_lens):
            prompts, gen_lens, prompt_lens = req
            expected["gen_lens"].append(gen_lens.to("cpu"))
            word_lens = [len(x.split()) for x in prompts]
            expected["prompt_lens"].append(prompt_lens)
            expected["word_lens"].append(word_lens)
            task = asyncio.create_task(async_llm_engine.put(prompts, gen_lens))
            tasks.append(task)
        await asyncio.gather(*tasks)
    
        if async_llm_engine.is_running:
            await async_llm_engine.engine_shutdown.wait()
            print(COLORS["YELLOW"]+f"\nWAITING FOR TEST {test_id} TO FINISH"+RESET)
            async_llm_engine.engine_shutdown.clear()

    print(COLORS["YELLOW"]+f"\nFINISHED TEST: {test_name} | TID: {test_id}"+RESET)
    validate_outputs(test_id, async_llm_engine.outputs, expected, n_reqs)
    reset()
        
async def main():
    # FORMAT: #REQS x BATCH_SIZE - MAX_RANGE(GEN_LENS) / RATE
    # await test_simple_reqs([req2], gen_lens=gl32, test_name="gl32 single req DE", n_reqs=1)
    # await test_simple_reqs([req2], gen_lens=gl64, test_name="gl64 single req DE", n_reqs=1)
    # await test_simple_reqs([req2], gen_lens=gl128, test_name="gl128 single req DE", n_reqs=1)

    # await test_simple_reqs([req2], gl_8_8, test_name="1x8 DE", n_reqs=1, req_rate=10)
    # await test_simple_reqs(simple_reqs, gl_8_8, "2x8-8/10 FR/DE", n_reqs=2, req_rate=10)
    # await test_simple_reqs(simple_reqs, gl_8_8, "4x8-8/100 FR/DE", n_reqs=4, req_rate=100)
    # await test_simple_reqs(simple_reqs, gl_8_8, "10x8-8/100000 FR/DE", n_reqs=10, req_rate=100000)

    # await test_simple_reqs(simple_reqs, gl_12_64, "4x12-64/10 FR/DE", 4, 10)
    # await test_simple_reqs(simple_reqs, gl_12_64, "6x12-64/100000000 FR/DE", 6, req_rate=100000000)

    # await test_simple_reqs(simple_reqs, gl_32_64, "1x32-(all)64 FR/DE", 1)
    # await test_simple_reqs(simple_reqs, gl32, "100x32/2 FR/DE", 100, 2)

    # await test_simple_reqs(simple_reqs, gl_12_64, "", 4, 2)

    # await test_simple_reqs(simple_reqs, gl_8_8, "40x8-8/2 FR/DE", 40, 2)
    
    return 

if __name__ == "__main__":
    async_llm_engine = AsyncLLMEngine(prompt_len=args.prompt_len, gen_len=args.gen_len)
    eos_token_id = async_llm_engine.tokenizer.eos_token_id

    test_id = 0
    expected = dict(
        prompt_lens = [],
        gen_lens = [],
        word_lens = []
    )
    res = asyncio.run(main())
    print_important("ALL TESTS DONE")