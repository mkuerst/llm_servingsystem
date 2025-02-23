# CUDA_VISIBLE_DEVICES=3 python -m my_project.tests.shareGPT_server_tests
import torch
import asyncio
from typing import List
import numpy as np
import aiohttp
import zmq, zmq.asyncio
import argparse
import gc
import os
from my_project.tests.utils import TestUtils

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name or path")
parser.add_argument("--batch_size", type=int, default=60) # 71 is max @ 128 PL & 128 GL
parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="dataset name or path")
parser.add_argument("--seq_len", type=int, default=256,  help="sequence length")
parser.add_argument("--prompt_len", type=int, default=128,  help="prompt length") # DEFAULT: 512
parser.add_argument("--gen_len", type=int, default=128,  help="number of tokens to generate") # DEFAULT: 32
parser.add_argument("--cache_dir", type=str, default="/scratch/kumichae", help="cache dir for model name")
parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
args = parser.parse_args()
gc.collect()

async def output_listener(server_socket):
    done = False
    while True:
        try:
            output = await asyncio.wait_for(server_socket.recv_pyobj(), 10)
            if output == "TIMEOUT":
                done = True
                continue
            outputs.append(output)
        except:
            if done:
                return

async def async_server_shareGPT_test(req_rate=0.5):
    TU.print_important(f"START TEST: Async Server ShareGPT Dataset Test | TID: {test_id}")
    reqs, hf_output_ids = TU.dump_hf_results()

    addr = "http://127.0.0.1:8002/"
    context = zmq.asyncio.Context()
    server_socket = context.socket(zmq.PULL)
    server_socket.connect(f"tcp://127.0.0.1:8003")
    tasks: List[asyncio.Task] = []
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(addr, connector=connector) as session:
        tasks.append(asyncio.create_task(output_listener(server_socket)))
        for req in reqs:
        #     for i,seq in enumerate(req):
        #         req[i] = seq.replace("\n", " ")
        #     req[29] = "Hello I am not russian"
            batch_size = len(req)
            gen_lens = [args.gen_len]*batch_size
            params = {"reqs": req, "gen_lens": gen_lens}
            task = asyncio.create_task(session.get("/queue", params=params))
            tasks.append(task)

            if req_rate == float("inf"):
                continue

            interval = np.random.exponential(1.0 / req_rate)
            await asyncio.sleep(interval)

        TU.print_important(f"\nWAITING FOR shareGPT TEST {test_id} TO FINISH")
        await asyncio.gather(*tasks)
    
    TU.print_important(f"FINISHED TEST: ShareGPT Dataset Test | TID: {test_id}")
    TU.validate_shareGPT_outputs(test_id, outputs, hf_output_ids, reqs)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    TU = TestUtils(seed=42, batch_size=60, n_reqs=10)
    outputs = []
    test_id = 0
    res = asyncio.run(async_server_shareGPT_test(req_rate=0.2))
    TU.print_important("ALL TESTS DONE")