# CUDA_VISIBLE_DEVICES=3 python -m my_project.tests.server_cli_overhead
import asyncio
import argparse
import gc
import my_project.tests.bench_utils as utils
import torch
from typing import List
import time
import numpy as np
import aiohttp
import zmq
from transformers import (
    AutoTokenizer,
)
from my_project.server.cli import CLI_Client


import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
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
parser.add_argument("--trials", type=int, default=50,  help="Number of token generation iterations") #50
parser.add_argument("--prompt_len", type=int, default=256,  help="prompt length") # DEFAULT: 512
parser.add_argument("--gen_len", type=int, default=128,  help="number of tokens to generate") # DEFAULT: 32
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

async def run_server_lat_bench(bs=args.batch_size, gen_len=args.gen_len, trials=args.trials):
    addr = "http://127.0.0.1:8002/"
    context = zmq.asyncio.Context()
    server_socket = context.socket(zmq.PULL)
    server_socket.connect(f"tcp://127.0.0.1:8003")

    req = paris_long
    total_timings = []
    forward_timings = []
    output_ids = []
    prompts = [req]*bs
    gen_lens = [gen_len]*bs
    warmup_gen_lens = [512]*bs
    params = {"reqs": prompts, "gen_lens": gen_lens}
    warmup_params = {"reqs": prompts, "gen_lens": warmup_gen_lens}

    connector = aiohttp.TCPConnector(force_close=True)
    # WARMUP
    async with aiohttp.ClientSession(addr, connector=connector) as session:
        for _ in range(3):
            _ = await session.get("/queue", params=warmup_params)
            output = await server_socket.recv_pyobj()
            # out = tokenizer.batch_decode(output[0], skip_special_tokens=True)
            # print(out[0])

        for _ in range(trials):
            start = time.time()
            _ = await session.get("/queue", params=params)
            outputs = await server_socket.recv_pyobj()
            end = time.time()
            output_ids.append(outputs[0])
            forward_timings.append(outputs[4])
            total_timings.append(end-start)

    for out in output_ids:
        assert all([len(x)==args.prompt_len+args.gen_len for x in out]), "LLM_ENGINE SERVER BENCH: ouput length mismatch"
    log_summary(total_timings=total_timings, forward_timings=forward_timings, name="Server")

async def run_cli_lat_bench(bs=args.batch_size, gen_len=args.gen_len, trials=args.trials):
    server_addr = "http://127.0.0.1:8002/"
    client = CLI_Client(server_addr)
    connector = aiohttp.TCPConnector(force_close=True)
    total_timings = []
    forward_timings = []
    output_ids = []

    async with aiohttp.ClientSession(server_addr, connector=connector) as session:
        client.session = session
        await client.run_lat_bench(session, bs=bs, gen_len=gen_len, trials=trials)
    await client.session.close()
    for outputs in client.test_outputs:
            output_ids.append(outputs[0])
            forward_timings.append(outputs[4])
    total_timings = client.total_timings
    log_summary(total_timings=total_timings, forward_timings=forward_timings, name="CLI")

        

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left', cache_dir=args.cache_dir)
    # res = asyncio.run(run_server_lat_bench())
    res = asyncio.run(run_cli_lat_bench())