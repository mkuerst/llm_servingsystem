# CUDA_VISIBLE_DEVICES=3 uvicorn  my_project.server.server:app --host 127.0.0.1 --port 8002 --workers 0
# CUDA_VISIBLE_DEVICES=3 python -m my_project.server.server
import asyncio
import uvicorn
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import argparse
from contextlib import asynccontextmanager
from my_project.engine.async_llm_engine import (
    start_engine_process
)
import multiprocessing as mp
import zmq
import zmq.asyncio
import torch
from my_project.server.utils import (
    start_processes,
    ReqIdGenerator,
)


# lsof -i :<port>
# kiil -9 PID
# ------------------TODOs ------------------
# TODO Implement permanent connections with aiohttp websockets?  -- Maybe later for multiple users?
# TODO Define BaseModel output type?
# ------------------------------------------


# import logging
# logging.basicConfig(level=logging.DEBUG)

import my_project.utils.logger_cfg as logger_cfg
logger = logger_cfg.get_logger()

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# TODO Remove id reset
async def output_listener():
    while True:
        outputs = await app.engine_socket.recv_pyobj()
        # if outputs == "TIMEOUT":
        #     app.id_generator.current_id = 0
        app.client_socket.send_pyobj(outputs)

@asynccontextmanager
async def lifespan(app: FastAPI, gen_len = 32):
    '''
    Run at startup to load the model and tokenizer
    '''
    app.context = zmq.asyncio.Context(2)
    app.engine_socket = app.context.socket(zmq.PAIR)
    app.engine_socket.connect(f"tcp://127.0.0.1:8001")
    app.client_socket = app.context.socket(zmq.PUSH)
    app.client_socket.bind(f"tcp://127.0.0.1:8003")
    app.gen_len = gen_len
    app.id_generator = ReqIdGenerator()
    asyncio.create_task(output_listener())
    yield


app = FastAPI(
    description="A fastapi server handling batch inference and finetuning requests",
    title="FineInfer FastAPI-Server",
    lifespan=lifespan,
)
server = uvicorn.Server(uvicorn.Config(app))

# class User(BaseModel):
#     """
#     Representation of a User
#     """
#     id: int = Field(description="Unique integer that specifies the user.")
#     requests: List[str] = Field(description="List of unfinished requests of the user.")
#     adaptors: None = Field(description="PEFT Params for the user.")

@app.get("/")
def ping_response() -> str:
    return "OK"

@app.get("/queue")
async def queue_req(
    reqs: Optional[List[str]] = Query(None, description="Reqeust parameter", examples=["['Prompt 1', 'Prompt 2', ...]"]),
    gen_lens: Optional[List[int]] = Query([], description="Generation Lengths parameter", examples=["[2, 32, ...]"]),
    is_chat_req: Optional[int] = Query(0, description="Is Chat Request parameter", examples=['1', '0'])
):
    '''
    Adds the batch of requests to the waiting queue of the engine.

    Args:
    req: List[str] of prompts. E.g. ["Prompt 1", "Prompt 2", ...]
    gen_lens: List[int] of gen_lens. E.g. [12, 13, 32, ...]

    Returns:
    batch_id and unfinished sequences of the submitted batch
    '''

    if not reqs: 
       return "No requests provided to add to the queue."

    try:
        gen_lens = torch.LongTensor(gen_lens) if len(gen_lens) \
        else torch.LongTensor([app.gen_len]*len(reqs))
        bid, unfinished_sequences = app.id_generator.generate_ids(len(reqs))
        await app.engine_socket.send_pyobj((reqs, gen_lens, bool(is_chat_req), bid, unfinished_sequences))
        return {"batch_id": bid[0][0].item(), "unfinished_sequences": torch.Tensor.tolist(unfinished_sequences)}
    except Exception as e:
        logger.exception(e, exc_info=True)
        return f"Failed at inserting {reqs} | {torch.Tensor.tolist(gen_lens)} into queue with: {e}"

# @app.post("/register_user")
# async def register_user(address: str) -> str:
#     # Requires Authentication Middleware
#     # data = await req.json()
#     # user_ip = req.user.host  # Get user's IP address
#     # user_port = data['port']  # Get user's listening port from the request data
#     user_id = 0    

#     session = aiohttp.ClientSession(f"http://{address}")
#     app.users[user_id] = {"address": address, "session": session}
#     await session.post(f"/callback/{user_id}", json={"response": f"Processed data for client {user_id}: TEST_ONLY"})
#     return f"Registered user {0} w/ {address}"
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name or path")
    parser.add_argument("--prompt_len", type=int, default=1024,  help="prompt length")
    parser.add_argument("--gen_len", type=int, default=1024,  help="number of tokens to generate")
    parser.add_argument("--cache_dir", type=str, default="/scratch/kumichae", help="cache dir for model name")
    parser.add_argument("--max_tokens", type=str, default=9500, help="max tokens that can be processed simultaneously")
    args = parser.parse_args()

    start_processes([start_engine_process], [(args.prompt_len, args.gen_len, args.max_tokens)])
    uvicorn.run("my_project.server.server:app", host=args.host, port=args.port, reload=False, loop="uvloop")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
