# EXAMPLES
# ["capital of france", "capital of france"] [32, 32]
# ["capital of france", "capital of france"] [128, 128]

#-------------- TODO ------------------------------------
#--------------------------------------------------------

# CUDA_VISIBLE_DEVICES=3 python -m my_project.server.cli
import aiohttp
import asyncio
import re
from typing import List, Dict
import sys
import zmq
import zmq.asyncio
import torch

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import  my_project.utils.logger_cfg as logger_cfg
logger = logger_cfg.get_logger()

class CLI_Client():
    
    def __init__(self, server_addr):
        self.server_addr = server_addr
        self.pending: Dict[int: List[int]] = {} 
        self.dec_seqs: Dict[int: List[str]] = {}
        self.loops: List[asyncio.Task] = []
        self.session: aiohttp.ClientSession = None
        self.start_polling: asyncio.Event = asyncio.Event()
        self.chat_mode = False
        self.chat_id = []
        self.chat_response = asyncio.Event()

    @property
    def reqs_in_progress(self) -> bool:
        return (len(self.pending.keys()))

    async def ainput(self, string: str) -> str:
        await asyncio.to_thread(sys.stdout.write, f'{string}')
        await asyncio.to_thread(sys.stdout.flush)
        return (await asyncio.to_thread(sys.stdin.readline)).rstrip('\n')

    def prep_reqs(self, input):
        res = [item for item in re.split(r'[\[\]]', input) if item.strip()]
        len_res = len(res)
        assert len_res >= 1 and len_res <= 2, "Make sure to correctly format your request submission.\nE.g. ['Prompt 1', 'Prompt 2', ...] [2, 32, ...]"
        prompts = res[0]

        prompts = [item for item in re.split(r"[\"\"]", prompts) if item.strip(", ")]
        gen_lens = [] 
        if len_res == 2:
            gen_lens = [item.strip(' ') for item in res[1].split(",")]
            assert len(prompts) == len(gen_lens), \
            f"Please provide one generation length per request.\nActual #gen_lens: {len(gen_lens)}. Expected #gen_lens: {len(prompts)}."
            assert all([x.isdigit() for x in gen_lens]), \
                f"Please provide ints for generation lenghs"
            gen_lens = [int(x) for x in gen_lens]
            assert all([x >= 2 for x in gen_lens]), "All generation lengths need to be bigger than 2."

        return prompts, gen_lens 

    def print_responses(self, bids):
            delete = []
            for bid in bids:
                unfinished_sequences = self.pending[bid]
                if max(unfinished_sequences) == 0:
                    if self.chat_mode and bid == self.chat_id[0]:
                        self.chat_id = [] 
                        batch_msg = f"\nFINISHED CHAT {bid}"
                        print(CURSOR_UP_ONE+ERASE_LINE+f"{batch_msg}\n{self.dec_seqs[bid]}")
                        self.chat_response.set()
                    else:
                        batch_msg = f"\nFINISHED BATCH {bid}" 
                        print(CURSOR_UP_ONE+ERASE_LINE+f"{batch_msg}\n{self.dec_seqs[bid]}")

                    delete.append(bid)

            for bid in delete:
                del self.pending[bid]
                del self.dec_seqs[bid]

    # TODO Handle errors here aswell?
    def detangle_responses(self, outputs):
            for dec_seq, (bid, rid) in zip(outputs[0], torch.Tensor.tolist(outputs[1])):
                self.pending[bid][rid] = 0
                self.dec_seqs[bid][rid] = dec_seq

    async def output_listener(self):
        while True:
            outputs = await self.server_socket.recv_pyobj()
            self.detangle_responses(outputs)
            ids = self.chat_id if self.chat_mode else self.pending.keys()
            self.print_responses(ids)
            if not self.chat_mode:
                sys.stdout.write(f'<BATCH> ')
                sys.stdout.flush()

    async def make_req(self, session, user_input, is_chat_req=0):
        user_input = self.prep_reqs(user_input)
        params = {"reqs": user_input[0], "gen_lens": user_input[1], "is_chat_req": is_chat_req}
        res = await session.get("/queue", params=params)
        res = await res.json()

        bid = res["batch_id"]
        unfinished_seqs = res["unfinished_sequences"]
        self.pending[bid] = unfinished_seqs
        self.dec_seqs[bid] = [None] * len(unfinished_seqs)

        if self.chat_mode:
            self.chat_id = [bid]

    async def main_loop(self, session):
        while True:
            user_input = await self.ainput("<BATCH> ")
            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Exiting...")
                for loop in self.loops:
                    loop.cancel()
                return

            if user_input.lower() == "chat":
                self.chat_mode = True
                print("Entering chat-mode")
                while True:
                    user_input = input("<CHAT> ")
                    if not user_input:
                        continue

                    if user_input.lower() == "exit":
                        print("Exiting chat-mode...")
                        self.chat_mode = False
                        self.print_responses(self.pending.keys())
                        break

                    await self.make_req(session, user_input, 1)
                    await self.chat_response.wait()
                    self.chat_response.clear()

            else:
                await self.make_req(session, user_input)

    async def run_cli(self, session):
        try:
            res = await session.get("/")
            print(f"Server ping response: {await res.text()}")
        except:
            print("Server is not up!")
            return 
        if res.status != 200:
            print(f"Trying to reach server led to error: {res.reason}")
            return

        self.context = zmq.asyncio.Context(2)
        self.server_socket = self.context.socket(zmq.PULL)
        self.server_socket.connect(f"tcp://127.0.0.1:8003")

        task1 = asyncio.create_task(self.output_listener())
        task2 = asyncio.create_task(self.main_loop(session))
        self.loops.append(task1)
        self.loops.append(task2)
        await asyncio.gather(*self.loops)


async def main():
    client = CLI_Client(server_addr)
    try:
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(server_addr, connector=connector) as session:
            client.session = session
            await client.run_cli(session)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(e, exc_info=True)
    await client.session.close()
        

if __name__ == '__main__':
    server_addr = "http://127.0.0.1:8002/"
    user_addr = "127.0.0.2:8002"
    asyncio.run(main())