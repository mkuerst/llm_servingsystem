import sys
import numpy as np
import time
import asyncio
from typing import List, Optional 
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from my_project.models.llama.model import LlamaTpPartModel
from my_project.engine.simple_req_manager import SimpleReqManager
from .utils import (
    BatchMeta,
    ReqIdGenerator,
    FineInferStoppingCriteria,
    prepare_prefill_inputs,
    prepare_decode_inputs,
    sample_top_p,
)

from line_profiler import profile

import zmq
import zmq.asyncio

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import my_project.utils.logger_cfg as logger_cfg
logger = logger_cfg.get_logger()

# ---------------------------TODO---------------------------
# Check what methods really need to be async
# ----------------------------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class AsyncLLMEngine():
    def __init__(
            self,
            model_name: str = "meta-llama/Meta-Llama-3-8B",
            pin_memory: int = 0,
            quant_bits: int = 16,
            prompt_len: int = 1024,
            gen_len: int = 1024,
            cache_dir: str = "/scratch/kumichae",
            model_dir: str = "/scratch/kumichae/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6",
            max_total_token_num = 52000, #65000 | 52000
            max_seq_length = 2048,
            model = None,
            model_kvargs = None,
            flash_attention = True,
    ):
        # CONFIG for benchmark showing model memory consumption
        if quant_bits == 4:
            raise NotImplementedError()
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        pin_memory = bool(pin_memory)
        dtype = torch.float16
        self.max_seq_length = max_seq_length
        self.device = torch.cuda.current_device()
        self.max_total_token_num = max_total_token_num

        if model is None:
            max_req_num = max_total_token_num // max_seq_length
            max_req_num = 1000    
            model_kvargs = {
                "tp_rank": 0,
                "world_size": 1,
                "weight_dir": model_dir,
                # "weight_dict": weight_dict,
                "max_total_token_num": max_total_token_num,
                "max_new_tokens": gen_len,
                "load_way": "HF",
                "mode": [],
                "max_req_num": max_req_num,
                "max_seq_length": max_seq_length,
                "is_token_healing": False,
                "return_all_prompt_logics": False,
                "use_dynamic_prompt_cache": False,
                "data_type": "float16",
                "flash_attention": flash_attention,
                # "config": config,
            } if model_kvargs is None else model_kvargs

            print("Preparing Llama Model w/ Triton Kernel")
            self.model = LlamaTpPartModel(model_kvargs) 
        else:
            self.model = model

        self.prompt_len: int = prompt_len
        # TODO remove?
        self.gen_len: int = gen_len

        self.finished = asyncio.Queue()
        self.outputs = []
        self.id_generator: ReqIdGenerator = ReqIdGenerator()
        self.new_reqs_event = asyncio.Event()

        # TODO For testing
        self.steps: int = 0
        self.test_done = asyncio.Event()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.bs = []
        self.token_nums = []

        self.running_batch = self.infer_batch()

        self.context = zmq.asyncio.Context(2)
        self.server_socket = self.context.socket(zmq.PAIR)
        self.server_socket.bind("tcp://127.0.0.1:8001")

        self.recv_loop: Optional[asyncio.Future] = None
        self.engine_loop: Optional[asyncio.Future] = None

    @property
    def recv_is_running(self) -> bool:
        return (self.recv_loop is not None
                and not self.recv_loop.done())

    @property
    def engine_is_running(self) -> bool:
        return (self.engine_loop is not None
                and not self.engine_loop.done())

    async def start_loops(self, start_recv=True):
        if self.recv_is_running:
            raise RuntimeError("recv loop is already running.")
        if self.engine_is_running:
            raise RuntimeError("engine loop is already running.")

        self.engine_loop = asyncio.create_task(self.start_engine())
        if start_recv:
            self.recv_loop = asyncio.create_task(self.start_recv_loop())
            await asyncio.gather(self.engine_loop, self.recv_loop)
        else:
            await asyncio.gather(self.engine_loop)

        
    # TODO simplify (currently needed for tests)
    async def wait_for_new_reqs(self):
            while True:
                logger.debug("Waiting for new requests")
                try:
                    if self.model.req_manager.waiting.empty():
                        await asyncio.wait_for(self.new_reqs_event.wait(), 10)
                        self.new_reqs_event.clear()
                    else:
                        break
                except asyncio.TimeoutError:
                    self.server_socket.send_pyobj("TIMEOUT")

    def infer_batch(
            self,
            input_ids = torch.empty(0,0), 
            unfinished_sequences=None,
            batch_meta=None, 
            stopping_criteria=None
        ):
        return {
            "input_ids": input_ids,
            "unfinished_sequences": unfinished_sequences,
            "batch_meta": batch_meta,
            "stopping_criteria": stopping_criteria
        }

    async def put(self,
            prompts: List[str],
            gen_lens: Optional[torch.LongTensor]=None, 
            is_chat_req=False, 
            ids: torch.IntTensor=None, 
            unfinished_sequences: Optional[torch.LongTensor]=None,
            timestamp = None,
        ):
        if timestamp is None:
            timestamp = time.time()
        input_tokens = self.tokenizer.batch_encode_plus(prompts, return_tensors="pt",
            padding=True, max_length=self.prompt_len, truncation=True) #padding="max_length"

        input_ids = input_tokens.pop("input_ids")
        batch_size = input_ids.shape[0]

        if unfinished_sequences is None:
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device="cpu")

        if gen_lens is None:
            gen_lens = torch.full(size=(batch_size,), fill_value=self.gen_len,
            dtype=torch.long, device="cpu")

        prompt_lens = torch.tensor([len(x[x != self.tokenizer.pad_token_id]) for x in input_ids], dtype=torch.int, device="cpu")

        max_lens = torch.clamp((gen_lens + prompt_lens), max=self.max_seq_length)
        stopping_criteria = FineInferStoppingCriteria(
            max_len=max_lens,
            eos_token_id=self.tokenizer.eos_token_id,
            device="cpu",
            batch_size=batch_size
        )
        timestamps = torch.zeros((batch_size, 4), dtype=torch.float64)
        timestamps[:,0] = timestamp
        batch_meta = BatchMeta(
            prompt_lens = prompt_lens,
            gen_lens = gen_lens,
            cur_lens = prompt_lens.clone(),
            ids = ids if ids is not None else self.id_generator.generate_id(batch_size=batch_size),
            req_cache_idxs = torch.zeros(size=(batch_size,),
                dtype=torch.int, device="cpu"),
            timestamps=timestamps
        )

        inputs = self.infer_batch(input_ids, unfinished_sequences, batch_meta, stopping_criteria)
        bid = batch_meta.ids[0][0].item()
        prio = 0 if is_chat_req else bid
        self.model.req_manager.waiting.put_nowait((prio, inputs))
        logger.info(f"QUEUED REQS @step({self.steps}) w/ IDs:\n{torch.Tensor.tolist(batch_meta.ids)}")
        self.steps = 0
        
        self.new_reqs_event.set()
        # TODO
        self.test_done.clear()
        return bid, unfinished_sequences 

    def add_reqs(self, inputs, new_inputs):
        input_ids = inputs["input_ids"]

        if len(input_ids) == 0:
            return new_inputs

        unfinished_sequences = torch.cat((inputs["unfinished_sequences"], new_inputs["unfinished_sequences"]), dim=0)
        new_shape = max(input_ids.shape[1], new_inputs["input_ids"].shape[1])
        old_input_ids = torch.full(
            (input_ids.shape[0], new_shape),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long, 
            device="cpu"
        )
        old_input_ids[:, -input_ids.shape[1]:] = input_ids
        cur_input_ids = torch.full(
            (new_inputs["input_ids"].shape[0], new_shape),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long, 
            device="cpu"
        )
        cur_input_ids[:, -new_inputs["input_ids"].shape[1]:] = new_inputs["input_ids"]
        input_ids = torch.cat((old_input_ids, cur_input_ids), dim=0)


        batch_meta = inputs["batch_meta"].add_new_batch_meta(new_inputs["batch_meta"])
        stopping_criteria = inputs["stopping_criteria"].add_new_criteria(new_inputs["stopping_criteria"])

        return self.infer_batch(input_ids, unfinished_sequences, batch_meta, stopping_criteria)

    def remove_reqs(self):
        batch_meta = self.running_batch["batch_meta"]
        masks = self.running_batch["unfinished_sequences"].to(torch.bool)

        end = time.time()
        batch_meta.timestamps[~masks,0] = end - batch_meta.timestamps[~masks, 0]
        # TODO For testing only
        input_ids = self.running_batch["input_ids"]
        outputs = (input_ids[~masks].to("cpu"), 
                   batch_meta.ids[~masks].to("cpu"), 
                   batch_meta.prompt_lens[~masks].to("cpu"), 
                   batch_meta.cur_lens[~masks].to("cpu"),
                   batch_meta.timestamps[~masks].to("cpu"),
                   None,
                   )
        self.outputs.append(outputs)

        unfinished_sequences = self.running_batch["unfinished_sequences"][masks]
        batch_meta.prompt_lens = batch_meta.prompt_lens[masks]
        finished_gen_lens = batch_meta.gen_lens[~masks]
        batch_meta.gen_lens = batch_meta.gen_lens[masks]
        batch_meta.cur_lens = batch_meta.cur_lens[masks]
        finished_ids = batch_meta.ids[~masks]
        batch_meta.ids = batch_meta.ids[masks]
        finished_req_cache_idxs = batch_meta.req_cache_idxs[~masks]
        batch_meta.req_cache_idxs = batch_meta.req_cache_idxs[masks]
        batch_meta.timestamps = batch_meta.timestamps[masks]
        self.model.req_manager.free(finished_req_cache_idxs, input_ids[~masks], outputs[2])

        stopping_criteria = self.running_batch["stopping_criteria"]
        stopping_criteria.max_lens = self.running_batch["stopping_criteria"].max_lens[masks]
        stopping_criteria.eos_token_ids = self.running_batch["stopping_criteria"].eos_token_ids[masks]

        logging_msg = "FINISHED REQUESTS:\n"
        out = [(id, gen_len, cur_len) for id, gen_len, cur_len in zip(finished_ids, finished_gen_lens, outputs[3])]
        for id, gen_len, cur_len in out:
            logging_msg += f"\n(BATCH, REQ, GEN_LEN, CUR_LEN): ({id[0]}, {id[1]}, {gen_len}, {cur_len})"
        logger.debug(logging_msg)

        # self.server_socket.send_pyobj(outputs)

        del outputs
        input_ids = input_ids[masks]
        if not len(input_ids):
            input_ids = torch.empty(0,0)

        # Needed to free the allocated physical memory on the GPU
        torch.cuda.empty_cache()
        return self.infer_batch(input_ids, unfinished_sequences, batch_meta, stopping_criteria)

    @profile
    async def async_step(self, input_ids=[], unfinished_sequences=[], batch_meta=[], stopping_criteria=[], is_prefill=True):
        start = time.time()
        temperature = 0
        top_p = 0.9 
        token_num = 0
        for in_ids in input_ids:
            token_num += len(in_ids[in_ids != self.tokenizer.pad_token_id])
        self.token_nums.append(token_num)
        self.bs.append(len(input_ids))

        if is_prefill:
            kwargs = prepare_prefill_inputs(input_ids, batch_meta, self.tokenizer.pad_token_id)
        else:
            kwargs = prepare_decode_inputs(input_ids, batch_meta, self.tokenizer.pad_token_id)
        
        with torch.no_grad():
            s2 = time.time()
            logits, elapsed_time = self.model.forward(**kwargs)
            e2 = time.time() - s2
            
        batch_meta.cur_lens += 1
        batch_meta.timestamps[:,1] += elapsed_time
        batch_meta.timestamps[:,3] += e2
        if temperature > 0:
            probs = torch.softmax(logits[:,-1] / temperature, dim=-1)
            next_tokens = sample_top_p(probs, top_p)
        else:
            next_tokens = torch.argmax(logits, dim=-1)

        has_eos_stopping_criteria = hasattr(stopping_criteria, "eos_token_ids")
        next_tokens = next_tokens.detach().cpu()
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + stopping_criteria.eos_token_ids * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1).cpu()

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, batch_meta)
        # new_tokens = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
        self.steps += 1
        del next_tokens, logits 
        batch_meta.timestamps[:,2] += time.time() - start
        return self.infer_batch(input_ids, unfinished_sequences, batch_meta, stopping_criteria)

    def get_new_reqs(self):
        reqs = self.infer_batch()
        fitting_batches = self.model.req_manager.get_fitting_batches(self.running_batch)
        for new_batch in fitting_batches:
            reqs = self.add_reqs(reqs, new_batch)
        return reqs

    @profile
    async def engine_step(self) -> bool:
        new_reqs = self.get_new_reqs()
        if len(new_reqs["input_ids"]):
            new_reqs = await self.async_step(**new_reqs, is_prefill=True)
            self.running_batch = self.add_reqs(self.running_batch, new_reqs)

        if len(self.running_batch["input_ids"]):
            self.running_batch = await self.async_step(**self.running_batch, is_prefill=False)

        has_finished_sequences = any(torch.isin(self.running_batch["unfinished_sequences"], 0))
        if has_finished_sequences:
            self.running_batch = self.remove_reqs()

        reqs_in_progress = self.running_batch["batch_meta"].cur_lens.shape[0] != 0

        ### NOTE: FOR TESTS ###
        if not reqs_in_progress and self.model.req_manager.waiting.empty():
            self.test_done.set()
        ##############################

        return reqs_in_progress

    async def start_recv_loop(self):
        while True:
            reqs, gen_lens, is_chat_req, bid, unfinished_sequences = await self.server_socket.recv_pyobj()
            await self.put(reqs, gen_lens, is_chat_req, bid, unfinished_sequences)

    async def start_engine(self):
        reqs_in_progress = False
        try:
            while True:
                if not reqs_in_progress:
                    await self.wait_for_new_reqs()
                reqs_in_progress = await self.engine_step()
                # self.model.req_manager.log_mem_usage(self.running_batch)

                if reqs_in_progress:
                    self.new_reqs_event.set()
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("engine loops canceled")
            self.engine_loop = None
            self.recv_loop = None
            self.server_socket.send_pyobj("DONE")
            return

        except Exception as e:
            last_info = ""
            if len(self.outputs):
                o_ids, _, _, _, _, _ = self.outputs[-1]
                lens = [len(x) for x in o_ids]
                num_tokens = sum(lens)
                batch_size = o_ids.shape[0]
                last_info += f"Last output: {batch_size} batch_size w/ lens:\n{lens}\n"
                last_info += f"Total of {num_tokens} tokens.\n"

            if len(self.running_batch["input_ids"]):
                batch_meta = self.running_batch["batch_meta"]
                self.outputs.append((self.running_batch['input_ids'], self.running_batch['batch_meta'].ids, None, None, None, e))
                last_info += f"RUNNING BATCH CUR_LENS:\n{batch_meta.cur_lens.tolist()}\n"
                last_info += f"RUNNING BATCH RESERVED TOKENS:\n{self.max_total_token_num - self.model.req_manager.can_use_mem_size / {self.max_total_token_num}}\n"
            logger.error('Error during async_llm_engine.engine_loop(): %s\n%s', e, last_info, exc_info=True)
            del self.running_batch
            self.running_batch = self.infer_batch()
            sys.exit(1)

    
def start_engine_process(prompt_len, gen_len, max_tokens, pipe_writer):
    engine = AsyncLLMEngine(prompt_len=prompt_len, gen_len=gen_len, max_tokens=max_tokens)
    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(engine.start_loops())
    # asyncio.run(engine.start_loops())