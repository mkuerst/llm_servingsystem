import asyncio
import torch
import numpy as np

import my_project.utils.logger_cfg as logger_cfg
logger = logger_cfg.get_logger()

MB = 1024 ** 2
GB = 1024 ** 3

class SimpleReqManager():
    def __init__(
            self, 
            max_tokens, 
            # config,
            prompt_len=512, 
            gen_len=32, 
            dtype=torch.float16, 
            num_heads=1, 
            head_dim=512, 
            num_layers=32, 
        ):
        # self.config = config
        self.max_seq_length = prompt_len + gen_len
        self.max_tokens = max_tokens
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.waiting = asyncio.PriorityQueue()
        self.reserved_mem = 0
        self.free_mem = max_tokens
        self.cur_pad_len = 0
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype

    
    def write_tokens_state(self, inputs, new_inputs):
        n_tokens = self.n_tokens(new_inputs)
        cur_tokens = self.n_tokens(inputs)

        info = "-"*60+"\n"
        info += f"FUTURE N_TOKENS NEEDED FOR:\n{inputs["batch_meta"].ids.tolist()}\n"
        info += f"N_TOKENS: {n_tokens}\n" 
        info += f"CURRENT TOKEN STATE:\n{cur_tokens}/{self.max_tokens}\n"
        info += "-"*60
        return info

    def update_cur_pad_len(self, input_ids):
        cur_len = input_ids.shape[1]
        self.cur_pad_len = cur_len if cur_len else self.prompt_len

    def n_tokens(self, inputs):
        input_ids = inputs.get("input_ids")
        batch_size = input_ids.shape[0]
        if not batch_size:
            return 0

        batch_meta = inputs["batch_meta"]
        generation_limit = batch_meta.prompt_lens+ batch_meta.gen_lens
        max_mem_requirement = input_ids.shape[0] * input_ids.shape[1] + (sum(generation_limit-batch_meta.cur_lens))
        return max_mem_requirement

    def get_fitting_batches(self, inputs):
        fitting_batches = []
        max_mem_requirement = 0
        # free_mem = self.max_tokens - self.n_tokens(inputs)
        free_mem = 10000 - self.n_tokens(inputs)

        try:
            while True:
                if self.waiting.empty():
                    break
                
                # TODO GEN LENS OF BATCH META OF POTENTIAL CANDIDATE NEEDED
                batch_id, inputs = self.waiting.get_nowait()
                batch_size = inputs["input_ids"].shape[0]
                max_mem_requirement += (self.cur_pad_len + self.gen_len) * batch_size
                if free_mem >= max_mem_requirement:
                    fitting_batches.append(inputs)
                    logger.info(f"adding batch {batch_id} w/ {batch_size} reqs")
                    self.waiting.task_done()
                else:
                    self.waiting.put_nowait((batch_id, inputs))
                    break
        except Exception as e:
            logger.error(f"Error when fetching batches from waiting queue: {e}")

        return fitting_batches