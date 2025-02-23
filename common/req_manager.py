import torch
from my_project.utils.log_utils import init_logger
import asyncio

logger = init_logger(__name__)

MB = 1024 ** 2
GB = 1024 ** 3
    
class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, max_new_tokens, mem_manager, pad_token_id=128001):
        self.req_state = torch.zeros((max_request_num,), dtype=torch.bool, device="cuda")
        self.req_to_token_indexs = torch.zeros((max_request_num, max_sequence_length), dtype=torch.int32, device="cuda")
        self.can_use_req_size = max_request_num
        self.mem_manager = mem_manager
        self.waiting = asyncio.PriorityQueue()
        self.max_sequence_length = max_sequence_length
        self.max_new_tokens = max_new_tokens
        self.pad_token_id = pad_token_id
        self.can_use_mem_size = mem_manager.can_use_mem_size
        self.max_total_token_num = mem_manager.can_use_mem_size

    def get_fitting_batches(self, inputs):
        fitting_batches = []
        try:
            while True:
                if self.waiting.empty():
                    break
                # TODO GEN LENS OF BATCH META OF POTENTIAL CANDIDATE NEEDED
                batch_id, inputs = self.waiting.get_nowait()
                batch_size = inputs["input_ids"].shape[0]
                select_index = self.alloc(inputs)
                if select_index is not None:
                    inputs["batch_meta"].req_cache_idxs = select_index
                    fitting_batches.append(inputs)
                    logger.info(f"adding batch {batch_id} w/ {batch_size} reqs\nReserved Tokens: {self.max_total_token_num - self.can_use_mem_size} / {self.max_total_token_num}")
                    self.waiting.task_done()
                else:
                    self.waiting.put_nowait((batch_id, inputs))
                    break

        except Exception as e:
            logger.error(f"Error when fetching batches from waiting queue: {e}")
        return fitting_batches

    def alloc(self, inputs):
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        if batch_size > self.can_use_req_size:
            # logger.error(f'Insufficient request capacity, remaining {self.can_use_req_size} and requested {batch_size}')
            return None
        batch_meta = inputs["batch_meta"]
        need_size = (self.max_new_tokens + batch_meta.prompt_lens).sum()
        if need_size > self.can_use_mem_size:
            # logger.error(f'Insufficient capacity on KV Cache, remaining {self.mem_manager.can_use_mem_size} and requested {need_size}')
            return None
        select_index = torch.nonzero(self.req_state==0).reshape(-1)[:batch_size]
        self.req_state[select_index] = 1
        self.can_use_req_size -= len(select_index)
        self.can_use_mem_size -= need_size
        return select_index
    
    def free(self, free_req_index, finished_seqs, prompt_lens):
        self.can_use_req_size += len(free_req_index)
        self.req_state[free_req_index] = 0
        if self.can_use_req_size == len(self.req_state):
            logger.debug(f"freed all: Current available request spots {self.can_use_req_size}")
        free_token_index = []
        for req_idx, seq in zip(free_req_index, finished_seqs):
            seq = seq[seq != self.pad_token_id]
            cur_kv_len = len(seq) - 1 
            free_token_index.append(self.req_to_token_indexs[req_idx][: cur_kv_len])
        free_token_index = torch.cat(free_token_index, dim=-1)
        num_tokens = prompt_lens.sum()
        self.can_use_mem_size += (1024*len(finished_seqs) + num_tokens).item()
        self.mem_manager.free(free_token_index, num_tokens)

    
    def free_req(self, free_req_index):
        self.can_use_req_size +=1
        self.req_state[free_req_index] = 0
        return
    
    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)

    def free_all(self):
        self.can_use_req_size = len(self.req_state)
        self.req_state[:] = 0

    def log_mem_usage(self, inputs, memory_info="", print_gpu_mem=False):
        if print_gpu_mem:
            device = torch.cuda.current_device()
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory = torch.cuda.memory_reserved(device)
            memory_info += f"Allocated memory: {allocated_memory / (MB):.3f} MB\nReserved memory: {reserved_memory / (MB):.3f} MB\n"
            gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))
            memory_info += f"Peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
            memory_info += "-"*60+"\n"

        items = list(self.waiting._queue)
        memory_info += "Current lens of waiting reqs:\n"
        waiting_info = ""
        for item in items:
            waiting_info += f"{item[1]["batch_meta"].cur_lens}\n"
        memory_info += waiting_info if waiting_info else "No reqs are in the waiting queue\n"
        memory_info += "-"*60+"\n"
        
        in_progress = "["
        for input_id in inputs["input_ids"]:
            in_progress += f"{len(input_id)}, "
        in_progress = in_progress[:-2] + "]"
        memory_info += f"Current lens of work in progress reqs:\n"
        memory_info += f"{in_progress}" if len(inputs["input_ids"]) \
            else "None are being processed"

        logger.info(memory_info)