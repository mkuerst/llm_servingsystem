import threading
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from transformers.utils import ModelOutput
from transformers.generation.stopping_criteria import (
    StoppingCriteria
)
import numpy as np

# TODO
@dataclass
class ReqStream():
    NotImplemented

# TODO With new cache design, cur_lens is prbly not needed anymore
@dataclass
class BatchMeta(ModelOutput):
    prompt_lens: Optional[torch.LongTensor] = None
    gen_lens: Optional[torch.LongTensor] = None
    cur_lens: Optional[torch.LongTensor] = None
    ids: Optional[torch.Tensor] = None
    req_cache_idxs: Optional[torch.Tensor] = None
    timestamps: Optional[torch.Tensor] = None

    def add_new_batch_meta(self, new_batch_meta):
        self.prompt_lens = torch.cat(tensors=(self.prompt_lens, \
            new_batch_meta.prompt_lens), dim=0)
        self.gen_lens = torch.cat(tensors=(self.gen_lens, \
            new_batch_meta.gen_lens), dim=0)
        self.cur_lens = torch.cat(tensors=(self.cur_lens, \
            new_batch_meta.cur_lens), dim=0)
        self.ids = torch.cat(tensors=(self.ids, \
            new_batch_meta.ids), dim=0)
        self.req_cache_idxs = torch.cat(tensors=(self.req_cache_idxs, \
            new_batch_meta.req_cache_idxs), dim=0)
        self.timestamps = torch.cat(tensors=(self.timestamps, \
            new_batch_meta.timestamps), dim=0)
        return self

# TODO Make stopping_criteria part of batch_meta
class FineInferStoppingCriteria(StoppingCriteria):
    def __init__(
            self,
            max_len: Union[int, List[int], torch.Tensor],
            eos_token_id: Union[int, List[int], torch.Tensor],
            device="cpu",
            batch_size: Optional[int] = 1,
    ):
        if not isinstance(max_len, torch.Tensor):
            if isinstance(max_len, int):
                max_len = [max_len] * batch_size
            max_len = torch.tensor(max_len, device=device)
        self.max_lens = max_len

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id] * batch_size
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_ids = eos_token_id

    def __call__(
            self,
            input_ids: torch.LongTensor,
            batch_meta: BatchMeta,
            scores: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> torch.BoolTensor:
        is_done_max_len = batch_meta.cur_lens >= self.max_lens
        is_done_eos = input_ids[:, -1] == self.eos_token_ids
        is_done = is_done_max_len | is_done_eos
        return is_done

    def add_new_criteria(self, new_criteria):
        self.max_lens = torch.cat((self.max_lens, new_criteria.max_lens))
        self.eos_token_ids = torch.cat((self.eos_token_ids, new_criteria.eos_token_ids))
        
        return self

class ReqIdGenerator:
    def __init__(self, start=0):
        # Should stay at 0 for test w/o chatting reqs
        self.current_id = start
        self.lock = threading.Lock()

    def generate_id(self, batch_size):
        with self.lock:
            id = self.current_id
            self.current_id += 1

        ids = torch.IntTensor([(id, ind) for ind in range(batch_size)])
        return ids

# TODO FOR later cache design needed
# class InferReq:
#     # All nopad
#     def __init__(self, b_id, r_id, prompt_len, gen_len):
#         self.b_id = b_id
#         self.r_id = r_id
#         self.prompt_len = prompt_len
#         self.gen_len = gen_len
#         self.finished = False
#         self.cur_kv_len = 0
#         self.token_ids = None

# class InferBatch:
#     def __init__(self, b_id, token_ids: List[int], gen_lens: List[int]):
#         self.b_id = b_id
#         for i, req in enumerate(token_ids):
#             id = (b_id, i)
#             gen_len = gen_lens[i]
#             prompt_len = len(req)
#             req = InferReq(b_id, i, prompt_len, gen_len)



def prepare_prefill_inputs(seqs, batch_meta, pad_token_id=128001, is_multimodal=False):
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for seq, idx in zip(seqs, batch_meta.req_cache_idxs):

        # batch_multimodal_params.append(req.multimodal_params)
        seq = seq[seq != pad_token_id]
        nopad_b_req_idx.append(idx)
        nopad_b_start_loc.append(start_loc)

        seq_len = len(seq)
        input_token_len = seq_len - 0#req.cur_kv_len

        input_id = seq#req.input_token_ids[req.cur_kv_len :]

        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)
        b_ready_cache_len.append(0)
        start_loc += input_token_len

    input_ids = np.concatenate(input_ids, dtype=np.int64)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")
    kwargs = {
        "batch_size": len(seqs),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "b_ready_cache_len": b_ready_cache_len,
        "is_prefill": True,
    }
    if is_multimodal:
        kwargs["multimodal_params"] = batch_multimodal_params

    return kwargs

def prepare_decode_inputs(seqs, batch_meta, pad_token_id=128001):
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for seq, idx in zip(seqs, batch_meta.req_cache_idxs):
        seq = seq[seq != pad_token_id]
        nopad_b_req_idx.append(idx)
        nopad_b_start_loc.append(start_loc)
        input_id = seq[-1]
        seq_len = len(seq)
        # assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    kwargs = {
        "batch_size": len(seqs),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "is_prefill": False,
    }

    return kwargs

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token