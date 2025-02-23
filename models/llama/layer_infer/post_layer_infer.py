import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from my_project.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from my_project.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo

from my_project.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from einops import rearrange
from my_project.models.llama.infer_struct import LlamaInferStateInfo
from my_project.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from my_project.common.basemodel import PostLayerInferTpl


class LlamaPostLayerInfer(PostLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        return

    def _norm(self, input, infer_state, layer_weight: LlamaPreAndPostLayerWeight) -> torch.Tensor:
        return rmsnorm_forward(input, layer_weight.final_norm_weight_, eps=self.eps_)

    def _slice_get_last_input(self, input_embdings, infer_state: LlamaInferStateInfo):
        if infer_state.is_splitfuse:
            # for SplitFuse
            batch_size = infer_state.batch_size
            last_input = torch.empty(
                (batch_size, self.embed_dim_), device=input_embdings.device, dtype=input_embdings.dtype
            )
            tmp_ = torch.cat(
                [
                    torch.ones(infer_state.decode_req_num, dtype=torch.int32, device="cuda"),
                    infer_state.prefill_b_seq_len - infer_state.prefill_b_split_ready_cache_len,
                ],
                dim=0,
            )
            last_index = torch.cumsum(tmp_, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if infer_state.is_prefill and infer_state.is_token_healing:
            batch_size = infer_state.batch_size
            b_seq_len_numpy = (infer_state.b_seq_len - infer_state.b_ready_cache_len).detach().cpu().numpy()
            select_index = []
            start_index = 0
            select_token_num = 0
            for cur_len in b_seq_len_numpy:
                if cur_len == 1:
                    select_index.append(start_index + cur_len - 1)
                    start_index += cur_len
                    select_token_num += 1
                else:
                    select_index.append(start_index + cur_len - 2)
                    select_index.append(start_index + cur_len - 1)
                    start_index += cur_len
                    select_token_num += 2

            last_index = torch.tensor(select_index, dtype=torch.long, device=input_embdings.device)
            last_input = torch.empty(
                (select_token_num, self.embed_dim_), device=input_embdings.device, dtype=input_embdings.dtype
            )

            last_input[:, :] = input_embdings[last_index, :]
            return last_input, select_token_num

        if not infer_state.is_splitfuse and infer_state.is_prefill and not infer_state.return_all_prompt_logics:
            batch_size = infer_state.batch_size
            last_input = torch.empty(
                (batch_size, self.embed_dim_), device=input_embdings.device, dtype=input_embdings.dtype
            )
            last_index = (
                torch.cumsum(infer_state.b_seq_len - infer_state.b_ready_cache_len, dim=0, dtype=torch.long) - 1
            )
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if not infer_state.is_splitfuse and infer_state.is_prefill and infer_state.return_all_prompt_logics:
            total_tokens = infer_state.total_token_num
            return input_embdings, total_tokens

        if not infer_state.is_splitfuse and not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[-batch_size:, :], batch_size

        assert False, "Error State"

    def token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)

        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty(
                (self.vocab_size_, token_num), device=logic_batch.device, dtype=input_embdings_dtype
            )
            split_indexes = np.linspace(0, self.vocab_size_, self.world_size_ + 1, dtype=np.int64)
            dist.all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.world_size_)],
                logic_batch,
                group=None,
                async_op=False,
            )
        logic_batch = None

        ans_logics = gather_data.permute(1, 0).float()
        gather_data = None
        return ans_logics

    def splitfuse_forward(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight: BaseLayerWeight):
        return self.token_forward(input_embdings, infer_state, layer_weight)
