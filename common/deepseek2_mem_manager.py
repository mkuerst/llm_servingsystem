import torch

from .mem_manager import MemoryManager


class Deepseek2MemoryManager(MemoryManager):
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = [torch.empty((size, head_num, head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)]
