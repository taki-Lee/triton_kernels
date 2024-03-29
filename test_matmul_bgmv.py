import triton
import triton.language as tl
import torch

@triton.jit
def _kernel():
    pass

def matmul_bgmv(input_embs, base_layer_infer, base_layer_weight, 
                y, x, w,
                start_indicies,
                lora_ranks,
                loc_indicies,
                indicies,
                qkvo,
                lora_scales):
    
    pass
