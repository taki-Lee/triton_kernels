from email.mime import base
import triton
import triton.language as tl
import torch
from slora._kernels import dispatch_bgmv
from utils import load_bgmv_inputs
# @triton.jit
# def _kernel():
#     pass

# def matmul_bgmv(input_embs, base_layer_infer, base_layer_weight, 
#                 y, x, w,
#                 start_indicies,
#                 lora_ranks,
#                 loc_indicies,
#                 indicies,
#                 qkvo,
#                 lora_scales):
#     # matmul


#     # bgmv shrink
#     # bgmv expand

#     pass

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)

@triton.jit
def _matmul_bgmv_kernel(o_ptr, x_ptr, base_w_ptr,
                        M, N, K,
                        stride_om, stride_on,
                        stride_xm, stride_xk,
                        stride_base_wk, stride_base_wn,
                        BLOCK_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr,
                        GROUP_SIZE_M: tl.constexpr,
                        ):
    # matmul
    # per block per program(pid)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m) 
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_base_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    base_w_ptrs = base_w_ptr + (offs_k[:, None] * stride_base_wk + offs_base_wn[None, :] * stride_base_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(base_w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator += tl.dot(a,b)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        base_w_ptrs += BLOCK_SIZE_K * stride_base_wk
    
    o = accumulator.to(tl.float16)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_ptrs = o_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    o_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(o_ptrs, o, mask=o_mask)


    # bgmv shrink


    # bgmv expand
    
    


@triton.jit
def _bgmv_kernel(o, x, w,
                 start_indicies, lora_ranks, loc_indicies, indicies, lora_scales):
    
    pass



def matmul_bgmv(o, x, base_w, ):
    '''
    '''
    assert x.shape[1] == base_w.shape[0]
    assert x.is_contiguous()
    assert base_w.is_contiguous()
    M, K = x.shape
    K, N = base_w.shape
    
    # c = torch.empty((M,N), device=a.device, dtype = a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    _matmul_bgmv_kernel[grid](
        o, x, base_w,
        M, N, K,
        o.stride(0), o.stride(1),
        x.stride(0), x.stride(1),
        base_w.stride(0), base_w.stride(1),
    )

    pass

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_bgmv(c, a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def test_op():
    torch.manual_seed(0)
    a = torch.randn((512,512), device='cuda', dtype = torch.float16)
    b = torch.randn((512,512), device='cuda', dtype = torch.float16)
    c = torch.empty((a.shape[0], b.shape[1]), device = 'cuda', dtype = torch.float16)

    (x , key_buffer, a_start, a_len, a_loc, a_scaling, req_bins) = load_bgmv_inputs(path='/workspace/S-LoRA/benchmarks/rand_datas/kernel_inputs_1711981690.json')
    
    matmul_bgmv(c,a,b)
    triton_output = c 
    torch_output = torch.matmul(a,b)

    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("Triton and torch match")
    else:
        print("Triton and Torch differ")

# benchmark.run(save_path='./results/', show_plots=True, print_data=True)
test_op()
        
