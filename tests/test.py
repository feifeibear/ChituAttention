import torch
import torch.nn.functional as F
from chitu.sageattention.core import sageattn
import time
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_kvpacked_func
from flash_attn.flash_attn_interface import _flash_attn_forward
from chitu.interface import _sage_attn_forward, _int8_flash_attn_forward
from chitu.int8_flash_attention.flash_atten_fp import attention as int8_fa_ref_attn
import os


def test_sage_vs_flash():
    # Set up test parameters
    batch_size = 1
    num_heads = 1
    seq_len = 1024
    head_dim = 64

    # current_cuda_device = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(current_cuda_device)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    dtype = torch.float16
    # Generate random input tensors
    # q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
    # k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
    # v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)

    # initialize q, k, v with normal distribution
    q = torch.empty(
        (batch_size, seq_len, num_heads, head_dim), dtype=dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    k = torch.empty(
        (batch_size, seq_len, num_heads, head_dim), dtype=dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    v = torch.empty(
        (batch_size, seq_len, num_heads, head_dim), dtype=dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)

    test_attention(q, k, v, is_causal=False)


def test_attention(q, k, v, is_causal):

    softmax_scale = q.shape[-1] ** (-0.5)
    int8_out = _int8_flash_attn_forward(q, k, v, is_causal, softmax_scale=softmax_scale)

    int8_out_ref = int8_fa_ref_attn(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal,
        softmax_scale,
    )
    int8_out_ref = int8_out_ref.transpose(1, 2)

    print("OUT DIFF (int8 vs int8_ref):")
    print(f"Max absolute difference: {torch.max(torch.abs(int8_out - int8_out_ref))}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(int8_out - int8_out_ref))}")

    sage_out, lse = _sage_attn_forward(q, k, v, causal=is_causal, ret_lse=True)

    block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p=0,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )

    if lse is not None:
        print("LSE DIFF (sage vs FA):")
        print(f"Max absolute difference: {torch.max(torch.abs(lse - block_lse))}")
        print(f"Mean absolute difference: {torch.mean(torch.abs(lse - block_lse))}")
        relative_error = torch.abs(lse - block_lse) / (torch.abs(block_lse) + 1e-6)
        print(f"Max relative difference: {torch.max(relative_error)}")
        print(f"Mean relative difference: {torch.mean(relative_error)}")

    print("OUT DIFF (FA vs int8_ref):")
    print(f"Max absolute difference: {torch.max(torch.abs(block_out - int8_out_ref))}")
    print(
        f"Mean absolute difference: {torch.mean(torch.abs(block_out - int8_out_ref))}"
    )

    print("OUT DIFF (int8 vs FA):")
    print(f"Max absolute difference: {torch.max(torch.abs(int8_out - block_out))}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(int8_out - block_out))}")

    print("OUT DIFF (sage vs FA):")
    print(f"Max absolute difference: {torch.max(torch.abs(sage_out - block_out))}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(sage_out - block_out))}")


if __name__ == "__main__":
    test_sage_vs_flash()
