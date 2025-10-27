import os
import triton
import triton.language as tl
import torch
import math
# from .quant import pseudo_quantization
import matplotlib.pyplot as plt

@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EQUAL_D": lambda args: args["D"] == args["BLOCK_d"],
    }
)
@triton.jit
def _quantized_fused_attention(
    q_ptr, k_ptr, v_ptr,
    o_ptr,
    # attn_ptr,
    # softmax_input_ptr,
    max_diff_ptr,
    scale_factor,
    # exp_sum_ptr, tmp_max_ptr,
    B, H, N, M, D,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_m, stride_k_d,
    stride_v_b, stride_v_h, stride_v_m, stride_v_d,
    stride_o_b, stride_o_h, stride_o_n, stride_o_d,
    stride_max_diff_b, stride_max_diff_h, stride_max_diff_n, stride_max_diff_bm,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_d: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EQUAL_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITWIDTH: tl.constexpr,
    BLOCK_NUM_BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_id = pid_bh // H
    head_id = pid_bh % H

    q_ptr = q_ptr + batch_id * stride_q_b + head_id * stride_q_h
    k_ptr = k_ptr + batch_id * stride_k_b + head_id * stride_k_h
    v_ptr = v_ptr + batch_id * stride_v_b + head_id * stride_v_h
    o_ptr = o_ptr + batch_id * stride_o_b + head_id * stride_o_h
    max_diff_ptr = max_diff_ptr + batch_id * stride_max_diff_b + head_id * stride_max_diff_h
    # attn_ptr = attn_ptr + batch_id * M * N * H + head_id * N * M

    offs_q_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_d)

    q_ptrs = q_ptr + (offs_q_n[:, None] * stride_q_n + offs_d[None, :] * stride_q_d)
    k_ptrs = k_ptr + (offs_m[:, None] * stride_k_m + offs_d[None, :] * stride_k_d)
    v_ptrs = v_ptr + (offs_m[:, None] * stride_v_m + offs_d[None, :] * stride_v_d)
    o_ptrs = o_ptr + (offs_q_n[:, None] * stride_o_n + offs_d[None, :] * stride_o_d)
    max_diff_ptrs = max_diff_ptr + (offs_q_n[:, None] * stride_max_diff_n)
    # attn_ptrs = attn_ptr + (offs_q_n[:, None] * M + offs_m[None, :])

    # use heuristics to avoid out of bound memory access
    if EVEN_N & EVEN_M:
        if EQUAL_D:
            query = tl.load(q_ptrs)
        else:
            query = tl.load(q_ptrs, mask = (offs_d[None, :] < D), other = 0.0)
    else:
        if EQUAL_D:
            query = tl.load(q_ptrs, mask = (offs_q_n[:, None] < N), other = 0.0)
        else:
            query = tl.load(q_ptrs, mask = (offs_q_n[:, None] < N) & (offs_d[None, :] < D), other = 0.0)
    
    l_i = tl.zeros([BLOCK_N], dtype = tl.float32)
    m_i = tl.zeros([BLOCK_N], dtype = tl.float32) - float("inf")
    part_sum = tl.zeros([BLOCK_N, BLOCK_d], dtype = tl.float32)

    for m in range(0, M, BLOCK_M):
        if EVEN_N & EVEN_M:
            if EQUAL_D:
                key = tl.load(k_ptrs)
                value = tl.load(v_ptrs)
            else:
                key = tl.load(k_ptrs, mask = (offs_d[None, :] < D), other = 0.0)
                value = tl.load(v_ptrs, mask = (offs_d[None, :] < D), other = 0.0)
        else:
            if EQUAL_D:
                key = tl.load(k_ptrs, mask = (offs_m[:, None] < M - m), other = 0.0)
                value = tl.load(v_ptrs, mask = (offs_m[:, None] < M - m), other = 0.0)
            else:
                key = tl.load(k_ptrs, mask = (offs_m[:, None] < M - m) & (offs_d[None, :] < D), other = 0.0)
                value = tl.load(v_ptrs, mask = (offs_m[:, None] < M - m) & (offs_d[None, :] < D), other = 0.0)

        # S_j = tl.dot(query, key, trans_b = True)
        key_T = tl.trans(key)
        S_j = tl.dot(query, key_T)

        # Mask out the out of range values to make softmax correct
        S_j = S_j * scale_factor
        S_j += tl.where(offs_m[None, :] < M - m, 0, float("-inf"))

        # P_mask = (offs_q_n[:, None] < N) & (offs_m[None, :] < M - m)

        m_ij = tl.maximum(m_i, tl.max(S_j, axis = -1))

        P_j = tl.exp(S_j - m_ij[:, None])
        # P_j = _fast_exp_triton(S_j - m_ij[:, None]) 

        # tl.store(attn_ptrs, P_j, mask = P_mask)

        # P_j = _pseudo_quantization(
        #     P_j,
        #     BLOCK_N,
        #     BLOCK_M,
        #     GROUP_SIZE,
        #     BITWIDTH,
        # )

        l_ij = tl.sum(P_j, axis = -1)

        scale = tl.exp(m_i - m_ij)
        # scale = _fast_exp_triton(m_i - m_ij)

        tl.store(max_diff_ptrs, (m_i - m_ij)[:, None])

        part_sum = part_sum * scale[:, None]
        P_j = P_j.to(value.dtype)

        # tl.store(attn_ptrs, P_j / l_ij[:, None], mask = P_mask)

        part_sum += tl.dot(P_j, value)

        # Update the statistics
        m_i = tl.maximum(m_i, m_ij)
        l_i *= scale
        l_i += l_ij

        k_ptrs += BLOCK_M * stride_k_m
        v_ptrs += BLOCK_M * stride_v_m
        max_diff_ptrs += stride_max_diff_bm
        # attn_ptrs += BLOCK_M
    
    l_i = tl.where(l_i > 0, l_i, 1.0)
    o_scale = 1 / l_i
    part_sum = part_sum * o_scale[:, None]

    if EVEN_N & EVEN_M:
        if EQUAL_D:
            tl.store(o_ptrs, part_sum)
        else:
            tl.store(o_ptrs, part_sum, mask = (offs_d[None, :] < D))
    else:
        if EQUAL_D:
            tl.store(o_ptrs, part_sum, mask = (offs_q_n[:, None] < N))
        else:
            tl.store(o_ptrs, part_sum, mask = (offs_q_n[:, None] < N) & (offs_d[None, :] < D))
@triton.jit
def _quantized_fused_attention_bf16(
    q_ptr, k_ptr, v_ptr,
    o_ptr,
    # attn_ptr,
    # softmax_input_ptr,
    # max_diff_ptr,
    scale_factor,
    # exp_sum_ptr, tmp_max_ptr,
    B, H, N, M, D,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_m, stride_k_d,
    stride_v_b, stride_v_h, stride_v_m, stride_v_d,
    stride_o_b, stride_o_h, stride_o_n, stride_o_d,
    # stride_max_diff_b, stride_max_diff_h, stride_max_diff_n, stride_max_diff_bm,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_d: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EQUAL_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITWIDTH: tl.constexpr,
    BLOCK_NUM_BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_id = pid_bh // H
    head_id = pid_bh % H

    q_ptr = q_ptr + batch_id * stride_q_b + head_id * stride_q_h
    k_ptr = k_ptr + batch_id * stride_k_b + head_id * stride_k_h
    v_ptr = v_ptr + batch_id * stride_v_b + head_id * stride_v_h
    o_ptr = o_ptr + batch_id * stride_o_b + head_id * stride_o_h
    # max_diff_ptr = max_diff_ptr + batch_id * stride_max_diff_b + head_id * stride_max_diff_h
    # attn_ptr = attn_ptr + batch_id * M * N * H + head_id * N * M

    offs_q_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_d)

    q_ptrs = q_ptr + (offs_q_n[:, None] * stride_q_n + offs_d[None, :] * stride_q_d)
    k_ptrs = k_ptr + (offs_m[:, None] * stride_k_m + offs_d[None, :] * stride_k_d)
    v_ptrs = v_ptr + (offs_m[:, None] * stride_v_m + offs_d[None, :] * stride_v_d)
    o_ptrs = o_ptr + (offs_q_n[:, None] * stride_o_n + offs_d[None, :] * stride_o_d)
    # max_diff_ptrs = max_diff_ptr + (offs_q_n[:, None] * stride_max_diff_n)
    # attn_ptrs = attn_ptr + (offs_q_n[:, None] * M + offs_m[None, :])

    # use heuristics to avoid out of bound memory access
    if EVEN_N & EVEN_M:
        if EQUAL_D:
            query = tl.load(q_ptrs)
        else:
            query = tl.load(q_ptrs, mask = (offs_d[None, :] < D), other = 0.0)
    else:
        if EQUAL_D:
            query = tl.load(q_ptrs, mask = (offs_q_n[:, None] < N), other = 0.0)
        else:
            query = tl.load(q_ptrs, mask = (offs_q_n[:, None] < N) & (offs_d[None, :] < D), other = 0.0)
    
    l_i = tl.zeros([BLOCK_N], dtype = tl.bfloat16)
    m_i = tl.full([BLOCK_N], value = float("-inf"), dtype = tl.bfloat16)
    part_sum = tl.zeros([BLOCK_N, BLOCK_d], dtype = tl.bfloat16)

    for m in range(0, M, BLOCK_M):
        if EVEN_N & EVEN_M:
            if EQUAL_D:
                key = tl.load(k_ptrs)
                value = tl.load(v_ptrs)
            else:
                key = tl.load(k_ptrs, mask = (offs_d[None, :] < D), other = 0.0)
                value = tl.load(v_ptrs, mask = (offs_d[None, :] < D), other = 0.0)
        else:
            if EQUAL_D:
                key = tl.load(k_ptrs, mask = (offs_m[:, None] < M - m), other = 0.0)
                value = tl.load(v_ptrs, mask = (offs_m[:, None] < M - m), other = 0.0)
            else:
                key = tl.load(k_ptrs, mask = (offs_m[:, None] < M - m) & (offs_d[None, :] < D), other = 0.0)
                value = tl.load(v_ptrs, mask = (offs_m[:, None] < M - m) & (offs_d[None, :] < D), other = 0.0)

        # S_j = tl.dot(query, key, trans_b = True)
        key_T = tl.trans(key)
        S_j = tl.dot(query, key_T).to(tl.bfloat16) # 无法控制内部计算精度为bf16

        # Mask out the out of range values to make softmax correct
        # S_j = S_j * scale_factor
        S_j += tl.where(offs_m[None, :] < M - m, 0, float("-inf"))

        # P_mask = (offs_q_n[:, None] < N) & (offs_m[None, :] < M - m)

        m_ij = tl.maximum(m_i, tl.max(S_j, axis = -1))

        P_j = tl.exp(S_j - m_ij[:, None])
        # P_j = _fast_exp_triton_bf16((S_j - m_ij[:, None]).to(tl.bfloat16))
        # P_j = tl.where(S_j - m_ij[:, None] == 0, 1.0, P_j)

        # tl.store(attn_ptrs, P_j, mask = P_mask)

        # P_j = _pseudo_quantization_bf16(
        #     P_j,
        #     BLOCK_N,
        #     BLOCK_M,
        #     GROUP_SIZE,
        #     BITWIDTH,
        # )

        l_ij = tl.sum(P_j, axis = -1).to(tl.bfloat16) # # 无法控制内部计算精度为bf16

        scale = tl.exp(m_i - m_ij)

        # scale = _fast_exp_triton_bf16((m_i - m_ij).to(tl.bfloat16))
        # scale = tl.where(m_i - m_ij == 0, 1.0, scale)

        # store the max_diff
        # tl.store(max_diff_ptrs, (m_i - m_ij)[:, None])

        part_sum = (part_sum * scale[:, None]).to(tl.bfloat16)
        P_j = P_j.to(value.dtype)

        # tl.store(attn_ptrs, P_j / l_ij[:, None], mask = P_mask)

        part_sum += tl.dot(P_j, value).to(tl.bfloat16) # 无法控制内部计算精度为bf16

        # Update the statistics
        m_i = tl.maximum(m_i, m_ij).to(tl.bfloat16)
        l_i *= scale
        l_i += l_ij
        # l_i = l_i.to(tl.bfloat16) # 实验：将l_i转为bfloat16

        k_ptrs += BLOCK_M * stride_k_m
        v_ptrs += BLOCK_M * stride_v_m
        # max_diff_ptrs += stride_max_diff_bm
        # attn_ptrs += BLOCK_M
    
    l_i = tl.where(l_i > 0, l_i, 1.0)
    o_scale = (1 / l_i).to(tl.bfloat16)
    part_sum = part_sum * o_scale[:, None]

    if EVEN_N & EVEN_M:
        if EQUAL_D:
            tl.store(o_ptrs, part_sum)
        else:
            tl.store(o_ptrs, part_sum, mask = (offs_d[None, :] < D))
    else:
        if EQUAL_D:
            tl.store(o_ptrs, part_sum, mask = (offs_q_n[:, None] < N))
        else:
            tl.store(o_ptrs, part_sum, mask = (offs_q_n[:, None] < N) & (offs_d[None, :] < D))


def quantized_fused_attention(
    query, key, value,
    BLOCK_N: int = 128,
    BLOCK_M: int = 128,
    BLOCK_d: int = 64,
    GROUP_SIZE: int = 16,
    BITWIDTH: int = 6,
):
    b, h, n, d = query.shape
    _, _, m, _ = key.shape

    assert key.shape == (b, h, m, d)
    assert value.shape == (b, h, m, d)
    assert query.dtype == key.dtype == value.dtype, "All tensors must have the same dtype"

    # query, key, value = [x if x.stride(-1) == 1 else x.contiguous() for x in [query, key, value]]

    scale_factor = 1 / math.sqrt(d)

    # n_rounded = math.ceil(n // BLOCK_N) * BLOCK_N
    # exp_sum = torch.empty((b, h, n_rounded), device = query.device, dtype = torch.float32)
    # tmp_max = torch.empty((b, h, n_rounded), device = query.device, dtype = torch.float32)

    hidden_states = torch.empty_like(query)
    # hidden_states_ref = torch.empty_like(query)
    # attn_value = torch.empty(b, h, n, m, device = query.device, dtype = torch.bfloat16)

    # softmax_input = torch.empty((b * h, n, m), dtype=torch.float32, device=query.device)
    # tmp_max_diff = torch.empty((b, h, n, (m + BLOCK_M - 1) // BLOCK_M), dtype=torch.float32, device=query.device)
    # tmp_max_diff_ref = torch.empty((b, h, n, (m + BLOCK_M - 1) // BLOCK_M), dtype=torch.float32, device=query.device)

    func = _quantized_fused_attention_bf16
    # func = _quantized_fused_attention

    grid = lambda META: (
        triton.cdiv(n, META["BLOCK_N"]),
        b * h,
    )
    func[grid](
        query, key, value,
        hidden_states,
        # softmax_input,
        # tmp_max_diff,
        # attn_value,
        # exp_sum, tmp_max,
        scale_factor,
        b, h, n, m, d,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        hidden_states.stride(3),
        # tmp_max_diff.stride(0),
        # tmp_max_diff.stride(1),
        # tmp_max_diff.stride(2),
        # tmp_max_diff.stride(3),
        BLOCK_N = BLOCK_N,
        BLOCK_M = BLOCK_M,
        BLOCK_d = BLOCK_d,
        GROUP_SIZE = GROUP_SIZE,
        BITWIDTH = BITWIDTH,
        BLOCK_NUM_BLOCK_M = (m + BLOCK_M - 1) // BLOCK_M,
    )
    
    return hidden_states

def quant_attn_score_triton(
    query, key, value,
    args,
    BLOCK_N: int = 128,
    BLOCK_M: int = 128,
    BLOCK_d: int = 64,
):
    query, key, value = [x if x.stride(-1) == 1 else x.contiguous() for x in [query, key, value]]

    BITWIDTH = args.score_bitwidth
    GROUP_SIZE = args.group_size
    output_hidden_states = quantized_fused_attention(
        query, key, value,
        BLOCK_N = 64,
        BLOCK_M = 128,
        BLOCK_d = 64,
        GROUP_SIZE = GROUP_SIZE,
        BITWIDTH = BITWIDTH,
    )

    # print (output_hidden_states)

    if torch.isnan(output_hidden_states).any():
        raise ValueError("NaN in output_hidden_states after pseudo quantization!")

    return output_hidden_states