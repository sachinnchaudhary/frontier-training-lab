import os

os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import triton   
import torch  
import triton.language as tl 
  

"""  

def deepseek_moe_layer(x, W_router, routed_experts, shared_experts, top_k):
    # x:              [T, D]
    # W_router:       [D, E]
    # routed_experts: E expert MLPs
    # shared_experts: S expert MLPs
    # top_k:          usually 2
    #
    # output:         [T, D]

    T, D = x.shape
    E = number_of_routed_experts
    S = number_of_shared_experts

    # ------------------------------------------------------------
    # Step 1: shared experts always run
    # ------------------------------------------------------------
    shared_out = zeros([T, D])

    for s in range(S):
        shared_out += shared_experts[s](x)     # [T, D]

    shared_out = shared_out / S                # optional averaging


    # ------------------------------------------------------------
    # Step 2: router scores for routed experts
    # ------------------------------------------------------------
    router_logits = x @ W_router               # [T, E]


    # ------------------------------------------------------------
    # Step 3: choose top-k routed experts per token
    # ------------------------------------------------------------
    top_values, top_indices = topk(
        router_logits,
        k=top_k,
        dim=-1,
    )
    # top_values:  [T, top_k]
    # top_indices: [T, top_k]


    # ------------------------------------------------------------
    # Step 4: normalize only selected expert weights
    # ------------------------------------------------------------
    router_weights = softmax(top_values, dim=-1)
    # router_weights: [T, top_k]
    #
    # For every token:
    # sum(router_weights[token, :]) == 1


    # ------------------------------------------------------------
    # Step 5: routed experts run only for selected tokens
    # ------------------------------------------------------------
    routed_out = zeros([T, D])

    for expert_id in range(E):

        # token_selected[t, slot] says:
        # "did token t choose this expert in this top-k slot?"
        token_selected = top_indices == expert_id
        # [T, top_k]

        # token_mask says:
        # "did token t choose this expert at all?"
        token_mask = any(token_selected, dim=-1)
        # [T]

        if no token_mask is true:
            continue

        # Gather tokens assigned to this expert.
        tokens_for_expert = x[token_mask]
        # [T_e, D]

        # Run this expert only on its assigned tokens.
        expert_out = routed_experts[expert_id](tokens_for_expert)
        # [T_e, D]

        # Get this expert's router weight for each selected token.
        selected_slots = token_selected[token_mask].float()
        # [T_e, top_k]

        expert_weight = sum(
            router_weights[token_mask] * selected_slots,
            dim=-1,
        )
        # [T_e]

        # Add weighted expert output back to token positions.
        routed_out[token_mask] += expert_weight[:, None] * expert_out


    # ------------------------------------------------------------
    # Step 6: final MoE output
    # ------------------------------------------------------------
    output = shared_out + routed_out
    # [T, D]

    return output  



"""



def torch_deepseek_moe_combine_ref(  
router_logits,  
expert_outputs, 
shared_out, 
top_k,
):  
    

    # router_logits:  [N, E]
    # expert_outputs: [N, E, D]
    # shared_out:     [N, D]  

    assert router_logits.ndim == 2 
    assert expert_outputs.ndim == 3 
    assert shared_out.ndim == 2  

    N, E = router_logits.shape 
    N2, E2, D = expert_outputs.shape 

    assert N2 == N   
    assert E2 == E  
    assert shared_out.shape == (N, D)  

    assert top_k <= E  

    top_values, top_indices = torch.topk(router_logits, k=top_k, dim=-1)  
    router_weights = torch.softmax(top_values, dim=-1)  

    batch_idx = torch.arange(N, device=router_logits.device)[:, None]  
    selected_outputs = expert_outputs[batch_idx, top_indices]  
    #[N, K, D]  

    routed_out = torch.sum(router_weights[...,None] * selected_outputs, dim=1)  
    out = shared_out + routed_out  

    return out, top_indices, router_weights  


def torch_deepseek_moe_combine_backward_ref(
    router_logits,
    expert_outputs,
    shared_out,
    dout,
    top_k,
):
    """
    Reference backward through torch_deepseek_moe_combine_ref.

    returns:
      drouter_logits:  [N, E]
      dexpert_outputs: [N, E, D]
      dshared_out:     [N, D]
    """

    router_logits = router_logits.detach().clone().requires_grad_(True)
    expert_outputs = expert_outputs.detach().clone().requires_grad_(True)
    shared_out = shared_out.detach().clone().requires_grad_(True)

    out, _, _ = torch_deepseek_moe_combine_ref(
        router_logits,
        expert_outputs,
        shared_out,
        top_k,
    )
    loss = (out * dout).sum()
    loss.backward()

    return router_logits.grad, expert_outputs.grad, shared_out.grad



@triton.jit  
def _deepseek_moe_combine_kernel(
  
  router_logits_ptr,   # [N, E]
  expert_outputs_ptr,  # [N, E, D]
  shared_out_ptr,      # [N, D]
  out_ptr,             # [N, D]
  top_indices_ptr,     # [N, K]
  router_weights_ptr,  # [N, K] 
  
  E: tl.constexpr, 
  D: tl.constexpr, 
  K: tl.constexpr, 
  BLOCK_E: tl.constexpr, 
  BLOCK_D: tl.constexpr,  
 
): 
    
    pid_n = tl.program_id(0)  
    pid_d = tl.program_id(1)  

    offs_e = tl.arange(0, BLOCK_E)  
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  

    mask_e = offs_e < E 
    mask_d = offs_d < D  
    
    logits = tl.load(  
        router_logits_ptr + pid_n * E + offs_e, 
        mask=mask_e, 
        other= float("-inf") 
    )

    top1_val = tl.max(logits, axis=0)  
    top1_idx = tl.argmax(logits, axis=0)  

    logits_without_top1 = tl.where(offs_e == top1_idx, float("-inf"), logits)
    
    top2_val = tl.max(logits_without_top1, axis=0) 
    top2_idx = tl.argmax(logits_without_top1, axis=0) 
    
    m = tl.maximum(top1_val, top2_val)  

    w1 = tl.exp(top1_val - m)  
    w2 = tl.exp(top2_val - m)  

    denom = w1 + w2  

    w1 = w1 / denom 
    w2 = w2 / denom  

    tl.store(top_indices_ptr+ pid_n * K + 0, top1_idx) 
    tl.store(top_indices_ptr + pid_n * K + 1, top2_idx) 

    tl.store(router_weights_ptr + pid_n * K + 0, w1)  
    tl.store(router_weights_ptr + pid_n * K + 1, w2)  

    acc = tl.load(
        shared_out_ptr + pid_n * D + offs_d,  
        mask= mask_d, 
        other=0.0,  
    )
    
    expert1 = tl.load(  
     expert_outputs_ptr + pid_n * E * D + top1_idx * D + offs_d, 
     mask=mask_d, 
     other=0.0   
    )

    expert2 = tl.load(
        expert_outputs_ptr + pid_n * E * D + top2_idx * D + offs_d, 
        mask = mask_d,  
        other=0.0
    )

    acc += w1 * expert1 
    acc += w2 * expert2  

    tl.store(  
        out_ptr + pid_n * D + offs_d, 
        acc, 
        mask=mask_d  
    )


@triton.jit
def _deepseek_moe_combine_backward_vec_kernel(
    dout_ptr,             # [N, D]
    top_indices_ptr,      # [N, K]
    router_weights_ptr,   # [N, K]
    dexpert_outputs_ptr,  # [N, E, D]
    dshared_out_ptr,      # [N, D]
    E: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Vector gradients:
      dshared_out[n,d] = dout[n,d]
      dexpert_outputs[n,e_k,d] = router_weights[n,k] * dout[n,d]
    """

    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    dout = tl.load(
        dout_ptr + pid_n * D + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        dshared_out_ptr + pid_n * D + offs_d,
        dout,
        mask=mask_d,
    )

    for slot in range(K):
        expert_id = tl.load(top_indices_ptr + pid_n * K + slot)
        weight = tl.load(router_weights_ptr + pid_n * K + slot).to(tl.float32)

        tl.store(
            dexpert_outputs_ptr + pid_n * E * D + expert_id * D + offs_d,
            weight * dout,
            mask=mask_d,
        )


@triton.jit
def _deepseek_moe_combine_backward_router_kernel(
    dout_ptr,             # [N, D]
    expert_outputs_ptr,   # [N, E, D]
    top_indices_ptr,      # [N, K]
    router_weights_ptr,   # [N, K]
    drouter_logits_ptr,   # [N, E]
    E: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Router-gradient path:
      dweight[k] = sum_d dout[n,d] * expert_outputs[n,e_k,d]
      dtop_value = softmax_backward(router_weights, dweight)
      drouter_logits[n,e_k] = dtop_value[k]
    """

    pid_n = tl.program_id(0)

    idx0 = tl.load(top_indices_ptr + pid_n * K + 0)
    idx1 = tl.load(top_indices_ptr + pid_n * K + 1)

    w0 = tl.load(router_weights_ptr + pid_n * K + 0).to(tl.float32)
    w1 = tl.load(router_weights_ptr + pid_n * K + 1).to(tl.float32)

    dw0_vec = tl.zeros((BLOCK_D,), tl.float32)
    dw1_vec = tl.zeros((BLOCK_D,), tl.float32)

    for start in range(0, D, BLOCK_D):
        offs_d = start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        dout = tl.load(
            dout_ptr + pid_n * D + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        expert0 = tl.load(
            expert_outputs_ptr + pid_n * E * D + idx0 * D + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        expert1 = tl.load(
            expert_outputs_ptr + pid_n * E * D + idx1 * D + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        dw0_vec += dout * expert0
        dw1_vec += dout * expert1

    dw0 = tl.sum(dw0_vec, axis=0)
    dw1 = tl.sum(dw1_vec, axis=0)

    dot = dw0 * w0 + dw1 * w1
    dlogit0 = w0 * (dw0 - dot)
    dlogit1 = w1 * (dw1 - dot)

    tl.store(drouter_logits_ptr + pid_n * E + idx0, dlogit0)
    tl.store(drouter_logits_ptr + pid_n * E + idx1, dlogit1)


def triton_deepseek_moe_combine(router_logits, expert_outputs, shared_out, top_k=2, block_d=128):  

    assert top_k == 2 
    assert router_logits.is_cuda
    assert expert_outputs.is_cuda
    assert shared_out.is_cuda
    assert router_logits.ndim == 2
    assert expert_outputs.ndim == 3
    assert shared_out.ndim == 2

    N, E = router_logits.shape 
    N2, E2, D = expert_outputs.shape 

    assert N2 == N 
    assert E2 == E  
    assert shared_out.shape == (N, D)  

    router_logits = router_logits.contiguous()
    expert_outputs = expert_outputs.contiguous()  
    shared_out = shared_out.contiguous()  

    out = torch.empty_like(shared_out)  
    top_indices = torch.empty((N, top_k), device=router_logits.device, dtype=torch.int64)  
    router_weights = torch.empty((N, top_k), device=router_logits.device, dtype=router_logits.dtype)  

    block_e = triton.next_power_of_2(E)
    grid = (N, triton.cdiv(D, block_d))

    _deepseek_moe_combine_kernel[grid](
        router_logits,
        expert_outputs,
        shared_out,
        out,
        top_indices,
        router_weights,
        E,
        D,
        top_k,
        block_e,
        block_d,
    )

    return out, top_indices, router_weights


def triton_deepseek_moe_combine_backward(
    router_logits,
    expert_outputs,
    shared_out,
    dout,
    top_indices,
    router_weights,
    top_k=2,
    block_d=128,
):
    """
    Backward for triton_deepseek_moe_combine.

    returns:
      drouter_logits:  [N, E]
      dexpert_outputs: [N, E, D]
      dshared_out:     [N, D]
    """

    assert top_k == 2
    assert router_logits.is_cuda
    assert expert_outputs.is_cuda
    assert shared_out.is_cuda
    assert dout.is_cuda
    assert top_indices.is_cuda
    assert router_weights.is_cuda

    assert router_logits.ndim == 2
    assert expert_outputs.ndim == 3
    assert shared_out.ndim == 2
    assert dout.ndim == 2
    assert top_indices.ndim == 2
    assert router_weights.ndim == 2

    N, E = router_logits.shape
    N2, E2, D = expert_outputs.shape

    assert N2 == N
    assert E2 == E
    assert shared_out.shape == (N, D)
    assert dout.shape == (N, D)
    assert top_indices.shape == (N, top_k)
    assert router_weights.shape == (N, top_k)

    router_logits = router_logits.contiguous()
    expert_outputs = expert_outputs.contiguous()
    shared_out = shared_out.contiguous()
    dout = dout.contiguous()
    top_indices = top_indices.contiguous()
    router_weights = router_weights.contiguous()

    drouter_logits = torch.zeros_like(router_logits)
    dexpert_outputs = torch.zeros_like(expert_outputs)
    dshared_out = torch.empty_like(shared_out)

    vec_grid = (N, triton.cdiv(D, block_d))
    _deepseek_moe_combine_backward_vec_kernel[vec_grid](
        dout,
        top_indices,
        router_weights,
        dexpert_outputs,
        dshared_out,
        E,
        D,
        top_k,
        block_d,
    )

    router_grid = (N,)
    _deepseek_moe_combine_backward_router_kernel[router_grid](
        dout,
        expert_outputs,
        top_indices,
        router_weights,
        drouter_logits,
        E,
        D,
        top_k,
        block_d,
    )

    return drouter_logits, dexpert_outputs, dshared_out


class _DeepSeekMoECombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits, expert_outputs, shared_out, top_k, block_d):
        out, top_indices, router_weights = triton_deepseek_moe_combine(
            router_logits,
            expert_outputs,
            shared_out,
            top_k=top_k,
            block_d=block_d,
        )

        ctx.save_for_backward(
            router_logits,
            expert_outputs,
            shared_out,
            top_indices,
            router_weights,
        )
        ctx.top_k = top_k
        ctx.block_d = block_d

        return out

    @staticmethod
    def backward(ctx, dout):
        (
            router_logits,
            expert_outputs,
            shared_out,
            top_indices,
            router_weights,
        ) = ctx.saved_tensors

        drouter_logits, dexpert_outputs, dshared_out = triton_deepseek_moe_combine_backward(
            router_logits,
            expert_outputs,
            shared_out,
            dout,
            top_indices,
            router_weights,
            top_k=ctx.top_k,
            block_d=ctx.block_d,
        )

        return drouter_logits, dexpert_outputs, dshared_out, None, None


def deepseek_moe_combine_autograd(
    router_logits,
    expert_outputs,
    shared_out,
    top_k=2,
    block_d=128,
):
    """
    Autograd-enabled Triton MoE combine.

    router_logits:  [N, E]
    expert_outputs: [N, E, D]
    shared_out:     [N, D]

    returns:
      out: [N, D]
    """

    return _DeepSeekMoECombineFunction.apply(
        router_logits,
        expert_outputs,
        shared_out,
        top_k,
        block_d,
    )


def benchmark(fn, *args, warmup=10, iters=100):
    def consume(out):
        if isinstance(out, tuple):
            return sum(x.float().sum() for x in out).item()
        return out.float().sum().item()

    torch.cuda.empty_cache()

    for _ in range(warmup):
        out = fn(*args)
        consume(out)
        del out
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = fn(*args)
    consume(out)
    del out
    end.record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / iters







if __name__  == "__main__":  

    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.float32

    N = 3
    E = 4
    D = 8
    top_k = 2

    router_logits = torch.randn((N, E), device=device, dtype=dtype)
    expert_outputs = torch.randn((N, E, D), device=device, dtype=dtype)
    shared_out = torch.randn((N, D), device=device, dtype=dtype)

    out, top_indices, router_weights = torch_deepseek_moe_combine_ref(
        router_logits,
        expert_outputs,
        shared_out,
        top_k,
    )
    tri_out, tri_top_indices, tri_router_weights = triton_deepseek_moe_combine(
        router_logits,
        expert_outputs,
        shared_out,
        top_k,
        block_d=4,
    )
    dout = torch.randn_like(out)
    drouter_ref, dexpert_ref, dshared_ref = torch_deepseek_moe_combine_backward_ref(
        router_logits,
        expert_outputs,
        shared_out,
        dout,
        top_k,
    )
    drouter_tri, dexpert_tri, dshared_tri = triton_deepseek_moe_combine_backward(
        router_logits,
        expert_outputs,
        shared_out,
        dout,
        tri_top_indices,
        tri_router_weights,
        top_k,
        block_d=4,
    )

    print("router_logits:", router_logits.shape)
    print("expert_outputs:", expert_outputs.shape)
    print("shared_out:", shared_out.shape)
    print("out:", out.shape)
    print("top_indices:", top_indices.shape)
    print("router_weights:", router_weights.shape)

    print("top_indices:")
    print(top_indices)

    print("router_weights:")
    print(router_weights)

    print("router weight sums:")
    print(router_weights.sum(dim=-1))

    # Manual correctness check for one token.
    n = 0
    manual = shared_out[n].clone()

    for slot in range(top_k):
        expert_id = top_indices[n, slot]
        weight = router_weights[n, slot]
        manual += weight * expert_outputs[n, expert_id]

    print("manual token 0 error:", (manual - out[n]).abs().max())
    print("triton out error:", (out - tri_out).abs().max())
    print("triton top index match:", torch.equal(top_indices, tri_top_indices))
    print("triton router weight error:", (router_weights - tri_router_weights).abs().max())
    print("drouter error:", (drouter_ref - drouter_tri).abs().max())
    print("dexpert error:", (dexpert_ref - dexpert_tri).abs().max())
    print("dshared error:", (dshared_ref - dshared_tri).abs().max())

    print("\nautograd check")
    router_ref = router_logits.detach().clone().requires_grad_(True)
    expert_ref = expert_outputs.detach().clone().requires_grad_(True)
    shared_ref = shared_out.detach().clone().requires_grad_(True)

    out_ref, _, _ = torch_deepseek_moe_combine_ref(
        router_ref,
        expert_ref,
        shared_ref,
        top_k,
    )
    grad_out = torch.randn_like(out_ref)
    (out_ref * grad_out).sum().backward()

    router_tri = router_logits.detach().clone().requires_grad_(True)
    expert_tri = expert_outputs.detach().clone().requires_grad_(True)
    shared_tri = shared_out.detach().clone().requires_grad_(True)

    out_tri_autograd = deepseek_moe_combine_autograd(
        router_tri,
        expert_tri,
        shared_tri,
        top_k=top_k,
        block_d=4,
    )
    (out_tri_autograd * grad_out).sum().backward()

    print("autograd out error:", (out_ref - out_tri_autograd).abs().max())
    print("autograd router grad error:", (router_ref.grad - router_tri.grad).abs().max())
    print("autograd expert grad error:", (expert_ref.grad - expert_tri.grad).abs().max())
    print("autograd shared grad error:", (shared_ref.grad - shared_tri.grad).abs().max())

    print("\nbenchmark")
    for N in [512, 2048, 4096]:
        for E in [8, 16, 32]:
            for D in [512, 1024]:
                # Keep the standalone benchmark friendly to 4 GB GPUs.
                if N * E * D > 4096 * 16 * 1024:
                    continue
                run_backward_bench = N * E * D <= 2048 * 16 * 512

                router_logits = torch.randn((N, E), device=device, dtype=dtype)
                expert_outputs = torch.randn((N, E, D), device=device, dtype=dtype)
                shared_out = torch.randn((N, D), device=device, dtype=dtype)
                dout = torch.randn((N, D), device=device, dtype=dtype)

                torch_ms = benchmark(
                    torch_deepseek_moe_combine_ref,
                    router_logits,
                    expert_outputs,
                    shared_out,
                    top_k,
                )
                triton_ms = benchmark(
                    triton_deepseek_moe_combine,
                    router_logits,
                    expert_outputs,
                    shared_out,
                    top_k,
                    128,
                )
                out_tri_tmp, idx_tri_tmp, w_tri_tmp = triton_deepseek_moe_combine(
                    router_logits,
                    expert_outputs,
                    shared_out,
                    top_k,
                    block_d=128,
                )
                if run_backward_bench:
                    torch_bwd_ms = benchmark(
                        torch_deepseek_moe_combine_backward_ref,
                        router_logits,
                        expert_outputs,
                        shared_out,
                        dout,
                        top_k,
                        warmup=3,
                        iters=20,
                    )
                    triton_bwd_ms = benchmark(
                        triton_deepseek_moe_combine_backward,
                        router_logits,
                        expert_outputs,
                        shared_out,
                        dout,
                        idx_tri_tmp,
                        w_tri_tmp,
                        top_k,
                        128,
                        warmup=3,
                        iters=20,
                    )
                else:
                    torch_bwd_ms = float("nan")
                    triton_bwd_ms = float("nan")

                out_ref, idx_ref, w_ref = torch_deepseek_moe_combine_ref(
                    router_logits,
                    expert_outputs,
                    shared_out,
                    top_k,
                )
                out_tri, idx_tri, w_tri = triton_deepseek_moe_combine(
                    router_logits,
                    expert_outputs,
                    shared_out,
                    top_k,
                    block_d=128,
                )
                if run_backward_bench:
                    drouter_ref, dexpert_ref, dshared_ref = torch_deepseek_moe_combine_backward_ref(
                        router_logits,
                        expert_outputs,
                        shared_out,
                        dout,
                        top_k,
                    )
                    drouter_tri, dexpert_tri, dshared_tri = triton_deepseek_moe_combine_backward(
                        router_logits,
                        expert_outputs,
                        shared_out,
                        dout,
                        idx_tri,
                        w_tri,
                        top_k,
                        block_d=128,
                    )
                    drouter_err = (drouter_ref - drouter_tri).abs().max()
                    dexpert_err = (dexpert_ref - dexpert_tri).abs().max()
                    dshared_err = (dshared_ref - dshared_tri).abs().max()
                else:
                    drouter_err = "skipped"
                    dexpert_err = "skipped"
                    dshared_err = "skipped"

                print(
                    f"N={N} "
                    f"E={E} "
                    f"D={D} "
                    f"torch_ms={torch_ms:.4f} "
                    f"triton_ms={triton_ms:.4f} "
                    f"torch_bwd_ms={torch_bwd_ms:.4f} "
                    f"triton_bwd_ms={triton_bwd_ms:.4f} "
                    f"speedup={torch_ms / triton_ms:.2f}x "
                    f"speedup_bwd={(torch_bwd_ms / triton_bwd_ms) if run_backward_bench else 'skipped'} "
                    f"out_err={(out_ref - out_tri).abs().max()} "
                    f"idx_match={torch.equal(idx_ref, idx_tri)} "
                    f"weight_err={(w_ref - w_tri).abs().max()} "
                    f"drouter_err={drouter_err} "
                    f"dexpert_err={dexpert_err} "
                    f"dshared_err={dshared_err}"
                )

                del router_logits, expert_outputs, shared_out, dout
                del out_ref, idx_ref, w_ref, out_tri, idx_tri, w_tri
                del out_tri_tmp, idx_tri_tmp, w_tri_tmp
                if run_backward_bench:
                    del drouter_ref, dexpert_ref, dshared_ref
                    del drouter_tri, dexpert_tri, dshared_tri
                torch.cuda.empty_cache()



    
