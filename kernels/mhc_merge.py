import os

os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import triton   
import torch  
import triton.language as tl   


def torch_mhc_merge_ref(x_streams, h_out, H_post, H_res):  
     
    """
    x_streams: [N, S, D]
    h_out:     [N, D]
    H_post:    [N, S]
    H_res:     [N, S, S]

    returns:
      x_next:  [N, S, D]
    
    """
     
    assert h_out.ndim == 2
    assert H_post.ndim == 2
    assert H_res.ndim == 3 

    N ,S, D = x_streams.shape  

    assert h_out.shape == (N, D)  
    assert H_post.shape == (N, S)  
    assert H_res.shape == (N, S, S)  

    highway = torch.einsum("nsr,nrd->nsd", H_res, x_streams) 
    update = H_post[..., None]  * h_out[:, None, :]  

    return highway + update  


def torch_mhc_merge_backward_ref(x_streams, h_out, H_post, H_res, dx_next):
    """
    Reference backward for torch_mhc_merge_ref.

    dx_next: [N, S, D]

    returns:
      dx_streams: [N, S, D]
      dh_out:     [N, D]
      dH_post:    [N, S]
      dH_res:     [N, S, S]
    """

    x_streams = x_streams.detach().clone().requires_grad_(True)
    h_out = h_out.detach().clone().requires_grad_(True)
    H_post = H_post.detach().clone().requires_grad_(True)
    H_res = H_res.detach().clone().requires_grad_(True)

    x_next = torch_mhc_merge_ref(x_streams, h_out, H_post, H_res)
    loss = (x_next * dx_next).sum()
    loss.backward()

    return x_streams.grad, h_out.grad, H_post.grad, H_res.grad


@triton.jit
def _mhc_merge_kernel(
  x_ptr, 
  h_ptr, 
  hpost_ptr, 
  hres_ptr, 
  out_ptr,
  S: tl.constexpr, 
  D: tl.constexpr,  
  BLOCK_D: tl.constexpr
): 
     
  pid_ns = tl.program_id(0)  
  pid_d = tl.program_id(1)  

  n = pid_ns // S  
  s = pid_ns % S  
  
  offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  
  mask_d = offs_d < D     

  """
    x_streams: [N, S, D]
    h_out:     [N, D]
    H_post:    [N, S]
    H_res:     [N, S, S]

    returns:
      x_next:  [N, S, D]
    
  """

  hpost = tl.load(hpost_ptr + n * S + s).to(tl.float32)  
  h = tl.load(
     h_ptr + n * D + offs_d, 
     mask = mask_d,  
     other = 0.0,
  ).to(tl.float32)    

  acc = hpost * h  

  for r in range(S):  
     hres = tl.load(hres_ptr + n  * S * S + s * S + r).to(tl.float32)  
     x = tl.load(
        x_ptr + n * S * D + r * D + offs_d, 
        mask= mask_d, 
        other=0.0, 
     ).to(tl.float32)   
     acc += hres * x  

  tl.store(
     out_ptr + n * S * D + s * D + offs_d, 
     acc, 
     mask=mask_d 
  )


def triton_mhc_merge(x_streams, h_out, H_post, H_res, block_d=128): 

    """
    x_streams: [N, S, D]
    h_out:     [N, D]
    H_post:    [N, S]
    H_res:     [N, S, S]

    returns:
      x_next:  [N, S, D]
    """ 

    assert x_streams.is_cuda
    assert h_out.is_cuda
    assert H_post.is_cuda
    assert H_res.is_cuda

    assert x_streams.ndim == 3
    assert h_out.ndim == 2
    assert H_post.ndim == 2
    assert H_res.ndim == 3

    N, S, D = x_streams.shape

    assert h_out.shape == (N, D)
    assert H_post.shape == (N, S)
    assert H_res.shape == (N, S, S)

    x_streams = x_streams.contiguous()
    h_out = h_out.contiguous()
    H_post = H_post.contiguous()
    H_res = H_res.contiguous()

    x_next = torch.empty_like(x_streams)  

    grid = (N * S, triton.cdiv(D, block_d))  
    
    _mhc_merge_kernel[grid]( 
       x_streams,
        h_out,
        H_post,
        H_res,
        x_next,
        S,
        D,
        block_d,       
    )

    return x_next  


@triton.jit
def _mhc_merge_backward_vec_kernel(
    dx_next_ptr,
    hpost_ptr,
    hres_ptr,
    dx_streams_ptr,
    dh_out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computes vector-shaped gradients:
      dx_streams[n,r,d] = sum_s dx_next[n,s,d] * H_res[n,s,r]
      dh_out[n,d]       = sum_s dx_next[n,s,d] * H_post[n,s]
    """

    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    dh = tl.zeros((BLOCK_D,), tl.float32)

    for s in range(S):
        dout = tl.load(
            dx_next_ptr + pid_n * S * D + s * D + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        hpost = tl.load(hpost_ptr + pid_n * S + s).to(tl.float32)
        dh += dout * hpost

    tl.store(
        dh_out_ptr + pid_n * D + offs_d,
        dh,
        mask=mask_d,
    )

    for r in range(S):
        dx = tl.zeros((BLOCK_D,), tl.float32)

        for s in range(S):
            dout = tl.load(
                dx_next_ptr + pid_n * S * D + s * D + offs_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            hres = tl.load(
                hres_ptr + pid_n * S * S + s * S + r,
            ).to(tl.float32)
            dx += dout * hres

        tl.store(
            dx_streams_ptr + pid_n * S * D + r * D + offs_d,
            dx,
            mask=mask_d,
        )


@triton.jit
def _mhc_merge_backward_coeff_kernel(
    x_ptr,
    h_ptr,
    dx_next_ptr,
    dHpost_ptr,
    dHres_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computes coefficient-shaped gradients:
      dH_post[n,s] = sum_d dx_next[n,s,d] * h_out[n,d]
      dH_res[n,s,r] = sum_d dx_next[n,s,d] * x_streams[n,r,d]
    """

    pid = tl.program_id(0)

    n = pid // (S * S)
    rem = pid - n * S * S
    s = rem // S
    r = rem - s * S

    dpost_acc = tl.zeros((BLOCK_D,), tl.float32)
    dres_acc = tl.zeros((BLOCK_D,), tl.float32)

    for start in range(0, D, BLOCK_D):
        offs_d = start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        dout = tl.load(
            dx_next_ptr + n * S * D + s * D + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + n * S * D + r * D + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        dres_acc += dout * x

        if r == 0:
            h = tl.load(
                h_ptr + n * D + offs_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            dpost_acc += dout * h

    dres = tl.sum(dres_acc, axis=0)
    tl.store(dHres_ptr + n * S * S + s * S + r, dres)

    if r == 0:
        dpost = tl.sum(dpost_acc, axis=0)
        tl.store(dHpost_ptr + n * S + s, dpost)


def triton_mhc_merge_backward(x_streams, h_out, H_post, H_res, dx_next, block_d=128):
    """
    Backward for triton_mhc_merge.

    returns:
      dx_streams: [N, S, D]
      dh_out:     [N, D]
      dH_post:    [N, S]
      dH_res:     [N, S, S]
    """

    assert x_streams.is_cuda
    assert h_out.is_cuda
    assert H_post.is_cuda
    assert H_res.is_cuda
    assert dx_next.is_cuda

    assert x_streams.ndim == 3
    assert h_out.ndim == 2
    assert H_post.ndim == 2
    assert H_res.ndim == 3
    assert dx_next.ndim == 3

    N, S, D = x_streams.shape

    assert h_out.shape == (N, D)
    assert H_post.shape == (N, S)
    assert H_res.shape == (N, S, S)
    assert dx_next.shape == (N, S, D)

    x_streams = x_streams.contiguous()
    h_out = h_out.contiguous()
    H_post = H_post.contiguous()
    H_res = H_res.contiguous()
    dx_next = dx_next.contiguous()

    dx_streams = torch.empty_like(x_streams)
    dh_out = torch.empty_like(h_out)
    dH_post = torch.empty_like(H_post)
    dH_res = torch.empty_like(H_res)

    vec_grid = (N, triton.cdiv(D, block_d))
    _mhc_merge_backward_vec_kernel[vec_grid](
        dx_next,
        H_post,
        H_res,
        dx_streams,
        dh_out,
        S,
        D,
        block_d,
    )

    coeff_grid = (N * S * S,)
    _mhc_merge_backward_coeff_kernel[coeff_grid](
        x_streams,
        h_out,
        dx_next,
        dH_post,
        dH_res,
        S,
        D,
        block_d,
    )

    return dx_streams, dh_out, dH_post, dH_res


def benchmark(fn, *args, warmup=10, iters=100):
    def consume(out):
        if isinstance(out, tuple):
            return sum(x.sum() for x in out).item()
        return out.sum().item()

    for _ in range(warmup):
        out = fn(*args)
        consume(out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = fn(*args)
    consume(out)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ ==  "__main__":  

    torch.manual_seed(0)  

    device = "cuda" 
    dtype = torch.float32

    N, S, D = 3,4,8  

    x_streams = torch.randn((N, S, D), device=device, dtype=dtype)
    h_out = torch.randn((N, D), device=device, dtype=dtype)
    H_post = torch.randn((N, S), device=device, dtype=dtype)
    H_res = torch.randn((N, S, S), device=device, dtype=dtype)

    x_next = torch_mhc_merge_ref(x_streams, h_out, H_post, H_res) 
    x_tri = triton_mhc_merge(x_streams, h_out, H_post, H_res, block_d=4)
    dx_next = torch.randn_like(x_next)

    dx_ref, dh_ref, dpost_ref, dres_ref = torch_mhc_merge_backward_ref(
        x_streams,
        h_out,
        H_post,
        H_res,
        dx_next,
    )
    dx_tri, dh_tri, dpost_tri, dres_tri = triton_mhc_merge_backward(
        x_streams,
        h_out,
        H_post,
        H_res,
        dx_next,
        block_d=4,
    )

    print("x_tri:", x_tri.shape)
    print("max error:", (x_next - x_tri).abs().max())
    print("dx error:", (dx_ref - dx_tri).abs().max())
    print("dh error:", (dh_ref - dh_tri).abs().max())
    print("dH_post error:", (dpost_ref - dpost_tri).abs().max())
    print("dH_res error:", (dres_ref - dres_tri).abs().max())

    print("x_streams:", x_streams.shape)
    print("h_out:", h_out.shape)
    print("H_post:", H_post.shape)
    print("H_res:", H_res.shape)
    print("x_next:", x_next.shape)  

    print("\nbenchmark")
    for N in [512, 2048]:
      for S in [4, 8]:
       for D in [512, 1024]:
        x_streams = torch.randn((N, S, D), device=device, dtype=dtype)
        h_out = torch.randn((N, D), device=device, dtype=dtype)
        H_post = torch.randn((N, S), device=device, dtype=dtype)
        H_res = torch.randn((N, S, S), device=device, dtype=dtype)
        dx_next = torch.randn((N, S, D), device=device, dtype=dtype)

        torch_ms = benchmark(torch_mhc_merge_ref, x_streams, h_out, H_post, H_res)
        triton_ms = benchmark(triton_mhc_merge, x_streams, h_out, H_post, H_res, 128)
        torch_bwd_ms = benchmark(
            torch_mhc_merge_backward_ref,
            x_streams,
            h_out,
            H_post,
            H_res,
            dx_next,
        )
        triton_bwd_ms = benchmark(
            triton_mhc_merge_backward,
            x_streams,
            h_out,
            H_post,
            H_res,
            dx_next,
            128,
        )

        x_ref = torch_mhc_merge_ref(x_streams, h_out, H_post, H_res)
        x_tri = triton_mhc_merge(x_streams, h_out, H_post, H_res, block_d=128)
        err = (x_ref - x_tri).abs().max()
        dx_ref, dh_ref, dpost_ref, dres_ref = torch_mhc_merge_backward_ref(
            x_streams,
            h_out,
            H_post,
            H_res,
            dx_next,
        )
        dx_tri, dh_tri, dpost_tri, dres_tri = triton_mhc_merge_backward(
            x_streams,
            h_out,
            H_post,
            H_res,
            dx_next,
            block_d=128,
        )

        print(
            f"N={N} "
            f"S={S} "
            f"D={D} "
            f"torch_ms={torch_ms:.4f} "
            f"triton_ms={triton_ms:.4f} "
            f"torch_bwd_ms={torch_bwd_ms:.4f} "
            f"triton_bwd_ms={triton_bwd_ms:.4f} "
            f"speedup={torch_ms / triton_ms:.2f}x "
            f"speedup_bwd={torch_bwd_ms / triton_bwd_ms:.2f}x "
            f"err={err} "
            f"dx_err={(dx_ref - dx_tri).abs().max()} "
            f"dh_err={(dh_ref - dh_tri).abs().max()} "
            f"dpost_err={(dpost_ref - dpost_tri).abs().max()} "
            f"dres_err={(dres_ref - dres_tri).abs().max()}"
        )
