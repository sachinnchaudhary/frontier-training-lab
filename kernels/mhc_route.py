import os

os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import triton   
import torch  
import triton.language as tl    


"""  

inputs:
  x_streams:  [N, S, D]
  H_pre_raw:  [N, S]
  H_post_raw: [N, S]
  H_res_raw:  [N, S, S]

outputs:
  h_in:       [N, D]
  H_post:     [N, S]
  H_res:      [N, S, S]


H_pre  = sigmoid(H_pre_raw)
H_post = 2 * sigmoid(H_post_raw)

H_res = exp(H_res_raw - max(H_res_raw))
repeat sinkhorn_iters:
    normalize columns
    normalize rows

h_in = sum_s H_pre[n,s] * x_streams[n,s,:]    


"""

def torch_sinkhorn_ref(H_res_raw, iters=8, eps=1e-6):   
   """
    H_res_raw: [N, S, S]
    returns:
      H_res:  [N, S, S]
   """

   assert H_res_raw.ndim == 3
   
   H = torch.exp(H_res_raw - H_res_raw.amax(dim=(-1, -2), keepdim=True)) 

   for _ in range(iters):  
       
       H = H / (H.sum(dim=-2, keepdim=True) + eps)   

       H = H / (H.sum(dim=-1, keepdim=True) + eps)  

   return H      


def torch_mhc_route_backward_ref(
    x_streams,
    H_pre_raw,
    H_post_raw,
    H_res_raw,
    dh_in,
    dH_post,
    dH_res,
    sinkhorn_iters=8,
    eps=1e-6,
):
    """
    Manual backward reference for torch_mhc_route_ref.

    Inputs:
      x_streams:  [N, S, D]
      H_pre_raw:  [N, S]
      H_post_raw: [N, S]
      H_res_raw:  [N, S, S]

      dh_in:      [N, D]
      dH_post:    [N, S]
      dH_res:     [N, S, S]

    Returns:
      dx_streams:   [N, S, D]
      dH_pre_raw:   [N, S]
      dH_post_raw:  [N, S]
      dH_res_raw:   [N, S, S]
    """

    x_streams = x_streams.detach().clone().requires_grad_(True)
    H_pre_raw = H_pre_raw.detach().clone().requires_grad_(True)
    H_post_raw = H_post_raw.detach().clone().requires_grad_(True)
    H_res_raw = H_res_raw.detach().clone().requires_grad_(True)

    h_in, H_post, H_res = torch_mhc_route_ref(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )

    loss = (
        (h_in * dh_in).sum()
        + (H_post * dH_post).sum()
        + (H_res * dH_res).sum()
    )

    loss.backward()

    return (
        x_streams.grad,
        H_pre_raw.grad,
        H_post_raw.grad,
        H_res_raw.grad,
    )


def torch_sinkhorn_manual_backward_ref(H_res_raw, dH_res, iters=8, eps=1e-6):
    raw = H_res_raw

    max_val = raw.amax(dim=(-1, -2), keepdim=True)
    Z = torch.exp(raw - max_val)

    H = Z
    states = []

    for _ in range(iters):
        col_sum = H.sum(dim=-2, keepdim=True) + eps
        H_col = H / col_sum

        row_sum = H_col.sum(dim=-1, keepdim=True) + eps
        H_row = H_col / row_sum

        states.append((H, col_sum, H_col, row_sum, H_row))

        H = H_row

    dH = dH_res

    for (H_before_col, col_sum, H_col, row_sum, H_row) in reversed(states):
        # H_row = H_col / row_sum
        row_dot = torch.sum(dH * H_row, dim=-1, keepdim=True)
        dH_col = (dH - row_dot) / row_sum

        # H_col = H_before_col / col_sum
        col_dot = torch.sum(dH_col * H_col, dim=-2, keepdim=True)
        dH_before_col = (dH_col - col_dot) / col_sum

        dH = dH_before_col

    dZ = dH

    # Z = exp(raw - max(raw))
    draw_no_max = dZ * Z

    # Backward through raw - max(raw)
    max_mask = raw == max_val
    max_count = max_mask.sum(dim=(-1, -2), keepdim=True)
    draw = draw_no_max - max_mask * (
        draw_no_max.sum(dim=(-1, -2), keepdim=True) / max_count
    )

    return draw   


def torch_mhc_route_ref(
    x_streams, 
    H_pre_raw,  
    H_post_raw,  
    H_res_raw,      
    sinkhorn_iters=8, 
    eps = 1e-6,    
):  
   
   """  
   
   x_streams:  [N, S, D]
    H_pre_raw:  [N, S]
    H_post_raw: [N, S]
    H_res_raw:  [N, S, S]

    returns:
      h_in:    [N, D]
      H_post:  [N, S]
      H_res:   [N, S, S]
   
   
   """
   
   assert x_streams.ndim == 3
   assert H_pre_raw.ndim == 2
   assert H_post_raw.ndim == 2
   assert H_res_raw.ndim == 3

   N, S, D = x_streams.shape

   assert H_pre_raw.shape == (N, S)
   assert H_post_raw.shape == (N, S)
   assert H_res_raw.shape == (N, S, S)

   H_pre = torch.sigmoid(H_pre_raw)  
   H_post = 2.0 *  torch.sigmoid(H_post_raw)  
   H_res =  torch_sinkhorn_ref(H_res_raw, iters=sinkhorn_iters, eps=eps)  

   h_in = torch.sum(H_pre[..., None] * x_streams, dim=1)  

   return  h_in, H_post, H_res  
 


@triton.jit

def _mhc_route_kernel(

   x_ptr, 
   hpre_raw_ptr, 
   hpost_raw_ptr,  
   hres_raw_ptr,  
   h_in_ptr,  
   hpost_ptr,  
   hres_ptr,  
   S: tl.constexpr, 
   D: tl.constexpr,  
   SINKHORN_ITERS: tl.constexpr,  
   EPS: tl.constexpr,  
   BLOCK_D: tl.constexpr,  
   BLOCK_S: tl.constexpr,
): 
     
   """  
   
   x_streams:  [N, S, D]
    H_pre_raw:  [N, S]
    H_post_raw: [N, S]
    H_res_raw:  [N, S, S]

    returns:
      h_in:    [N, D]
      H_post:  [N, S]
      H_res:   [N, S, S]
   
   
   """    
  
   pid_n = tl.program_id(0)  
   pid_d = tl.program_id(1)  

   n = pid_n    
   offs_d =  pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  
   mask_d =  offs_d < D  

   acc = tl.zeros((BLOCK_D, ), tl.float32)   

   for s in range(S):  

      hpre_raw = tl.load(hpre_raw_ptr + n * S + s).to(tl.float32)     
      hpre = tl.sigmoid(hpre_raw)  

      x = tl.load(  
          x_ptr + n * S * D  + s * D + offs_d, 
          mask=mask_d, 
          other=0.0 
      ).to(tl.float32) 

      acc += hpre * x   

   tl.store(h_in_ptr + n * D + offs_d, acc, mask=mask_d)   
   if pid_d == 0:
     for s in range(S):  

       raw = tl.load(hpost_raw_ptr + n * S + s).to(tl.float32)  
       hpost = 2.0 / (1.0 +tl.exp(-raw))  
       tl.store(hpost_ptr + n * S + s, hpost)  

     offs_i = tl.arange(0, BLOCK_S) 
     offs_j = tl.arange(0, BLOCK_S)  

     valid = (offs_i[:, None] < S) & (offs_j[None, :] < S)  

     #H_res_raw:  [N, S, S]  
     raw = tl.load(  
        hres_raw_ptr + n * S * S + offs_i[:, None] * S + offs_j[None, :],
        mask=valid,  
        other=-float("inf"),       
     ).to(tl.float32)
     
     row_max = tl.max(raw, axis=1)  
     max_val = tl.max(row_max, axis=0)   

     H = tl.exp(raw -max_val)  
     H = tl.where(valid, H, 0.0)  

     for _ in range(SINKHORN_ITERS):  
        col_sum = tl.sum(H, axis=0)  
        H = H / (col_sum[None, :] + EPS)  
        H = tl.where(valid, H, 0.0) 

        row_sum = tl.sum(H, axis=1)  
        H = H / (row_sum[:, None] + EPS)  
        H = tl.where(valid, H, 0.0)  

     tl.store(
        hres_ptr + n * S * S + offs_i[:, None] * S + offs_j[None, :],
        H, 
        mask=valid,
        ) 
     

@triton.jit
def _mhc_route_read_backward_kernel(

    x_ptr,              # [N, S, D]
    hpre_raw_ptr,       # [N, S]
    hpost_raw_ptr,      # [N, S]
    dh_in_ptr,          # [N, D]
    dH_post_ptr,        # [N, S]

    dx_ptr,             # [N, S, D]
    dH_pre_raw_ptr,     # [N, S]
    dH_post_raw_ptr,    # [N, S]

    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,  

):  

  pid_ns = tl.program_id(0) 
  pid_d = tl.program_id(1)  

  n = pid_ns // S  
  s = pid_ns % S  

  offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  
  mask_d = offs_d < D    

  hpre_raw = tl.load(hpre_raw_ptr + n * S + s).to(tl.float32)   
  hpre = 1.0 / (1.0 + tl.exp(-hpre_raw))
  
  dh = tl.load(
     dh_in_ptr + n * D + offs_d,
     mask = mask_d, 
     other=0.0,  
     ).to(tl.float32)  

  x = tl.load(
      x_ptr + n * S * D + s * D + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32) 
   
  dx = dh * hpre  

  tl.store(
     dx_ptr + n * S * D + s * D + offs_d, 
     dx, 
     mask=mask_d, 
  )  

  partial = tl.sum(dh * x, axis=0)  
  partial_raw = partial * hpre * (1.0 - hpre)  

  tl.atomic_add(
     dH_pre_raw_ptr + n * S +  s, 
     partial_raw, 
  ) 

  if pid_d == 0:  
     hpost_raw = tl.load(hpost_raw_ptr + n * S + s).to(tl.float32)  
     hpost_sig = 1.0 / (1.0 + tl.exp(-hpost_raw))  

     dH_post = tl.load(dH_post_ptr+ n * S + s).to(tl.float32)  

     dH_post_raw = dH_post * 2.0 * hpost_sig * (1.0 - hpost_sig)  

     tl.store(
        dH_post_raw_ptr + n * S + s,  
        dH_post_raw, 
     )

  
def triton_mhc_route_read_backward(
    x_streams,
    H_pre_raw,
    H_post_raw,
    dh_in,
    dH_post,
    block_d=128, 
):  
    
    """
    x_streams:  [N, S, D]
    H_pre_raw:  [N, S]
    H_post_raw: [N, S]

    dh_in:      [N, D]
    dH_post:    [N, S]

    returns:
      dx_streams:   [N, S, D]
      dH_pre_raw:   [N, S]
      dH_post_raw:  [N, S]
    """  

    assert x_streams.is_cuda
    assert H_pre_raw.is_cuda
    assert H_post_raw.is_cuda
    assert dh_in.is_cuda
    assert dH_post.is_cuda

    assert x_streams.ndim == 3
    assert H_pre_raw.ndim == 2
    assert H_post_raw.ndim == 2
    assert dh_in.ndim == 2
    assert dH_post.ndim == 2

    N, S, D = x_streams.shape

    assert H_pre_raw.shape == (N, S)
    assert H_post_raw.shape == (N, S)
    assert dh_in.shape == (N, D)
    assert dH_post.shape == (N, S)  

    x_streams = x_streams.contiguous()
    H_pre_raw = H_pre_raw.contiguous()
    H_post_raw = H_post_raw.contiguous()
    dh_in = dh_in.contiguous()
    dH_post = dH_post.contiguous()

    dx_streams = torch.empty_like(x_streams) 

    dH_pre_raw = torch.zeros_like(H_pre_raw)

    dH_post_raw = torch.empty_like(H_post_raw)

    grid = (N * S, triton.cdiv(D, block_d)) 

    _mhc_route_read_backward_kernel[grid](
       x_streams,
        H_pre_raw,
        H_post_raw,
        dh_in,
        dH_post,
        dx_streams,
        dH_pre_raw,
        dH_post_raw,
        S,
        D,
        block_d,
    )

    return dx_streams, dH_pre_raw, dH_post_raw


@triton.jit 
def _sinkhorn_backward_kernel(  
    raw_ptr,        # [N,S,S]
    dH_ptr,         # [N,S,S]
    draw_ptr,       # [N,S,S]
    S: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_S: tl.constexpr,   
):  
   
   n = tl.program_id(0)  

   offs_i = tl.arange(0, BLOCK_S)  
   offs_j = tl.arange(0, BLOCK_S)  

   valid = (offs_i[:, None] < S) & (offs_j[None, :] < S)   

   raw = tl.load(
      raw_ptr + n*S*S + offs_i[:, None]*S + offs_j[None, :],  
      mask=valid,  
      other=-float("inf"),  
   ).to(tl.float32)  

   dH = tl.load(
        dH_ptr + n*S*S + offs_i[:, None]*S + offs_j[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)

   row_max = tl.max(raw, axis=1)  
   max_val = tl.max(row_max, axis=0)  

   Z = tl.exp(raw - max_val)  
   Z = tl.where(valid, Z, 0.0)  

   dCur = dH  
   
   for rev in range(SINKHORN_ITERS):  
       H = Z  
       col_sum = tl.sum(H, axis=0) + EPS
       H_col = H
       row_sum = tl.sum(H, axis=1) + EPS
       H_row = H

       for _ in range(SINKHORN_ITERS - rev):  
          col_sum = tl.sum(H, axis=0) + EPS 
          H_col = H / col_sum[None, :]  
          H_col = tl.where(valid, H_col, 0.0)  

          row_sum = tl.sum(H_col, axis=1) + EPS 
          H_row = H_col / row_sum[:, None]  
          H_row = tl.where(valid, H_row, 0.0)  

          H = H_row

       row_dot = tl.sum(dCur * H_row, axis=1)
       dH_col = (dCur - row_dot[:, None]) / row_sum[:, None]
       dH_col = tl.where(valid, dH_col, 0.0)

       col_dot = tl.sum(dH_col * H_col, axis=0)
       d_before_col = (dH_col - col_dot[None, :]) / col_sum[None, :]
       dCur = tl.where(valid, d_before_col, 0.0)

   draw_no_max = dCur * Z
   total = tl.sum(tl.sum(draw_no_max, axis=1), axis=0)

   max_mask = raw == max_val
   max_count = tl.sum(tl.sum(max_mask.to(tl.float32), axis=1), axis=0)
   draw = draw_no_max - max_mask.to(tl.float32) * (total / max_count)
   draw = tl.where(valid, draw, 0.0)

   tl.store(
      draw_ptr + n*S*S + offs_i[:, None]*S + offs_j[None, :],
      draw,
      mask=valid,
   )


def triton_sinkhorn_backward(H_res_raw, dH_res, sinkhorn_iters=8, eps=1e-6):
    """
    H_res_raw: [N, S, S]
    dH_res:    [N, S, S]

    returns:
      dH_res_raw: [N, S, S]
    """

    assert H_res_raw.is_cuda and dH_res.is_cuda
    assert H_res_raw.ndim == 3
    assert dH_res.ndim == 3
    assert H_res_raw.shape == dH_res.shape

    N, S, S2 = H_res_raw.shape
    assert S == S2

    H_res_raw = H_res_raw.contiguous()
    dH_res = dH_res.contiguous()

    dH_res_raw = torch.empty_like(H_res_raw)
    grid = (N,)

    _sinkhorn_backward_kernel[grid](
        H_res_raw,
        dH_res,
        dH_res_raw,
        S,
        sinkhorn_iters,
        eps,
        triton.next_power_of_2(S),
    )

    return dH_res_raw


def triton_mhc_route(  
      
    x_streams,
    H_pre_raw,
    H_post_raw,
    H_res_raw,
    sinkhorn_iters=8,
    eps=1e-6,
    block_d=128,

):  
   
    """
    x_streams:  [N, S, D]
    H_pre_raw:  [N, S]
    H_post_raw: [N, S]
    H_res_raw:  [N, S, S]

    returns:
      h_in:    [N, D]
      H_post:  [N, S]
      H_res:   [N, S, S]
    """
   
    assert x_streams.is_cuda
    assert H_pre_raw.is_cuda
    assert H_post_raw.is_cuda
    assert H_res_raw.is_cuda

    assert x_streams.ndim == 3
    assert H_pre_raw.ndim == 2
    assert H_post_raw.ndim == 2
    assert H_res_raw.ndim == 3

    N, S, D = x_streams.shape

    assert H_pre_raw.shape == (N, S)
    assert H_post_raw.shape == (N, S)
    assert H_res_raw.shape == (N, S, S)

    x_streams = x_streams.contiguous()
    H_pre_raw = H_pre_raw.contiguous()
    H_post_raw = H_post_raw.contiguous()
    H_res_raw = H_res_raw.contiguous()

    h_in = torch.empty((N, D), device=x_streams.device, dtype=x_streams.dtype) 
    H_post = torch.empty((N, S), device=x_streams.device, dtype=x_streams.dtype) 
    H_res = torch.empty((N, S, S), device = x_streams.device, dtype=x_streams.dtype)  

    grid = (N, triton.cdiv(D, block_d))

    _mhc_route_kernel[grid](   
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        h_in,
        H_post,
        H_res,
        S,
        D,
        sinkhorn_iters,
        eps,
        block_d,
        triton.next_power_of_2(S),
    )   

    return h_in, H_post, H_res  


def torch_mhc_route_read_backward_ref(
    x_streams,
    H_pre_raw,
    H_post_raw,
    dh_in,
    dH_post,
):
    """
    Reference for the non-Sinkhorn route backward path.

    Covers:
      h_in = sum_s sigmoid(H_pre_raw)[s] * x_streams[s]
      H_post = 2 * sigmoid(H_post_raw)

    Does not compute dH_res_raw.
    """

    H_pre = torch.sigmoid(H_pre_raw)
    dx_streams = dh_in[:, None, :] * H_pre[:, :, None]

    dH_pre = torch.sum(dh_in[:, None, :] * x_streams, dim=-1)
    dH_pre_raw = dH_pre * H_pre * (1.0 - H_pre)

    H_post_sig = torch.sigmoid(H_post_raw)
    dH_post_raw = dH_post * 2.0 * H_post_sig * (1.0 - H_post_sig)

    return dx_streams, dH_pre_raw, dH_post_raw


def benchmark(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, tuple):
            out[0].sum().item()
        else:
            out.sum().item()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = fn(*args)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters





if __name__ == "__main__":  
   
    torch.manual_seed(0)  

    device = "cuda"  

    dtype = torch.float32  

    N, S, D = 3, 4, 8  

    x_streams = torch.randn((N, S, D), device=device, dtype=dtype)
    H_pre_raw = torch.randn((N, S), device=device, dtype=dtype)
    H_post_raw = torch.randn((N, S), device=device, dtype=dtype)
    H_res_raw = torch.randn((N, S, S), device=device, dtype=dtype)

    h_in, H_post, H_res = torch_mhc_route_ref(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=8,
    )
    
    h_tri, H_post_tri, H_res_tri = triton_mhc_route(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=8,
        block_d=4,
    )

    print("h_tri:", h_tri.shape)
    print("H_post_tri:", H_post_tri.shape)
    print("H_res_tri:", H_res_tri.shape)

    print("h error:", (h_in - h_tri).abs().max())
    print("H_post error:", (H_post - H_post_tri).abs().max())
    print("H_res error:", (H_res - H_res_tri).abs().max())

    print("H_res_tri row sums:", H_res_tri.sum(dim=-1))
    print("H_res_tri col sums:", H_res_tri.sum(dim=-2)) 

    print("\nbackward ref check")
    dh_in = torch.randn_like(h_in)
    dH_post = torch.randn_like(H_post)
    dH_res = torch.randn_like(H_res)

    dx, dpre, dpost, dres = torch_mhc_route_backward_ref(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        dh_in,
        dH_post,
        dH_res,
        sinkhorn_iters=8,
    )

    print("dx:", dx.shape)
    print("dH_pre_raw:", dpre.shape)
    print("dH_post_raw:", dpost.shape)
    print("dH_res_raw:", dres.shape)
    print("finite:", torch.isfinite(dx).all(), torch.isfinite(dres).all())

    dx_easy, dpre_easy, dpost_easy = triton_mhc_route_read_backward(
        x_streams,
        H_pre_raw,
        H_post_raw,
        dh_in,
        dH_post,
        block_d=4,
    )
    print("easy backward dx error:", (dx - dx_easy).abs().max())
    print("easy backward dpre error:", (dpre - dpre_easy).abs().max())
    print("easy backward dpost error:", (dpost - dpost_easy).abs().max())

    print("\nbenchmark")
    for N in [512, 2048]:
      for S in [4, 8]:
       for D in [512, 1024]:
        x_streams = torch.randn((N, S, D), device=device, dtype=dtype)
        H_pre_raw = torch.randn((N, S), device=device, dtype=dtype)
        H_post_raw = torch.randn((N, S), device=device, dtype=dtype)
        H_res_raw = torch.randn((N, S, S), device=device, dtype=dtype)
        dh_in = torch.randn((N, D), device=device, dtype=dtype)
        dH_post = torch.randn((N, S), device=device, dtype=dtype)
        dH_res = torch.randn((N, S, S), device=device, dtype=dtype)

        torch_ms = benchmark(
            torch_mhc_route_ref,
            x_streams,
            H_pre_raw,
            H_post_raw,
            H_res_raw,
            8,
            1e-6,
        )
        triton_ms = benchmark(
            triton_mhc_route,
            x_streams,
            H_pre_raw,
            H_post_raw,
            H_res_raw,
            8,
            1e-6,
            128,
        )
        torch_bwd_ms = benchmark(
            torch_mhc_route_read_backward_ref,
            x_streams,
            H_pre_raw,
            H_post_raw,
            dh_in,
            dH_post,
        )
        triton_bwd_ms = benchmark(
            triton_mhc_route_read_backward,
            x_streams,
            H_pre_raw,
            H_post_raw,
            dh_in,
            dH_post,
            128,
        )
        torch_sink_bwd_ms = benchmark(
            torch_sinkhorn_manual_backward_ref,
            H_res_raw,
            dH_res,
            8,
            1e-6,
        )
        triton_sink_bwd_ms = benchmark(
            triton_sinkhorn_backward,
            H_res_raw,
            dH_res,
            8,
            1e-6,
        )

        h_ref, hp_ref, hr_ref = torch_mhc_route_ref(
            x_streams,
            H_pre_raw,
            H_post_raw,
            H_res_raw,
            sinkhorn_iters=8,
        )
        h_tri, hp_tri, hr_tri = triton_mhc_route(
            x_streams,
            H_pre_raw,
            H_post_raw,
            H_res_raw,
            sinkhorn_iters=8,
            block_d=128,
        )

        h_err = (h_ref - h_tri).abs().max()
        hp_err = (hp_ref - hp_tri).abs().max()
        hr_err = (hr_ref - hr_tri).abs().max()
        dx_ref, dpre_ref, dpost_ref = torch_mhc_route_read_backward_ref(
            x_streams,
            H_pre_raw,
            H_post_raw,
            dh_in,
            dH_post,
        )
        dx_tri, dpre_tri, dpost_tri = triton_mhc_route_read_backward(
            x_streams,
            H_pre_raw,
            H_post_raw,
            dh_in,
            dH_post,
            block_d=128,
        )
        dx_err = (dx_ref - dx_tri).abs().max()
        dpre_err = (dpre_ref - dpre_tri).abs().max()
        dpost_err = (dpost_ref - dpost_tri).abs().max()
        dres_ref = torch_sinkhorn_manual_backward_ref(H_res_raw, dH_res, iters=8)
        dres_tri = triton_sinkhorn_backward(H_res_raw, dH_res, sinkhorn_iters=8)
        dres_err = (dres_ref - dres_tri).abs().max()

        print(
            f"N={N} "
            f"S={S} "
            f"D={D} "
            f"torch_ms={torch_ms:.4f} "
            f"triton_ms={triton_ms:.4f} "
            f"torch_bwd_ms={torch_bwd_ms:.4f} "
            f"triton_bwd_ms={triton_bwd_ms:.4f} "
            f"torch_sink_bwd_ms={torch_sink_bwd_ms:.4f} "
            f"triton_sink_bwd_ms={triton_sink_bwd_ms:.4f} "
            f"speedup={torch_ms / triton_ms:.2f}x "
            f"speedup_bwd={torch_bwd_ms / triton_bwd_ms:.2f}x "
            f"speedup_sink_bwd={torch_sink_bwd_ms / triton_sink_bwd_ms:.2f}x "
            f"h_err={h_err} "
            f"H_post_err={hp_err} "
            f"H_res_err={hr_err} "
            f"dx_err={dx_err} "
            f"dpre_err={dpre_err} "
            f"dpost_err={dpost_err} "
            f"dres_err={dres_err}"
        )
    
    print("\nsinkhorn manual backward check")

    raw = torch.randn((3, 4, 4), device=device, dtype=dtype, requires_grad=True)
    H = torch_sinkhorn_ref(raw, iters=8)
    dH = torch.randn_like(H)

    (H * dH).sum().backward()

    manual = torch_sinkhorn_manual_backward_ref(
    raw.detach(),
    dH,
    iters=8,)
    tri = triton_sinkhorn_backward(raw.detach(), dH, sinkhorn_iters=8)

    print("sinkhorn draw error:", (raw.grad - manual).abs().max())
    print("sinkhorn triton error:", (manual - tri).abs().max())
    print("sinkhorn finite:", torch.isfinite(manual).all())
