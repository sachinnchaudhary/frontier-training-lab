import os

os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import triton   
import torch  
import triton.language as tl 
import math


"""
dQ kernel
dK/dV kernel
autograd wrapper"""


"""Start with forward-only scaled dot-product attention.
Then add masking, causal mode, dropout optional.
Then backward pass.
Goal: understand tiling, SRAM reuse, online softmax, numerical stability, and memory bandwidth limits."""
  

"""q, k, v:  [32, 128, 1024]  

we have head = 16  

makes [10, 100, 16, 64]
and then transpose:   [32, 16, 128, 64]  b, hn, t, hd

q: [32, 16, 128, 64] 
k: [32, 16, 128, 64] 
v: [32, 16, 128, 64]  

blk_m =8 
blk_n = 8

load Q block  [blk_m, d]
   : its like [0...8, 64] then [8...16, 64] to [112...128, 64]  

same with loading k/v block:  
   : k/v [0...8, 64] then [8...16, 64] to [112...128, 64]  

    q[8, 64] @ k[8,64].T  
    log= softmax([8,8])  
    log[8,8] @ [8, 64]  
    = [8, 64] this is for Q block 0 to 16(128 / 8) means 16  tiles will be there for [8,64]  

parallelism is 
 batch * heads * query_blocks
 = 32 * 16 * (128 / 8)
 = 32 * 16 * 16
 = 8192 Triton programs 

one program:
    Q:      [8, 64]
    K tile: [8, 64]
    V tile: [8, 64]
    score:  [8, 8]
    acc:    [8, 64]
    output: [8, 64]

all programs together:
    O: [32, 16, 128, 64]

"""  

"""
for backward:
dV     = P.T @ dO

dP     = dO @ V.T

delta  = sum(dO * O, axis=-1)      ← per row dot product, shape [T]

dS     = P * (dP - delta[:, None])

dQ     = dS @ K / sqrt(D)

dK     = dS.T @ Q / sqrt(D) """


def torch_attention(q,k,v): 
     
     d = q.shape[-1]  
     scale = 1.0 / math.sqrt(d)   

     scores = q @ k.transpose(-1, -2)  
     scores = scores * scale  
     
     T = q.shape[-2] 
     mask = torch.ones((T, T), device=q.device, dtype=torch.bool).tril()
     scores = scores.masked_fill(~mask, float("-inf"))
     probs  = torch.softmax(scores, dim= -1)  
     out = probs @ v  

     lse = torch.logsumexp(scores, dim=-1)

     return out, lse

def torch_delta_ref(out, dout):
    return torch.sum(out.float() * dout.float(), dim=-1)


def torch_dq_ref(q, k, v, out, lse, dout):
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    qf = q.float()
    kf = k.float()
    vf = v.float()
    do = dout.float()

    scores = qf @ kf.transpose(-1, -2) * scale

    T = q.shape[-2]
    mask = torch.ones((T, T), device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))

    p = torch.exp(scores - lse.float()[..., None])
    dp = do @ vf.transpose(-1, -2)
    delta = torch.sum(out.float() * do, dim=-1, keepdim=True)
    ds = p * (dp - delta)

    dq = ds @ kf * scale
    return dq   


def torch_dkdv_ref(q, k, v, out, lse, dout):
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    qf = q.float()
    kf = k.float()
    vf = v.float()
    do = dout.float()

    scores = qf @ kf.transpose(-1, -2) * scale

    T = q.shape[-2]
    mask = torch.ones((T, T), device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))

    p = torch.exp(scores - lse.float()[..., None])

    dp = do @ vf.transpose(-1, -2)
    delta = torch.sum(out.float() * do, dim=-1, keepdim=True)
    ds = p * (dp - delta)

    dv = p.transpose(-1, -2) @ do
    dk = ds.transpose(-1, -2) @ qf * scale

    return dk, dv





def flash_attention_fwd(q, k ,v):  
     
          B, H ,T, D = q.shape 

          out  = torch.empty_like(q)  
          lse = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

          BLOCK_M = 8 
          BLOCK_N = 16   

          """launching grid where x axis is =  T / block_m and y = b * h (16,  [32 * 16]) = 8192 triton programs)"""   
          grid =  (  
                  triton.cdiv(T, BLOCK_M),        
                  B * H           
          ) 
          """ 
              Q[b][h][t][d] = Q_flat[b * stride0 + h * stride1 + t * stride2 + d * stride3]    
          
           [d]stride3 = 1              moving one step in d skips 1 element
           [t]stride2 = 64             moving one step in t skips 64 elements (one full d row)
           [h]stride1 = 128 * 64       moving one step in h skips 128*64 elements (one full t,d block)
           [b]stride0 = 16 * 128 * 64  moving one step in b skips 16*128*64 elements (one full h,t,d block)
              
              Q.stride(0)  →  16*128*64 = 131072   skip one batch
              Q.stride(1)  →  128*64   = 8192      skip one head
              Q.stride(2)  →  64                   skip one token
              Q.stride(3)  →  1                    skip one dimension
          """

          _flash_attention_fwd_kernel[grid] (
                  q, k, v, out, lse,
                  B, H, T, D, 
                  q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
                  k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                  v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                  out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                  lse.stride(0), lse.stride(1), lse.stride(2),
                  BLOCK_M = BLOCK_M, 
                  BLOCK_N= BLOCK_N, 
                  BLOCK_D = D,
          )

          return out, lse


def flash_attention_delta(out, dout):
      
     B, H, T, D = out.shape  

     delta = torch.empty((B, H, T), device= out.device, dtype=torch.float32) 
     BLOCK_M = 8
     BLOCK_D = D  

     grid = ( 
           triton.cdiv(T, BLOCK_M),  
           B * H  
     )   
     
     _delta_kernel[grid](
                  out, dout, delta, 
                  B, H, T, D,  
                  out.stride(0), out.stride(1), out.stride(2), out.stride(3),  
                  dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                  delta.stride(0), delta.stride(1), delta.stride(2), 
                  BLOCK_M = BLOCK_M, 
                  BLOCK_D = BLOCK_D,
     )
     return delta  


def flash_attention_bwd_dq(q, k, v, dout, lse, delta):   
     
      B, H ,T, D = q.shape  

      dq = torch.empty_like(q)  

      BLOCK_M = 8  
      BLOCK_N = 16 
      BLOCK_D = D  

      grid =  (
           triton.cdiv(T, BLOCK_M),  
           B * H  
      ) 

      _dq_kernel[grid] (  

        q, k, v, dout, lse, delta, dq,
        B, H, T, D,

        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),

        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,                
 
      )

      return dq   


def flash_attention_bwd_dkdv(q, k, v, dout, lse, delta):  

     B, H, T, D = q.shape 
     
     dk = torch.zeros_like(k)
     dv = torch.zeros_like(v)   

     BLOCK_M = 16
     BLOCK_N = 16  
     BLOCK_D = D  

     grid = (  
          triton.cdiv(T, BLOCK_M), 
          triton.cdiv(T, BLOCK_N), 
          B * H  
     )  

     _dkdv_kernel[grid] (  
        q, k, v, dout, lse, delta, dk, dv,
        B, H, T, D,

        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),

        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),

        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,

     ) 

     return dk, dv  
 

@triton.jit  
def _dkdv_kernel(

    q_ptr, k_ptr, v_ptr, dout_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    B, H, T, D,

    q_stride_b, q_stride_h, q_stride_t, q_stride_d,
    k_stride_b, k_stride_h, k_stride_t, k_stride_d,
    v_stride_b, v_stride_h, v_stride_t, v_stride_d,
    dout_stride_b, dout_stride_h, dout_stride_t, dout_stride_d,

    lse_stride_b, lse_stride_h, lse_stride_t,
    delta_stride_b, delta_stride_h, delta_stride_t,

    dk_stride_b, dk_stride_h, dk_stride_t, dk_stride_d,
    dv_stride_b, dv_stride_h, dv_stride_t, dv_stride_d,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,

):      


  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  pid_bh = tl.program_id(2)
  
  b = pid_bh // H  
  h = pid_bh % H

  q_start  = pid_m * BLOCK_M  
  kv_start = pid_n * BLOCK_N  

  offs_m = q_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
  offs_n = kv_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
  offs_d = tl.arange(0, BLOCK_D)  

  q_ptrs = (
        q_ptr
        + b * q_stride_b
        + h * q_stride_h
        + offs_m[:, None] * q_stride_t
        + offs_d[None, :] * q_stride_d
    )

  dout_ptrs = (
        dout_ptr
        + b * dout_stride_b
        + h * dout_stride_h
        + offs_m[:, None] * dout_stride_t
        + offs_d[None, :] * dout_stride_d
    )

  k_ptrs = (
        k_ptr
        + b * k_stride_b
        + h * k_stride_h
        + offs_n[:, None] * k_stride_t
        + offs_d[None, :] * k_stride_d
    )

  v_ptrs = (
        v_ptr
        + b * v_stride_b
        + h * v_stride_h
        + offs_n[:, None] * v_stride_t
        + offs_d[None, :] * v_stride_d
    )
  
  q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)        # [M, D]
  dout = tl.load(dout_ptrs, mask=offs_m[:, None] < T, other=0.0)  # [M, D]
  k = tl.load(k_ptrs, mask=offs_n[:, None] < T, other=0.0)        # [N, D]
  v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0) 

  lse_ptrs = (
        lse_ptr
        + b * lse_stride_b
        + h * lse_stride_h
        + offs_m * lse_stride_t
    )

  delta_ptrs = (
        delta_ptr
        + b * delta_stride_b
        + h * delta_stride_h
        + offs_m * delta_stride_t
    )

  lse = tl.load(lse_ptrs, mask=offs_m < T, other=0.0)        
  delta = tl.load(delta_ptrs, mask=offs_m < T, other=0.0)
    
  scale = 1.0 / tl.sqrt(D.to(dtype=tl.float32))
    
  scores = tl.dot(q, tl.trans(k)) * scale
    
  scores = tl.where(offs_m[:, None] < T, scores, -float("inf")) 
  scores = tl.where(offs_n[None, :] < T, scores, -float("inf")) 
  
  scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -float("inf"))

  p = tl.exp(scores - lse[:, None]) 

  dp = tl.dot(dout, tl.trans(v))               # [M, N]
  ds = p * (dp - delta[:, None]) 
  
  dv_partial = tl.dot(tl.trans(p).to(dout.dtype), dout)
  dk_partial = tl.dot(tl.trans(ds).to(q.dtype), q) * scale 

  dk_ptrs = (
        dk_ptr
        + b * dk_stride_b
        + h * dk_stride_h
        + offs_n[:, None] * dk_stride_t
        + offs_d[None, :] * dk_stride_d
    )

  dv_ptrs = (
        dv_ptr
        + b * dv_stride_b
        + h * dv_stride_h
        + offs_n[:, None] * dv_stride_t
        + offs_d[None, :] * dv_stride_d
    ) 

  mask_n_d = offs_n[:, None] < T

  tl.atomic_add(dk_ptrs, dk_partial, mask=mask_n_d)
  tl.atomic_add(dv_ptrs, dv_partial, mask=mask_n_d)

@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, dout_ptr, lse_ptr, delta_ptr, dq_ptr,
    B, H, T, D,

    q_stride_b, q_stride_h, q_stride_t, q_stride_d,
    k_stride_b, k_stride_h, k_stride_t, k_stride_d,
    v_stride_b, v_stride_h, v_stride_t, v_stride_d,
    dout_stride_b, dout_stride_h, dout_stride_t, dout_stride_d,
    dq_stride_b, dq_stride_h, dq_stride_t, dq_stride_d,

    lse_stride_b, lse_stride_h, lse_stride_t,
    delta_stride_b, delta_stride_h, delta_stride_t,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):  
       
   pid_m = tl.program_id(0)
   pid_bh = tl.program_id(1)  

   b = pid_bh // H  
   h = pid_bh % H  

   q_start = pid_m * BLOCK_M  

   offs_m = q_start + tl.arange(0, BLOCK_M)  
   offs_n = tl.arange(0, BLOCK_N) 
   offs_d = tl.arange(0, BLOCK_D)  

   q_ptrs = (  
        q_ptr + b * q_stride_b + h * q_stride_h + offs_m[:, None] * q_stride_t + offs_d[None, :] * q_stride_d
   )  
   
   dout_ptrs = (  
         dout_ptr + b * dout_stride_b + h * dout_stride_h + offs_m[:, None] * dout_stride_t + offs_d[None, :] * dout_stride_d
   )

   mask_m_d = offs_m[:, None] < T

   q = tl.load(q_ptrs, mask=mask_m_d, other=0.0)
   dout = tl.load(dout_ptrs, mask=mask_m_d, other=0.0) 

   lse_ptrs = (
        lse_ptr  + b * lse_stride_b + h * lse_stride_h + offs_m * lse_stride_t
    )

   delta_ptrs = (
        delta_ptr + b * delta_stride_b + h * delta_stride_h + offs_m * delta_stride_t
    ) 
   
   lse = tl.load(lse_ptrs, mask=offs_m < T, other=0.0)  
   delta = tl.load(delta_ptrs, mask=offs_m < T, other= 0.0)
   
   dq_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32) 

   scale = 1.0 / tl.sqrt(D.to(tl.float32))  

   for n_start in range(0, T, BLOCK_N):  
        
        n = n_start + offs_n  

        k_ptrs = ( 
             k_ptr
            + b * k_stride_b
            + h * k_stride_h
            + n[:, None] * k_stride_t
            + offs_d[None, :] * k_stride_d
            )
        
        k = tl.load(k_ptrs, mask=n[: , None] < T, other=0.0)  

        v_ptrs = (
            v_ptr
            + b * v_stride_b
            + h * v_stride_h
            + n[:, None] * v_stride_t
            + offs_d[None, :] * v_stride_d
        )

        v = tl.load(v_ptrs, mask=n[:, None] < T, other=0.0)

        scores = tl.dot(q, tl.trans(k) * scale) 

        scores = tl.where(n[None, :] < T, scores, -float("inf"))
        scores = tl.where(offs_m[:, None] >= n[None,:], scores, -float("inf"))  

        p = tl.exp(scores - lse[:, None]) 

        dp = tl.dot(dout, tl.trans(v)) 
        ds = p * (dp - delta[:, None])  

        dq_acc += tl.dot(ds.to(k.dtype), k) * scale

   dq_ptrs = (
        dq_ptr
        + b * dq_stride_b
        + h * dq_stride_h
        + offs_m[:, None] * dq_stride_t
        + offs_d[None, :] * dq_stride_d
    )

   tl.store(dq_ptrs, dq_acc, mask=offs_m[:, None] < T)     



@triton.jit
def _delta_kernel(
    out_ptr, dout_ptr, delta_ptr,
    B, H, T, D,
    out_stride_b, out_stride_h, out_stride_t, out_stride_d,
    dout_stride_b, dout_stride_h, dout_stride_t, dout_stride_d,
    delta_stride_b, delta_stride_h, delta_stride_t,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    t_start = pid_m * BLOCK_M

    offs_m = t_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask = offs_m[:, None] < T

    out_ptrs = (
        out_ptr + b * out_stride_b + h * out_stride_h
        + offs_m[:, None] * out_stride_t
        + offs_d[None, :] * out_stride_d
    )

    dout_ptrs = (
        dout_ptr + b * dout_stride_b + h * dout_stride_h
        + offs_m[:, None] * dout_stride_t
        + offs_d[None, :] * dout_stride_d
    )

    out  = tl.load(out_ptrs,  mask=mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=mask, other=0.0)

    # delta[i] = sum_d dO[i,d] * O[i,d]
    product = out.to(tl.float32) * dout.to(tl.float32)
    delta   = tl.sum(product, axis=1)   # shape [BLOCK_M]

    delta_ptrs = (
        delta_ptr + b * delta_stride_b + h * delta_stride_h
        + offs_m * delta_stride_t
    )

    tl.store(delta_ptrs, delta, mask=offs_m < T)
     


@triton.jit
def _flash_attention_fwd_kernel(   

      q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
      B, H, T, D, 
      q_stride_b, q_stride_h, q_stride_t, q_stride_d,
      k_stride_b, k_stride_h, k_stride_t, k_stride_d,
      v_stride_b, v_stride_h, v_stride_t, v_stride_d,
      out_stride_b, out_stride_h, out_stride_t, out_stride_d,  
      lse_stride_b, lse_stride_h, lse_stride_t,
      BLOCK_M: tl.constexpr, 
      BLOCK_N: tl.constexpr,
      BLOCK_D: tl.constexpr,  
    ):
    
    #for which output block this Triton program computes 

    pid_m = tl.program_id(0) # Q block 
    pid_bh = tl.program_id(1) #batch/head pair  

    b = pid_bh // H 
    h = pid_bh % H 
    
    q_start = pid_m * BLOCK_M 

    #offset vectors 

    offs_m = q_start + tl.arange(0, BLOCK_M) 
    offs_n = tl.arange(0, BLOCK_N) 
    offs_d = tl.arange(0, BLOCK_D)   

    # load Q block : [BLOCK_M , BLOCK_D]
    
    """
     offs_m[:, None] = [[16],   shape [8, 1]
                   [17],
                   [18],
                   ...
                   [23]]

    offs_d[None, :] = [[0,1,2,...,63]]   shape [1, 64]

    broadcast together → shape [8, 64]

    Q_ptrs[i][j] = Q_ptr + offs_m[i] * stride_qt + offs_d[j] * stride_qd
    
    """

    q_ptrs = ( 
             q_ptr + b * q_stride_b + h * q_stride_h + offs_m[: , None] * q_stride_t + offs_d[None, :] * q_stride_d 
             )  
    
    #masking. 

    q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0) 
    
    
    #online softmax state.  

    m = tl.full((BLOCK_M, ), -float("inf"), tl.float32)
    l = tl.full((BLOCK_M, ), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)  
    
    scale = 1.0 / tl.sqrt(D.to(dtype= tl.float32))  

    #loop over K/V blocks. 

    for n_start in range(0, T, BLOCK_N):  
        n = n_start + offs_n            
        
        k_ptrs = (  
               k_ptr + b * k_stride_b + h * k_stride_h + n[:, None] * k_stride_t + offs_d[None, :] * k_stride_d
        )
        
        k = tl.load(k_ptrs, mask= n[:, None] < T, other=0.0)  

        v_ptrs = (  
                 v_ptr + b * v_stride_b + h * v_stride_h + n[:, None] * v_stride_t + offs_d[None, :] * v_stride_d
        )
        
        v = tl.load(v_ptrs, mask = n[:, None] < T, other =0.0) 

        #scores: [BLOCK_m, BLOCK_n]  
        scores = tl.dot(q, tl.trans(k)) * scale 
        
        #mask invalid k positions 
        scores = tl.where(n[None, :] < T, scores, -float("inf"))
        scores = tl.where(offs_m[:, None] >= n[None, :], scores, -float("inf"))

        #online softmax update.  

        m_new = tl.maximum(m, tl.max(scores, axis = 1)) #[BLOCK_M]
        alpha = tl.exp(m - m_new) #[BLOCK_M]
        p = tl.exp(scores - m_new[:, None]) #[BLOCK_M, BLOCK_N]

        l_new = l * alpha + tl.sum(p, axis =1)  
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  #[BLOCK_M, BLOCK_D]

        m = m_new 
        l = l_new  

    #normalize accumulator  
    out = acc / l[:, None]                              #[BLOCK_M, BLOCK_D] 
    lse = m + tl.log(l)                                 #[BLOCK_M]
        
    #store output 

    out_ptrs = ( 
                out_ptr + b * out_stride_b + h * out_stride_h + offs_m[:, None] * out_stride_t + offs_d[None, :] * out_stride_d 
        )
    tl.store(out_ptrs, out,  mask=offs_m[:, None] < T)  

    lse_ptrs = (
                lse_ptr + b * lse_stride_b + h * lse_stride_h + offs_m * lse_stride_t
        )
    tl.store(lse_ptrs, lse, mask=offs_m < T)


class FLashAttention(torch.autograd.Function):  
     
      @staticmethod
      def forward(ctx, q, k ,v):  
           out, lse = flash_attention_fwd(q, k ,v) 
           ctx.save_for_backward(q, k ,v, out, lse)  

           return out  
       
      @staticmethod  
      def backward(ctx, dout):  
        q, k, v, out, lse = ctx.saved_tensors

        dout = dout.contiguous()

        delta = flash_attention_delta(out, dout)

        dq = flash_attention_bwd_dq(q, k, v, dout, lse, delta)
        dk, dv = flash_attention_bwd_dkdv(q, k, v, dout, lse, delta)

        return dq, dk, dv 

def flash_attention(q, k ,v):  
     return FLashAttention.apply(q, k ,v)   




if __name__ == "__main__":  
      
      torch.cuda.empty_cache()
      B, H, T, D = 32, 16, 128, 64  

      q = torch.randn((B, H, T, D), device="cuda", dtype=torch.float32) 
      k = torch.randn((B, H, T, D), device="cuda", dtype=torch.float32) 
      v = torch.randn((B, H, T, D), device="cuda", dtype=torch.float32)

      q.requires_grad_(True) 
      k.requires_grad_(True) 
      v.requires_grad_(True)
      
      
      opt = torch.optim.Adam([q, k, v], lr=1e-2)
      for i in range(10): 
       opt.zero_grad()
       out = flash_attention(q, k, v)
       loss = out.square().mean()
       print(loss)
       loss.backward()
       opt.step()   

      print(q.grad.shape)
      print(k.grad.shape)
      print(v.grad.shape) 
      
      """ref, ref_lse = torch_attention(q, k, v)
      out, lse = flash_attention_fwd(q, k, v) 
      dout = torch.randn_like(out)
      ref_delta = torch.sum(out.float() * dout.float(), dim=-1)
      delta = flash_attention_delta(out, dout)
      dq = flash_attention_bwd_dq(q, k, v, dout, lse, delta)
      dq_ref = torch_dq_ref(q, k, v, out, lse, dout)     
      
      dk, dv = flash_attention_bwd_dkdv(q, k, v, dout, lse, delta)
      dk_ref, dv_ref = torch_dkdv_ref(q, k, v, out, lse, dout)

      print("dk error:", torch.max(torch.abs(dk_ref - dk.float())))
      print("dv error:", torch.max(torch.abs(dv_ref - dv.float())))

       
      print(ref.shape, out.shape) 
      print(torch.max(torch.abs(ref - out))) 
      print(torch.max(torch.abs(ref_lse - lse)))
      print(torch.max(torch.abs(ref_delta - delta)))
      print(torch.max(torch.abs(dq - dq_ref)))


      torch.testing.assert_close(
            out.float(), 
            ref.float(), 
            atol=2e-2, 
            rtol = 2e-2,
      )

      torch.testing.assert_close(
            lse.float(),
            ref_lse.float(),
            atol=2e-2,
            rtol=2e-2,
      )

      torch.testing.assert_close(
            delta.float(),
            ref_delta.float(),
            atol=2e-2,
            rtol=2e-2,
      )

      torch.testing.assert_close(dq.float(), dq_ref.float(), atol=3e-2, rtol=3e-2)
      torch.testing.assert_close(dk.float(), dk_ref.float(), atol=3e-2, rtol=3e-2)  
      torch.testing.assert_close(dv.float(), dv_ref.float(), atol=3e-2, rtol=3e-2)"""
      
