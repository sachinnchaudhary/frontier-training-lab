import os

os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import triton   
import torch  
import triton.language as tl 
  


"""
current corrected write =
(raw desired write - earlier contamination) / self coefficient


this is for triangular solve: 
i=0:
W[n,0] = U[n,0] / A[n,0,0]

i=1:
W[n,1] = (U[n,1] - A[n,1,0]*W[n,0]) / A[n,1,1]

i=2:
W[n,2] = (U[n,2] - A[n,2,0]*W[n,0] - A[n,2,1]*W[n,1]) / A[n,2,2]

i=3:
W[n,3] = (U[n,3] - A[n,3,0]*W[n,0] - A[n,3,1]*W[n,1] - A[n,3,2]*W[n,2]) / A[n,3,3]

i=4:
W[n,4] = (U[n,4] - A[n,4,0]*W[n,0] - A[n,4,1]*W[n,1] - A[n,4,2]*W[n,2] - A[n,4,3]*W[n,3]) / A[n,4,4]

"""


def torch_triangular_solve_ref(A, U):  

    """
    A; [N, C, C] lower-triangular correction matrix 
    U: [N, C] raw write delta 
    returns W: [N, C] corrected write  

    Solves A @ W = U for each independent system N.  
    
    """ 

    return torch.linalg.solve_triangular(A, U[..., None], upper=False).squeeze(-1) 



def torch_triangular_solve_backward_ref(A, U, dW):  
     
     W = torch_triangular_solve_ref(A, U)  

     dU = torch.linalg.solve_triangular(

          A.transpose(-1, -2), 
          dW[..., None], 
          upper= True,
     ).squeeze(-1)  

     dA = -dU[..., : , None] * W[..., None, :]  

     C= A.shape[-1]  
     mask = torch.tril(torch.ones((C, C), device=A.device, dtype=torch.bool))
     dA = torch.where(mask[None, :, :], dA, torch.zeros_like(dA))  

     return dA, dU  


@triton.jit  
def _triangular_solve_kernel(
 A_ptr, 
 U_ptr, 
 W_ptr, 
 C: tl.constexpr,
):  


   n = tl.program_id(0)  

   A_base = A_ptr + n * C * C  
   U_base = U_ptr + n * C 
   W_base = W_ptr + n * C  

   for i in range(C):  
       acc = tl.load(U_base + i)  

       for j in range(i):  
           a_ij = tl.load(A_base + i * C + j)  
           w_j = tl.load(W_base + j)
           acc -= a_ij * w_j  

       diag = tl.load(A_base + i * C + i )  
       w_i = acc / diag  

       tl.store(W_base + i, w_i)   

@triton.jit
def _triangular_solve_vec_kernel(
 A_ptr,
 U_ptr,
 W_ptr,
 C: tl.constexpr,
 BLOCK_C: tl.constexpr,
):
   n = tl.program_id(0)

   A_base = A_ptr + n * C * C
   U_base = U_ptr + n * C
   W_base = W_ptr + n * C

   offs = tl.arange(0, BLOCK_C)
   w_vec = tl.zeros((BLOCK_C,), tl.float32)

   for i in range(C):
       u_i = tl.load(U_base + i).to(tl.float32)
       row = tl.load(
           A_base + i * C + offs,
           mask=offs < i,
           other=0.0,
       ).to(tl.float32)

       correction = tl.sum(row * w_vec, axis=0)
       diag = tl.load(A_base + i * C + i).to(tl.float32)
       w_i = (u_i - correction) / diag

       w_vec = tl.where(offs == i, w_i, w_vec)

   tl.store(W_base + offs, w_vec, mask=offs < C)


@triton.jit
def _triangular_solve_blockn_kernel(
 A_ptr,
 U_ptr,
 W_ptr,
 N: tl.constexpr,
 C: tl.constexpr,
 BLOCK_N: tl.constexpr,
 BLOCK_C: tl.constexpr,
):
   pid = tl.program_id(0)

   offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
   offs_c = tl.arange(0, BLOCK_C)
   n_mask = offs_n < N

   w_mat = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)

   for i in range(C):
       u_i = tl.load(
           U_ptr + offs_n * C + i,
           mask=n_mask,
           other=0.0,
       ).to(tl.float32)

       row = tl.load(
           A_ptr + offs_n[:, None] * C * C + i * C + offs_c[None, :],
           mask=(n_mask[:, None]) & (offs_c[None, :] < i),
           other=0.0,
       ).to(tl.float32)

       correction = tl.sum(row * w_mat, axis=1)
       diag = tl.load(
           A_ptr + offs_n * C * C + i * C + i,
           mask=n_mask,
           other=1.0,
       ).to(tl.float32)
       w_i = (u_i - correction) / diag

       w_mat = tl.where(offs_c[None, :] == i, w_i[:, None], w_mat)

   tl.store(
       W_ptr + offs_n[:, None] * C + offs_c[None, :],
       w_mat,
       mask=(n_mask[:, None]) & (offs_c[None, :] < C),
   )


@triton.jit
def _triangular_solve_backward_blockn_kernel(
 A_ptr,
 W_ptr,
 dW_ptr,
 dA_ptr,
 dU_ptr,
 N: tl.constexpr,
 C: tl.constexpr,
 BLOCK_N: tl.constexpr,
 BLOCK_C: tl.constexpr,
):
   pid = tl.program_id(0)

   offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
   offs_c = tl.arange(0, BLOCK_C)
   n_mask = offs_n < N

   du_mat = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)

   for step in range(C):
       i = C - 1 - step

       dw_i = tl.load(
           dW_ptr + offs_n * C + i,
           mask=n_mask,
           other=0.0,
       ).to(tl.float32)

       col = tl.load(
           A_ptr + offs_n[:, None] * C * C + offs_c[None, :] * C + i,
           mask=(n_mask[:, None]) & (offs_c[None, :] > i) & (offs_c[None, :] < C),
           other=0.0,
       ).to(tl.float32)

       correction = tl.sum(col * du_mat, axis=1)
       diag = tl.load(
           A_ptr + offs_n * C * C + i * C + i,
           mask=n_mask,
           other=1.0,
       ).to(tl.float32)
       du_i = (dw_i - correction) / diag

       du_mat = tl.where(offs_c[None, :] == i, du_i[:, None], du_mat)

   tl.store(
       dU_ptr + offs_n[:, None] * C + offs_c[None, :],
       du_mat,
       mask=(n_mask[:, None]) & (offs_c[None, :] < C),
   )

   w_mat = tl.load(
       W_ptr + offs_n[:, None] * C + offs_c[None, :], 
       mask=(n_mask[:, None]) & (offs_c[None, :] < C),
       other = 0.0,
   ).to(tl.float32)
   
   for i in range(C):  
       du_i = tl.sum(
          tl.where(offs_c[None, :] == i, du_mat, 0.0),
          axis=1, 
       )

       dA_row = -du_i[:, None] * w_mat  
       dA_row = tl.where(offs_c[None, :] <= i, dA_row, 0.0)

       tl.store(
           dA_ptr + offs_n[:, None] * C * C + i * C + offs_c[None, :],
           dA_row,
           mask=(n_mask[:, None]) & (offs_c[None, :] < C),
       )



def triton_triangular_solve(A, U):  

     """  
     A: [N, C, C] contiguous CUDA tensor 
     U: [N, C] contiguous CUDA tensor 
     returns W: [N, C]  
     
     """

     assert A.is_cuda and U.is_cuda 
     assert A.ndim == 3
     assert A.shape[0] == U.shape[0]
     assert A.shape[1] == A.shape[2]
     assert A.shape[1] == U.shape[1] 
     
     A = A.contiguous()
     U = U.contiguous()
     
     N = A.shape[0]  
     C = A.shape[1]  

     W = torch.empty_like(U)  

     grid = (N,)  

     _triangular_solve_kernel[grid]( A, U, W, C,) 
     return W  


def triton_triangular_solve_vec(A, U):
     """
     A: [N, C, C] contiguous CUDA tensor
     U: [N, C] contiguous CUDA tensor
     returns W: [N, C]

     Keeps the solved W vector in registers and stores once at the end.
     """

     assert A.is_cuda and U.is_cuda
     assert A.ndim == 3
     assert U.ndim == 2
     assert A.shape[0] == U.shape[0]
     assert A.shape[1] == A.shape[2]
     assert A.shape[1] == U.shape[1]

     A = A.contiguous()
     U = U.contiguous()

     N = A.shape[0]
     C = A.shape[1]
     assert C <= 128

     W = torch.empty_like(U)
     grid = (N,)

     _triangular_solve_vec_kernel[grid](
        A,
        U,
        W,
        C,
        triton.next_power_of_2(C),
     )
     return W


def triton_triangular_solve_blockn(A, U, block_n=4):
     """
     A: [N, C, C] contiguous CUDA tensor
     U: [N, C] contiguous CUDA tensor
     returns W: [N, C]

     Solves BLOCK_N independent systems per Triton program.
     """

     assert A.is_cuda and U.is_cuda
     assert A.ndim == 3
     assert U.ndim == 2
     assert A.shape[0] == U.shape[0]
     assert A.shape[1] == A.shape[2]
     assert A.shape[1] == U.shape[1]

     A = A.contiguous()
     U = U.contiguous()

     N = A.shape[0]
     C = A.shape[1]
     assert C <= 128

     W = torch.empty_like(U)
     grid = (triton.cdiv(N, block_n),)

     _triangular_solve_blockn_kernel[grid](
        A,
        U,
        W,
        N,
        C,
        block_n,
        triton.next_power_of_2(C),
     )
     return W


def triton_triangular_solve_backward_blockn(A, W, dW, block_n=4):
     """
     A: [N, C, C] contiguous CUDA tensor
     W: [N, C] forward triangular-solve output
     dW: [N, C] upstream gradient
     returns:
       dA: [N, C, C]
       dU: [N, C]
     """

     assert A.is_cuda and W.is_cuda and dW.is_cuda
     assert A.ndim == 3
     assert W.ndim == 2
     assert dW.ndim == 2
     assert A.shape[0] == W.shape[0] == dW.shape[0]
     assert A.shape[1] == A.shape[2]
     assert A.shape[1] == W.shape[1] == dW.shape[1]

     A = A.contiguous()
     W = W.contiguous()
     dW = dW.contiguous()

     N = A.shape[0]
     C = A.shape[1]
     assert C <= 128

     dA = torch.empty_like(A)
     dU = torch.empty_like(W)
     grid = (triton.cdiv(N, block_n),)

     _triangular_solve_backward_blockn_kernel[grid](
        A,
        W,
        dW,
        dA,
        dU,
        N,
        C,
        block_n,
        triton.next_power_of_2(C),
     )
     return dA, dU


def triton_triangular_solve_bhvd(A, U):  
    
     """ 
     A: [B, H, Vd, C, C]
     U: [B, H, Vd, C]
     returns W: [B, H, Vd, C]

     """
    
     assert A.is_cuda and U.is_cuda 
     assert A.ndim == 5
     assert U.ndim == 4  
     

     B, H, Vd, C, C2 = A.shape 
     assert C == C2  
     assert U.shape == (B, H, Vd, C)  

     A_flat = A.contiguous().view(B * H * Vd, C, C)  
     U_flat = U.contiguous().view(B * H * Vd, C)  

     W_flat = triton_triangular_solve(A_flat, U_flat) 
     return W_flat.view(B, H, Vd, C) 


def triton_triangular_solve_vec_bhvd(A, U):
     """
     A: [B, H, Vd, C, C]
     U: [B, H, Vd, C]
     returns W: [B, H, Vd, C]
     """

     assert A.is_cuda and U.is_cuda
     assert A.ndim == 5
     assert U.ndim == 4

     B, H, Vd, C, C2 = A.shape
     assert C == C2
     assert U.shape == (B, H, Vd, C)

     A_flat = A.contiguous().view(B * H * Vd, C, C)
     U_flat = U.contiguous().view(B * H * Vd, C)

     W_flat = triton_triangular_solve_vec(A_flat, U_flat)
     return W_flat.view(B, H, Vd, C)


def triton_triangular_solve_blockn_bhvd(A, U, block_n=4):
     """
     A: [B, H, Vd, C, C]
     U: [B, H, Vd, C]
     returns W: [B, H, Vd, C]
     """

     assert A.is_cuda and U.is_cuda
     assert A.ndim == 5
     assert U.ndim == 4

     B, H, Vd, C, C2 = A.shape
     assert C == C2
     assert U.shape == (B, H, Vd, C)

     A_flat = A.contiguous().view(B * H * Vd, C, C)
     U_flat = U.contiguous().view(B * H * Vd, C)

     W_flat = triton_triangular_solve_blockn(A_flat, U_flat, block_n=block_n)
     return W_flat.view(B, H, Vd, C)


def triton_triangular_solve_backward_blockn_bhvd(A, W, dW, block_n=4):
     """
     A: [B, H, Vd, C, C]
     W: [B, H, Vd, C]
     dW: [B, H, Vd, C]
     returns:
       dA: [B, H, Vd, C, C]
       dU: [B, H, Vd, C]
     """

     assert A.is_cuda and W.is_cuda and dW.is_cuda
     assert A.ndim == 5
     assert W.ndim == 4
     assert dW.ndim == 4

     B, H, Vd, C, C2 = A.shape
     assert C == C2
     assert W.shape == (B, H, Vd, C)
     assert dW.shape == (B, H, Vd, C)

     A_flat = A.contiguous().view(B * H * Vd, C, C)
     W_flat = W.contiguous().view(B * H * Vd, C)
     dW_flat = dW.contiguous().view(B * H * Vd, C)

     dA_flat, dU_flat = triton_triangular_solve_backward_blockn(
        A_flat,
        W_flat,
        dW_flat,
        block_n=block_n,
     )
     return dA_flat.view(B, H, Vd, C, C), dU_flat.view(B, H, Vd, C)


class KimiTriangularSolve(torch.autograd.Function):
     @staticmethod
     def forward(ctx, A, U, block_n=4):
          A = A.contiguous()
          U = U.contiguous()

          if A.ndim == 3:
               W = triton_triangular_solve_blockn(A, U, block_n=block_n)
          elif A.ndim == 5:
               W = triton_triangular_solve_blockn_bhvd(A, U, block_n=block_n)
          else:
               raise ValueError("A must have shape [N,C,C] or [B,H,Vd,C,C]")

          ctx.save_for_backward(A, W)
          ctx.block_n = block_n
          return W

     @staticmethod
     def backward(ctx, dW):
          A, W = ctx.saved_tensors
          dW = dW.contiguous()

          if A.ndim == 3:
               dA, dU = triton_triangular_solve_backward_blockn(
                    A,
                    W,
                    dW,
                    block_n=ctx.block_n,
               )
          else:
               dA, dU = triton_triangular_solve_backward_blockn_bhvd(
                    A,
                    W,
                    dW,
                    block_n=ctx.block_n,
               )

          return dA, dU, None


def kimi_triangular_solve(A, U, block_n=4):
     return KimiTriangularSolve.apply(A, U, block_n)


def benchmark(fn, *args, warmup=10, iters =100):  

    for _ in range(warmup):  
        out = fn(*args) 
    torch.cuda.synchronize()  

    start = torch.cuda.Event(enable_timing=True) 
    end = torch.cuda.Event(enable_timing=True)  

    start.record()  

    for _ in range(iters):  
        out = fn(*args)  
    end.record()  

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  


def torch_triangular_solve_backward_given_w_ref(A, W, dW):
     dU = torch.linalg.solve_triangular(
          A.transpose(-1, -2),
          dW[..., None],
          upper=True,
     ).squeeze(-1)

     dA = -dU[..., :, None] * W[..., None, :]
     C = A.shape[-1]
     mask = torch.tril(torch.ones((C, C), device=A.device, dtype=torch.bool))
     dA = torch.where(mask[None, :, :], dA, torch.zeros_like(dA))
     return dA, dU
         


if __name__ == "__main__":  

  torch.manual_seed(0)  

  device = "cuda"  
  dtype = torch.float32  
  print("\nautograd check")
  N, C = 32, 16
  A = torch.randn((N, C, C), device=device, dtype=dtype) * 0.1
  A = torch.tril(A)
  A = A + torch.eye(C, device=device, dtype=dtype)[None, :, :]
  U = torch.randn((N, C), device=device, dtype=dtype)

  A_ref = A.detach().clone().requires_grad_(True)
  U_ref = U.detach().clone().requires_grad_(True)
  A_tri = A.detach().clone().requires_grad_(True)
  U_tri = U.detach().clone().requires_grad_(True)

  W_ref = torch_triangular_solve_ref(A_ref, U_ref)
  W_tri = kimi_triangular_solve(A_tri, U_tri, block_n=4)
  dW = torch.randn_like(W_ref)

  (W_ref * dW).sum().backward()
  (W_tri * dW).sum().backward()

  print("W error:", (W_ref.detach() - W_tri.detach()).abs().max())
  print("A.grad error:", (A_ref.grad - A_tri.grad).abs().max())
  print("U.grad error:", (U_ref.grad - U_tri.grad).abs().max())

  print("\nbenchmark")
  for N in [128, 512, 2048, 8192]:
   for C in [16, 32, 64]:  
    B, H, Vd = 1, 1, N

    A5 = torch.randn((B, H, Vd, C, C), device=device, dtype=dtype) * 0.1
    A5 = torch.tril(A5)
    eye = torch.eye(C, device=device, dtype=dtype)[None, None, None, :, :]
    A5 = A5 + eye

    U4 = torch.randn((B, H, Vd, C), device=device, dtype=dtype)

    A_flat = A5.reshape(B * H * Vd, C, C)
    U_flat = U4.reshape(B * H * Vd, C)

    torch_ms = benchmark(torch_triangular_solve_ref,A_flat, U_flat)
    tri_ms = benchmark(triton_triangular_solve_bhvd,A5, U4)
    vec_ms = benchmark(triton_triangular_solve_vec_bhvd, A5, U4) 
    blockn_ms = benchmark(triton_triangular_solve_blockn_bhvd, A5, U4, 4)

    W_vec = triton_triangular_solve_vec_bhvd(A5, U4)
    W_blockn = triton_triangular_solve_blockn_bhvd(A5, U4, 4)
    W_ref = torch_triangular_solve_ref(A_flat, U_flat).reshape(B, H, Vd, C) 
    W_tri = triton_triangular_solve_bhvd(A5, U4)  
    dW = torch.randn_like(W_ref)
    dW_flat = dW.reshape(B * H * Vd, C)
    torch_bwd_ms = benchmark(
        torch_triangular_solve_backward_given_w_ref,
        A_flat,
        W_ref.reshape(B * H * Vd, C),
        dW_flat,
    )
    triton_bwd_ms = benchmark(
        triton_triangular_solve_backward_blockn_bhvd,
        A5,
        W_blockn,
        dW,
        4,
    )
    dA_ref, dU_ref = torch_triangular_solve_backward_given_w_ref(
        A_flat,
        W_ref.reshape(B * H * Vd, C),
        dW_flat,
    )
    dA_tri, dU_tri = triton_triangular_solve_backward_blockn_bhvd(A5, W_blockn, dW, 4)
   
    err = (W_ref - W_tri).abs().max()  
    vec_err = (W_ref - W_vec).abs().max()  
    blockn_err = (W_ref - W_blockn).abs().max()
    dA_err = (dA_ref.reshape(B, H, Vd, C, C) - dA_tri).abs().max()
    dU_err = (dU_ref.reshape(B, H, Vd, C) - dU_tri).abs().max()

    print(
            f"C={C} "
            f"N={B*H*Vd} "
            f"torch_ms={torch_ms:.4f} "
            f"triton_ms={tri_ms:.4f} "
            f"vec_ms={vec_ms:.4f} "
            f"blockn4_ms={blockn_ms:.4f} "
            f"torch_bwd_ms={torch_bwd_ms:.4f} "
            f"triton_bwd_ms={triton_bwd_ms:.4f} "
            f"speedup={torch_ms / tri_ms:.2f}x "
            f"speedup_vec={torch_ms / vec_ms:.2f}x "
            f"speedup_blockn4={torch_ms / blockn_ms:.2f}x "
            f"speedup_bwd={torch_bwd_ms / triton_bwd_ms:.2f}x "
            f"err={err} "
            f"vec_err={vec_err} "
            f"blockn_err={blockn_err} "
            f"dA_err={dA_err} "
            f"dU_err={dU_err} "
        )
