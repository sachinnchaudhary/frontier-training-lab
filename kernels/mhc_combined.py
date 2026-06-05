import os

os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))

import torch

from kernels.mhc_merge import (
    torch_mhc_merge_ref,
    triton_mhc_merge,
    triton_mhc_merge_backward,
)
from kernels.mhc_route import (
    torch_mhc_route_ref,
    triton_mhc_route,
    triton_mhc_route_read_backward,
    triton_sinkhorn_backward,
)


def torch_mhc_route_merge_ref(
    x_streams,
    h_out,
    H_pre_raw,
    H_post_raw,
    H_res_raw,
    sinkhorn_iters=8,
    eps=1e-6,
):
    """
    Reference non-GEMM mHC path.

    x_streams:  [N, S, D]
    h_out:      [N, D]
    H_pre_raw:  [N, S]
    H_post_raw: [N, S]
    H_res_raw:  [N, S, S]

    returns:
      x_next:   [N, S, D]
    """

    _, H_post, H_res = torch_mhc_route_ref(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )
    return torch_mhc_merge_ref(x_streams, h_out, H_post, H_res)


def torch_mhc_route_merge_aux_ref(
    x_streams,
    h_out,
    H_pre_raw,
    H_post_raw,
    H_res_raw,
    sinkhorn_iters=8,
    eps=1e-6,
):
    """
    Reference path that returns the framework boundary h_in too.

    h_in is consumed by Layer_F outside this non-GEMM path.
    x_next is the residual highway output after h_out is merged back.
    """

    h_in, H_post, H_res = torch_mhc_route_ref(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )
    x_next = torch_mhc_merge_ref(x_streams, h_out, H_post, H_res)
    return h_in, x_next


def triton_mhc_route_merge(
    x_streams,
    h_out,
    H_pre_raw,
    H_post_raw,
    H_res_raw,
    sinkhorn_iters=8,
    eps=1e-6,
    block_d=128,
):
    """
    Triton non-GEMM mHC path.

    Route kernel:
      x_streams + raw route logits -> h_in, H_post, H_res

    Framework boundary:
      h_out is passed in as if produced by Layer_F(h_in)

    Merge kernel:
      x_streams + h_out + H_post + H_res -> x_next
    """

    _, H_post, H_res = triton_mhc_route(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        block_d=block_d,
    )
    return triton_mhc_merge(x_streams, h_out, H_post, H_res, block_d=block_d)


def triton_mhc_route_merge_aux(
    x_streams,
    h_out,
    H_pre_raw,
    H_post_raw,
    H_res_raw,
    sinkhorn_iters=8,
    eps=1e-6,
    block_d=128,
):
    h_in, H_post, H_res = triton_mhc_route(
        x_streams,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        block_d=block_d,
    )
    x_next = triton_mhc_merge(x_streams, h_out, H_post, H_res, block_d=block_d)
    return h_in, x_next


class _MhcRouteMergeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_streams,
        h_out,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters,
        eps,
        block_d,
    ):
        h_in, H_post, H_res = triton_mhc_route(
            x_streams,
            H_pre_raw,
            H_post_raw,
            H_res_raw,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            block_d=block_d,
        )
        x_next = triton_mhc_merge(
            x_streams,
            h_out,
            H_post,
            H_res,
            block_d=block_d,
        )

        ctx.save_for_backward(
            x_streams,
            h_out,
            H_pre_raw,
            H_post_raw,
            H_res_raw,
            H_post,
            H_res,
        )
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps
        ctx.block_d = block_d

        return h_in, x_next

    @staticmethod
    def backward(ctx, dh_in, dx_next):
        (
            x_streams,
            h_out,
            H_pre_raw,
            H_post_raw,
            H_res_raw,
            H_post,
            H_res,
        ) = ctx.saved_tensors

        if dh_in is None:
            dh_in = torch.zeros(
                (x_streams.shape[0], x_streams.shape[2]),
                device=x_streams.device,
                dtype=x_streams.dtype,
            )
        if dx_next is None:
            dx_next = torch.zeros_like(x_streams)

        dx_merge, dh_out, dH_post, dH_res = triton_mhc_merge_backward(
            x_streams,
            h_out,
            H_post,
            H_res,
            dx_next,
            block_d=ctx.block_d,
        )

        dx_route, dH_pre_raw, dH_post_raw = triton_mhc_route_read_backward(
            x_streams,
            H_pre_raw,
            H_post_raw,
            dh_in,
            dH_post,
            block_d=ctx.block_d,
        )
        dH_res_raw = triton_sinkhorn_backward(
            H_res_raw,
            dH_res,
            sinkhorn_iters=ctx.sinkhorn_iters,
            eps=ctx.eps,
        )

        dx_streams = dx_merge + dx_route

        return (
            dx_streams,
            dh_out,
            dH_pre_raw,
            dH_post_raw,
            dH_res_raw,
            None,
            None,
            None,
        )


def mhc_route_merge_autograd(
    x_streams,
    h_out,
    H_pre_raw,
    H_post_raw,
    H_res_raw,
    sinkhorn_iters=8,
    eps=1e-6,
    block_d=128,
):
    """
    Autograd-enabled Triton non-GEMM mHC path.

    returns:
      h_in:   [N, D]
      x_next: [N, S, D]
    """

    return _MhcRouteMergeFunction.apply(
        x_streams,
        h_out,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters,
        eps,
        block_d,
    )


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


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.float32

    print("correctness")
    N, S, D = 3, 4, 8
    x_streams = torch.randn((N, S, D), device=device, dtype=dtype)
    h_out = torch.randn((N, D), device=device, dtype=dtype)
    H_pre_raw = torch.randn((N, S), device=device, dtype=dtype)
    H_post_raw = torch.randn((N, S), device=device, dtype=dtype)
    H_res_raw = torch.randn((N, S, S), device=device, dtype=dtype)

    x_ref = torch_mhc_route_merge_ref(
        x_streams,
        h_out,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=8,
    )
    x_tri = triton_mhc_route_merge(
        x_streams,
        h_out,
        H_pre_raw,
        H_post_raw,
        H_res_raw,
        sinkhorn_iters=8,
        block_d=4,
    )

    print("x_ref:", x_ref.shape)
    print("x_tri:", x_tri.shape)
    print("max error:", (x_ref - x_tri).abs().max())

    print("\nautograd correctness")
    x_streams_ref = x_streams.detach().clone().requires_grad_(True)
    h_out_ref = h_out.detach().clone().requires_grad_(True)
    H_pre_ref = H_pre_raw.detach().clone().requires_grad_(True)
    H_post_ref = H_post_raw.detach().clone().requires_grad_(True)
    H_res_ref = H_res_raw.detach().clone().requires_grad_(True)

    h_ref, x_ref = torch_mhc_route_merge_aux_ref(
        x_streams_ref,
        h_out_ref,
        H_pre_ref,
        H_post_ref,
        H_res_ref,
        sinkhorn_iters=8,
    )
    dh = torch.randn_like(h_ref)
    dx = torch.randn_like(x_ref)
    ((h_ref * dh).sum() + (x_ref * dx).sum()).backward()

    x_streams_tri = x_streams.detach().clone().requires_grad_(True)
    h_out_tri = h_out.detach().clone().requires_grad_(True)
    H_pre_tri = H_pre_raw.detach().clone().requires_grad_(True)
    H_post_tri = H_post_raw.detach().clone().requires_grad_(True)
    H_res_tri = H_res_raw.detach().clone().requires_grad_(True)

    h_tri, x_tri = mhc_route_merge_autograd(
        x_streams_tri,
        h_out_tri,
        H_pre_tri,
        H_post_tri,
        H_res_tri,
        sinkhorn_iters=8,
        block_d=4,
    )
    ((h_tri * dh).sum() + (x_tri * dx).sum()).backward()

    print("h error:", (h_ref - h_tri).abs().max())
    print("x error:", (x_ref - x_tri).abs().max())
    print("x.grad error:", (x_streams_ref.grad - x_streams_tri.grad).abs().max())
    print("h_out.grad error:", (h_out_ref.grad - h_out_tri.grad).abs().max())
    print("H_pre.grad error:", (H_pre_ref.grad - H_pre_tri.grad).abs().max())
    print("H_post.grad error:", (H_post_ref.grad - H_post_tri.grad).abs().max())
    print("H_res.grad error:", (H_res_ref.grad - H_res_tri.grad).abs().max())

    print("\nbenchmark")
    for N in [512, 2048]:
        for S in [4, 8]:
            for D in [512, 1024]:
                x_streams = torch.randn((N, S, D), device=device, dtype=dtype)
                h_out = torch.randn((N, D), device=device, dtype=dtype)
                H_pre_raw = torch.randn((N, S), device=device, dtype=dtype)
                H_post_raw = torch.randn((N, S), device=device, dtype=dtype)
                H_res_raw = torch.randn((N, S, S), device=device, dtype=dtype)
                dh = torch.randn((N, D), device=device, dtype=dtype)
                dx = torch.randn((N, S, D), device=device, dtype=dtype)

                def torch_fwd_bwd(
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    dh,
                    dx,
                ):
                    x_streams = x_streams.detach().clone().requires_grad_(True)
                    h_out = h_out.detach().clone().requires_grad_(True)
                    H_pre_raw = H_pre_raw.detach().clone().requires_grad_(True)
                    H_post_raw = H_post_raw.detach().clone().requires_grad_(True)
                    H_res_raw = H_res_raw.detach().clone().requires_grad_(True)

                    h_in, x_next = torch_mhc_route_merge_aux_ref(
                        x_streams,
                        h_out,
                        H_pre_raw,
                        H_post_raw,
                        H_res_raw,
                        sinkhorn_iters=8,
                    )
                    loss = (h_in * dh).sum() + (x_next * dx).sum()
                    loss.backward()
                    return (
                        x_streams.grad,
                        h_out.grad,
                        H_pre_raw.grad,
                        H_post_raw.grad,
                        H_res_raw.grad,
                    )

                def triton_fwd_bwd(
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    dh,
                    dx,
                ):
                    x_streams = x_streams.detach().clone().requires_grad_(True)
                    h_out = h_out.detach().clone().requires_grad_(True)
                    H_pre_raw = H_pre_raw.detach().clone().requires_grad_(True)
                    H_post_raw = H_post_raw.detach().clone().requires_grad_(True)
                    H_res_raw = H_res_raw.detach().clone().requires_grad_(True)

                    h_in, x_next = mhc_route_merge_autograd(
                        x_streams,
                        h_out,
                        H_pre_raw,
                        H_post_raw,
                        H_res_raw,
                        sinkhorn_iters=8,
                        block_d=128,
                    )
                    loss = (h_in * dh).sum() + (x_next * dx).sum()
                    loss.backward()
                    return (
                        x_streams.grad,
                        h_out.grad,
                        H_pre_raw.grad,
                        H_post_raw.grad,
                        H_res_raw.grad,
                    )

                torch_ms = benchmark(
                    torch_mhc_route_merge_ref,
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    8,
                    1e-6,
                )
                triton_ms = benchmark(
                    triton_mhc_route_merge,
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    8,
                    1e-6,
                    128,
                )
                torch_fb_ms = benchmark(
                    torch_fwd_bwd,
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    dh,
                    dx,
                    warmup=3,
                    iters=20,
                )
                triton_fb_ms = benchmark(
                    triton_fwd_bwd,
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    dh,
                    dx,
                    warmup=3,
                    iters=20,
                )

                x_ref = torch_mhc_route_merge_ref(
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    sinkhorn_iters=8,
                )
                x_tri = triton_mhc_route_merge(
                    x_streams,
                    h_out,
                    H_pre_raw,
                    H_post_raw,
                    H_res_raw,
                    sinkhorn_iters=8,
                    block_d=128,
                )
                err = (x_ref - x_tri).abs().max()

                print(
                    f"N={N} "
                    f"S={S} "
                    f"D={D} "
                    f"torch_ms={torch_ms:.4f} "
                    f"triton_ms={triton_ms:.4f} "
                    f"torch_fb_ms={torch_fb_ms:.4f} "
                    f"triton_fb_ms={triton_fb_ms:.4f} "
                    f"speedup={torch_ms / triton_ms:.2f}x "
                    f"speedup_fb={torch_fb_ms / triton_fb_ms:.2f}x "
                    f"err={err}"
                )
