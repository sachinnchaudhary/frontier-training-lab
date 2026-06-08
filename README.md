Frontier Training Lab Technical Report
Date: 2026-06-02


Idea: 
this repo contain not just educational implementation alones but it has advanced architectures, kernels and training runs with experiements.  Majority of things ran in Nvidia geforce rtx 2050 and some experiments ran on Nvidia 4090. 
 

Essence
=====

This report summarizes the work completed so far in the repository, from the
first Triton FlashAttention implementation through the JAX reference
architectures, JAX training setup, experiment runners, correctness cleanups, and
the first profiling results on an NVIDIA GeForce RTX 2050.
(it also has some standard pytorch training scripts and some standard attention variant with others like gqa and sliding window.)

it has implementations + experiments + training runs + profillng and then kernels designed specifically to those bottlenecks. 
so its not just discrete implementations of kernels but its otpimization in real sense.  
figures out that kimi triangular solve is bottleneck and got speedup: 2.28x  
mhc also hase lot of small ops apart from GEMM gaves also speedup: speedup=13.79x
and others kernels are deepseek_moe.  



Implementations + training runs.
==========================

The repo now contains several JAX architecture references:

1.1 MHLA / MLA Attention + DeepSeekMoE
--------------------------------------

File:
- model/mhlatent_attention.py

Implemented:
- Dense causal MHLA/MLA reference attention.
- Compressed query and KV latent streams.
- Content + RoPE query/key expansion.
- Dense causal attention over expanded latent-derived keys/values.
- Single-token decode helper.
- DeepSeek-style MoE with shared experts and routed experts.

Recent cleanup:
- Added explicit MHLA config/input/parameter validation.
- Fixed decode helper convention:
  - x_new is [B, 1, D].
  - past_kv_latent_cache is previous tokens only.
  - current KV latent is computed inside the function.
  - function returns output and updated cache.
- Added decode smoke test comparing full forward last token against step output.
- Added DeepSeekMoE validation for router and expert shapes.

Important correctness result:
- Decode step max error against full dense last token was around 1e-6 in the
  smoke test.


1.2 Kimi DeltaNet
-----------------

File:
- model/kimi_deltanet.py

Implemented:
- Stepwise DeltaNet recurrence.
- Chunk-scan DeltaNet recurrence.
- Parallel chunkwise DeltaNet algorithm.
- Gate modes: no gate, scalar gate, vector/fine-grained gate.
- Kimi DeltaNet + MoE block path.

Recent cleanup:
- Replaced token-level Python recurrence loops with jax.lax.scan.
- Shared the recurrence between stepwise and chunk-scan paths.
- Fixed vector-gate parallel chunkwise math:
  - Previous version collapsed vector beta using mean(beta).
  - New version solves one triangular system per value dimension.
- Added solve_triangular_per_value for vector-gated chunkwise correction.

Correctness result:
- scan max error: 0.0.
- parallel max error dropped to about 2.38e-07.

Profiling result:
- Kimi is the dominant bottleneck.
- Parallel chunkwise is much faster than stepwise/chunk-scan at small size, but
  medium preset causes severe XLA compile pressure.


1.3 DeepSeek Sparse Attention
-----------------------------

File:
- model/deepseek_sparseatt.py

Implemented:
- Compressed KV sparse attention.
- MQA-style compressed key/value path.
- Lightning indexer:
  - idx_q, idx_k, idx_w.
  - dense reference index score [B, T, T, index_heads].
  - top-k selected token indices.
- Sparse gather of selected compressed KV.
- Sparse MQA attention over selected KV.

Recent cleanup:
- Confirmed causal mask happens before top-k.
- Added selected-index causal mask after gather.
- Added selected_causal_valid helper.
- Added smoke check:
  - top-k can return invalid filler indices for early tokens.
  - invalid filler slots are masked before softmax.
- Added config, input, and parameter validation.

Important correctness result:
- top-k filler invalid count was nonzero, as expected.
- unmasked invalid count was 0, meaning invalid slots cannot affect attention.

Current limitation:
- The reference indexer still builds [B, T, T, index_heads], so it is not
  truly subquadratic yet. It is sparse in final attention but dense in indexer
  scoring.


1.4 DeepSeek CSA / HCA / Hybrid Attention
-----------------------------------------

File:
- model/deepseek_csa.py

Implemented:
- CSA compressed block sparse attention.
- HCA highly-compressed dense chunk + local token path.
- Hybrid CSA + HCA attention.
- DeepSeekMoE block integration.

Recent cleanup:
- Fixed compressed-block causality:
  - Previous mask used block_start <= token_pos.
  - New mask uses block_end <= token_pos.
  - This prevents a token from attending to a compressed block containing
    future tokens.
- Added selected top-k block validity masking.
- Added safe_masked_softmax for early tokens with zero valid completed blocks.
- Changed compressed block RoPE anchor from block start to block end.
- Added config, CSA/HCA param, input, MoE, and norm validation.

Important correctness result:
- After block-end masking, early all-invalid softmax originally produced NaNs.
- safe_masked_softmax removed the NaNs.
- Smoke test has finite loss and grad norm.

Profiling result:
- CSA scales better than MHLA and sparse at medium preset.
- HCA has fastest forward at medium preset, but backward remains heavier.
- CSA+HCA is roughly additive and slower than either alone.


1.5 mHC Residual Highway
------------------------

File:
- model/deepseek_mhc.py

Implemented:
- N-stream mHC highway.
- Dynamic route generation:
  - H_pre.
  - H_post.
  - H_res.
- Sinkhorn projection for doubly-stochastic residual mixing.
- Test layer and readout path.
- Tiny training smoke test.

Recent cleanup:
- Added config, parameter, and input validation.
- Made Sinkhorn numerically safer by subtracting max before exp.
- Replaced Sinkhorn Python loop with jax.lax.fori_loop.
- Added finite output checks and Sinkhorn row/column constraint checks.

Smoke result:
- row error around 1e-6.
- col error around 1e-5.
- short training loss decreased across steps.

Profiling result:
- mHC is slower than single attention paths and will compound across layers.
- It is not the first bottleneck compared with Kimi, but it is worth profiling
  deeper before stacking many blocks.


JAX Training Infrastructure
==============================

Folder:
- jax_training/

Implemented:
- data.py:
  - Uses existing cached data/tokenizer paths where possible.
  - Avoids requiring torch in the JAX data loader.
- model.py:
  - JAX language model wrapper.
  - Supports MHA, MHLA, Kimi DeltaNet, sparse attention, CSA/HCA, CSA/HCA+mHC,
    and MHA+mHC variants.
- train.py:
  - JAX training loop.
  - AdamW and custom Muon-style optimizer path.
  - Warmup/cosine schedule.
  - Train/eval logging.
  - JSONL output.

Purpose:
- Keep PyTorch baseline training intact.
- Build a separate JAX-native training pipeline for experimental architecture
  references.
- Later decide whether kernels should target JAX, PyTorch, Triton, TileLang, or
  some hybrid.



Experiments.  
===========================
  
This repo have two tpyes of experiments one is standard pytorch and second one is jax advanced architectures.  

Folder:
- experiment/

JAX experiment runners:
- experiment/deepseek_mla_latent_sweep/run.py
  - MLA latent_dim sweep plus MHA reference.
- experiment/kimi_deltanet_memory_sweep/run.py
  - state/key/value dim sweep.
  - chunk size sweep.
  - gate type sweep.
- experiment/deepseek_sparse_topk_sweep/run.py
  - sparse attention top_k sweep.
- experiment/csa_hca_compression_sweep/run.py
  - CSA compress rate and HCA compress rate sweep.
- experiment/mhc_depth_scaling_sweep/run.py
  - ordinary residual vs mHC residual across 4/8/12 layers.


pytorch experiments runners:  
-experiemnt/scaling_laws.py  
  - reproducing scaling laws.   
- experiement/optimizer_experiment.py  
  - optimizer compare between adamw and muon.  
- experiment/sliding_window_experiment.py  
  - runnning various slidning window attention.  
- experiment/gqa_experiment.py  
  - various group-query size experiment.  
- experiement/ffn_experiment.py  
  - comparison between gelu and swiglu in transformer ffnetwork.   




Profiling Infrastructure
===========================

Files:
- profiling/__init__.py
- profiling/jax_profile.py
- profiling/runs/jax_profile_summary.jsonl

Implemented:
- JAX timing profiler for reference models.
- Supports:
  - mhla
  - sparse
  - csa
  - hca
  - csa_hca
  - kimi_stepwise
  - kimi_chunkwise
  - kimi_parallel
  - mhc
- Measures:
  - compile_forward_ms
  - compile_train_ms
  - forward_ms_mean/min/max
  - train_ms_mean/min/max
  - tokens/sec forward
  - tokens/sec train
  - param count
  - loss
  - grad norm
  - backend/devices

Correct JAX profiling rules used:
- jax.jit wraps forward and forward+backward.
- First call measures compile+run separately.
- Later calls measure steady-state execution.
- block_until_ready is used so async GPU execution does not corrupt timing.


Profiling Results: Small Preset
==================================

Small preset:
- B = 1
- T = 128
- D = 256
- GPU = cuda:0

Results:

model             fwd_ms    train_ms   tok/s train   params
------------------------------------------------------------
mhla              2.05      4.62       27,702        155,648
sparse            2.12      3.98       32,198        156,160
csa               2.26      4.40       29,073        152,320
hca               2.26      5.36       23,895        119,040
csa_hca           3.60      8.43       15,180        271,360
kimi_stepwise     22.85     124.15     1,031         393,216
kimi_chunkwise    23.36     112.81     1,135         393,216
kimi_parallel     6.87      17.59      7,277         393,216
mhc               3.81      11.25      11,374        484,632

Small-preset interpretation:
- MHLA, sparse, CSA, and HCA are all in a similar runtime band.
- CSA+HCA is slower because it computes both paths.
- mHC is medium-cost because route generation and Sinkhorn add overhead.
- Kimi stepwise/chunkwise are far slower than other models.
- Kimi parallel is much better than Kimi stepwise/chunkwise, but still slower
  than attention references and has very high compile cost.


8. Profiling Results: Medium Preset
===================================

Medium preset:
- B = 1
- T = 256
- D = 512
- num_heads = 4
- latent_dim = 128
- rope_dim = 32
- top_k = 8
- chunk_size = 32
- csa_compress_rate = 16
- hca_compress_rate = 96
- local_window_size = 96
- mhc_streams = 6
- mhc_hidden_dim = 1024
- mhc_sinkhorn_iters = 12

Results before termination:

model             fwd_ms    train_ms   tok/s train   params
------------------------------------------------------------
mhla              7.06      15.64      16,368        622,592
sparse            7.40      15.15      16,897        755,712
csa               6.10      12.58      20,345        739,840
hca               5.72      14.99      17,079        475,648
csa_hca           9.68      27.87      9,185         1,215,488
kimi_stepwise     83.56     246.52     1,038         1,572,864
kimi_chunkwise    96.25     379.80     674           1,572,864

The run terminated while compiling/running Kimi parallel. XLA reported a slow
compile:
- "Compiling module input_reduce_fusion for GPU"
- One compile operation took about 2m31s.
- The process was terminated before the full all-model medium run completed.

Medium-preset interpretation:
- CSA is the best train-time result among completed attention paths.
- HCA has the fastest forward pass, but backward is heavier.
- Sparse is close to MHLA because its indexer is still dense [B, T, T, Ih].
- CSA/HCA begin showing the benefit of block compression at T=256.
- Kimi remains the dominant bottleneck.
- Kimi parallel is promising at small sizes but currently causes severe compile
  pressure at medium.



Current Bottleneck Ranking from profiling. 
=============================

Highest-priority bottleneck:
1. Kimi DeltaNet
   - Stepwise/chunkwise are far too slow.
   - Parallel chunkwise improves runtime but causes large XLA compile pressure.
   - Likely expensive components:
     - pairwise decay [B, H, C, C, Kd]
     - vector-gated triangular solve [B, H, Vd, C, C]
     - many small triangular systems

Second-priority bottleneck:
2. mHC residual
   - Not as bad as Kimi, but route generation + Sinkhorn + stream mixing will
     compound across layers.



kernels.  
========


====>  Kimi triangular solve Triton:
- Implemented a standalone PyTorch/Triton triangular solve kernel for the Kimi
  vector-gated correction solve.
- Mathematical target:
  - A @ W = U
  - A is the lower-triangular contamination/correction matrix.
  - U is the raw write delta.
  - W is the corrected write actually applied to memory.
- First version used one Triton program per independent solve:
  - A: [N, C, C]
  - U: [N, C]
  - W: [N, C]
  - N = B * H * Vd after flattening model dimensions.
- Scalar version was correct but repeatedly reloaded previous W[j] values from
  global memory.
- Vector/register version kept the solved W vector inside the Triton program
  using tl.arange/tl.zeros and stored once at the end.
- Benchmark result:
  - For small/medium N, Triton beats torch.linalg.solve_triangular.
  - For large N, PyTorch catches up or wins.
  - Example pattern:
    - N=128/512: Triton usually wins.
    - N=2048: vector Triton helps for C=16/32, loses for C=64.
    - N=8192: PyTorch is generally stronger.
- Obserevation:
  - The custom kernel improved memory behavior, not the sequential dependency.
  - The triangular dependency over chunk positions remains sequential.
  - A custom kernel has a useful operating region, but library kernels can win
    once batching is large enough.
- Added a BLOCK_N version:
  - One Triton program solves multiple independent triangular systems.
  - Local state changes from w_vec: [BLOCK_C] to w_mat: [BLOCK_N, BLOCK_C].
  - With BLOCK_N=4, the large-N weakness improved substantially.
  - Example observed wins:
    - C=64, N=2048: torch 0.4187 ms vs blockn4 0.2185 ms.
    - C=32, N=8192: torch 0.5998 ms vs blockn4 0.2518 ms.
    - C=64, N=8192: torch 1.5645 ms vs blockn4 0.7671 ms.
  - The win comes from better work granularity and vectorizing across
    independent systems, not from removing the triangular dependency.
- Next kernel idea:
  - Sweep BLOCK_N = 2, 4, 8.
  - Larger BLOCK_N may improve batching but can increase register pressure,
    especially when C=64.
- Added a PyTorch autograd wrapper:
  - Public entry point: kimi_triangular_solve(A, U, block_n=4).
  - Forward uses the BLOCK_N Triton solve.
  - Backward uses the fused Triton backward:
    - dU = solve(A.T, dW)
    - dA = -dU outer W, masked to the lower triangle.
  - Gradient check against torch.linalg.solve_triangular autograd passed:
    - W error around 4.8e-7.
    - A.grad error around 1.4e-6.
    - U.grad error around 7.2e-7.

====> mHC route/merge Triton:
- Split mHC around the natural framework boundary:
  - PyTorch/framework keeps GEMM-heavy routing projections and Layer_F.
  - Triton route/read kernel handles sigmoid, Sinkhorn-Knopp, and weighted
    stream read.
  - Triton merge kernel handles write + stream mixing.
- Route/read kernel:
  - Inputs:
    - x_streams: [N, S, D]
    - H_pre_raw: [N, S]
    - H_post_raw: [N, S]
    - H_res_raw: [N, S, S]
  - Outputs:
    - h_in: [N, D]
    - H_post: [N, S]
    - H_res: [N, S, S]
  - Computes:
    - H_pre = sigmoid(H_pre_raw)
    - H_post = 2 * sigmoid(H_post_raw)
    - H_res = Sinkhorn(exp(H_res_raw - max))
    - h_in = sum_s H_pre[s] * x_streams[s]
  - Correctness errors were around 1e-6 or lower.
  - Observed route-only speedups ranged from about 6.7x to 28.9x on tested
    sizes.
- Merge kernel:
  - Inputs:
    - x_streams: [N, S, D]
    - h_out: [N, D]
    - H_post: [N, S]
    - H_res: [N, S, S]
  - Output:
    - x_next: [N, S, D]
  - Computes:
    - x_next[n,s,d] = sum_r H_res[n,s,r] * x_streams[n,r,d]
      + H_post[n,s] * h_out[n,d]
  - Correctness error was around 5e-7 on the small test.
- Combined route+merge benchmark:
  - Added kernels/mhc_combined.py for the non-GEMM mHC path.
  - Combined speedups ranged from about 3.3x to 13.7x on tested sizes.
  - Best observed combined example:
    - N=512, S=4, D=512: about 13.7x speedup.
  - Larger S/D cases still showed meaningful speedups, commonly 3x-8x.
- Obserevation:
  - This is a high-value kernel target because it fuses awkward small-matrix
    routing, Sinkhorn normalization, and stream read/write operations while
    leaving large GEMMs in the framework.
- Route backward and Sinkhorn backward:
  - Added a split backward path:
    - read/post backward kernel handles dx_streams, dH_pre_raw, dH_post_raw.
    - Sinkhorn backward kernel handles dH_res_raw.
  - Read/post backward correctness:
    - dx error around 5e-7 to 7e-7.
    - dH_pre_raw error around 3e-6 to 8e-6.
    - dH_post_raw error around 3e-7 to 5e-7.
  - Sinkhorn backward correctness:
    - manual backward vs PyTorch autograd: around 6e-8.
    - Triton Sinkhorn backward vs manual backward: around 2e-7 to 4e-7.
  - Observed route/read forward speedups in the latest sweep:
    - about 3.8x to 25.1x across N={512,2048}, S={4,8}, D={512,1024}.
  - Observed read/post backward speedups:
    - about 2.0x to 3.0x on most tested sizes.
  - Observed Sinkhorn backward speedups:
    - about 18x to 61x on tested sizes.
  - Main lesson:
    - Sinkhorn backward is an especially strong custom-kernel target because
      it is small-matrix, iterative, and framework eager/autograd overhead is
      large relative to the actual math.
- Autograd-enabled combined route+merge:
  - Added kernels/mhc_combined.py as the full non-GEMM mHC wrapper.
  - Public autograd entry point returns:
    - h_in: [N, D]
    - x_next: [N, S, D]
  - Returning h_in is important because Layer_F remains outside the fused
    non-GEMM path and must still receive gradients normally.
  - Backward composes:
    - merge backward for dx_streams, dh_out, dH_post, dH_res.
    - route/read backward for dx_streams, dH_pre_raw, dH_post_raw.
    - Sinkhorn backward for dH_res_raw.
  - Correctness:
    - forward error around 2.4e-7.
    - x_streams grad error around 2.4e-7.
    - h_out grad error around 4.8e-7.
    - H_pre/H_post/H_res raw grad errors around 6e-8 to 1.8e-7.
  - Latest observed forward speedups:
    - N=512, S=4, D=512: 10.36x.
    - N=512, S=4, D=1024: 12.17x.
    - N=512, S=8, D=512: 9.12x.
    - N=2048, larger S/D cases: about 3.24x to 4.96x.
  - Latest observed forward+backward speedups:
    - Best small/mid case: 5.87x.
    - Larger cases stayed positive, about 1.88x to 3.56x.

====> DeepSeekMoE combine Triton:

- Added a non-GEMM MoE combine kernel in kernels/deepseek_moe_combine.py.
- Scope:
  - Framework/PyTorch still owns expert MLP GEMMs.
  - Triton owns top-2 router selection, softmax over selected logits,
    weighted routed expert combine, and shared expert add.
- Forward inputs:
  - router_logits: [N, E]
  - expert_outputs: [N, E, D]
  - shared_out: [N, D]
- Forward outputs:
  - out: [N, D]
  - top_indices: [N, 2]
  - router_weights: [N, 2]
- Backward outputs:
  - drouter_logits: [N, E]
  - dexpert_outputs: [N, E, D]
  - dshared_out: [N, D]
- Correctness:
  - forward error around 1e-7 to 1e-6.
  - top-index match was exact.
  - router weight error around 3e-8 to 1e-7.
  - small backward check:
    - drouter error around 2.2e-7.
    - dexpert error around 6e-8.
    - dshared error 0.
- Latest observed speedups on RTX 2050-safe sizes:
  - Forward: mostly about 2.7x to 4.5x.
  - Backward: about 3.2x to 11x on runnable PyTorch-reference cases.
  - Larger backward reference cases were skipped because PyTorch autograd
    became memory-heavy on the 4 GB GPU.
- Main lesson:
  - MoE routing/combine is a strong non-GEMM kernel target because it fuses
    top-k selection, selected softmax, indexed expert-output loads, weighted
    combine, and backward scatter into a small number of kernels.










1. Baseline Context
===================

The original PyTorch baseline training setup is in model/training.py, with
notes captured in baseline run.txt.

Baseline run notes:
- Device: CUDA.
- Batch size: 32.
- Sequence length: 128.
- Learning rate: 3e-4.
- Actual vocab size: 4204.
- Model size: 1,459,820 trainable parameters.
- Warm throughput was usually around 70k-78k tokens/sec.


And then there is jax_training from which you can do training run with different architecture of jax.  
python -m jax_training.train --architecture csa_hca_moe
python -m jax_training.train --architecture kimi_deltanet_moe
python -m jax_training.train --architecture deepseek_sparse_moe
python -m jax_training.train --architecture deepseek_mhla_moe  

2. Triton FlashAttention Work
=============================

File:
- kernels/flash_attention.py

Implemented:
- Forward-only scaled dot-product attention in Triton.
- Causal masking.
- Online softmax with running max and running denominator.
- LSE/log-sum-exp storage for backward.
- Delta kernel for dO * O reduction.
- dQ kernel.
- dK/dV kernel.
- PyTorch autograd wrapper using torch.autograd.Function.





10. Lessons Learned
===================

JAX reference implementation lessons:
- JAX forces explicit math:
  - params are pytrees
  - RNG is explicit
  - loss/grad/update are explicit
  - shape assumptions must be written down
- This makes it good for algorithmic research but less hidden than PyTorch.

Profiling lessons:
- JAX execution is async, so block_until_ready is mandatory.
- First call includes compilation; steady-state timing must be measured after
  warmup.
- "More parallel" code is not automatically better:
  - Kimi parallel is faster at small runtime but has severe compile pressure at
    medium size.
- Correctness cleanup exposed performance-relevant behavior:
  - CSA block-start masking was faster-looking but causally wrong.
  - block-end masking required safe softmax for early tokens.



IMportant commands of repo: 
JAX experiments:  
 - python -m experiment.deepseek_mla_latent_sweep.run --runs small medium large mha
 - python -m experiment.kimi_deltanet_memory_sweep.run --mode pilot
 - python -m experiment.kimi_deltanet_memory_sweep.run --mode full --runs all 
 - python -m experiment.deepseek_sparse_topk_sweep.run --mode pilot
 - python -m experiment.deepseek_sparse_topk_sweep.run --mode full --runs topk4 topk8 topk16 topk32
 - python -m experiment.csa_hca_compression_sweep.run --mode pilot
 - python -m experiment.csa_hca_compression_sweep.run --mode full
 - python -m experiment.mhc_depth_scaling_sweep.run --mode pilot
 - python -m experiment.mhc_depth_scaling_sweep.run --mode full

Jax profiling:  
 - python -m profiling.jax_profile --model all --preset small 
 - python -m profiling.jax_profile --model sparse --preset small --trace --trace-mode train --iters 3 
 - opening tensorboard: tensorboard --logdir "$(pwd)/profiling/traces" --host 0.0.0.0 --port 6006 --load_fast=false  


Triton kernels:  
 - python -m kernels.flash_attention 
 - python -m kernels.kimi_triangular_solve 
 - python -m kernels.mhc_route 
 - python -m kernels.mhc_merge 
 - python -m kernels.mhc_combined 
 - python -m kernels.deepseek_moe_combine 

Jax implementation:  
 - python -m model.mhlatent_attention
 - python -m model.kimi_deltanet 
 - python -m model.deepseek_sparseatt 
 - python -m model.deepseek_sparseatt 
 - python -m model.deepseek_csa 
 - python -m model.deepseek_mhc 


Pytorch training:  
 - python -m model.training

Pytorch experiments:  
 - python -m experiment.gqa_experiment
 - python -m experiment.ffn_experiment
 - python -m experiment.optimizer_experiment
 - python -m experiment.sliding_window_experiment
 - python -m experiment.scaling_laws