# EdgeForge benchmark report: DESKTOP-IH5KR28

```
date      : 2026-10-31T22:53:24+00:00
edgeforge : EdgeForge 0.8.0 (stage 8), git bf9bd32 (uncommitted changes)
model     : /home/manas/edgeforge-data/tinyllama
efq       : /home/manas/edgeforge-data/tinyllama-q4.efq (714M), sha256 83c742061eea567a...
```

## System

```
$ /home/manas/ef-build/edgeforge sysinfo
EdgeForge 0.8.0 (stage 8) - system
CPU      : AMD Ryzen 5 7530U with Radeon Graphics
Cores    : 6 physical / 12 logical | OS Linux 6.18.54.1-microsoft-standard-WSL2
RAM      : 7.42 GiB total, 6.77 GiB available
Features : SSE4.2 AVX AVX2 FMA F16C
Build    : gcc 15.2.0 | Release | flags: -O3 -DNDEBUG -march=native -ffp-contract=off
```

## Memory read bandwidth (membench)

```
$ /home/manas/ef-build/edgeforge membench
EdgeForge 0.8.0 (stage 8) - memory read bandwidth (STREAM-style, software-measured)
CPU      : AMD Ryzen 5 7530U with Radeon Graphics
Cores    : 6 physical / 12 logical | OS Linux 6.18.54.1-microsoft-standard-WSL2
RAM      : 7.42 GiB total, 6.77 GiB available
Features : SSE4.2 AVX AVX2 FMA F16C
Build    : gcc 15.2.0 | Release | flags: -O3 -DNDEBUG -march=native -ffp-contract=off
Setup    : 1.00 GiB working set | 5 timed runs x 3 passes | 1 warm-up run(s)

  threads   best GB/s   median GB/s   worst GB/s   checksum
        1       14.86         13.84        12.18   OK
        2       19.42         18.06        16.45   OK
        4       18.13         17.34        16.51   OK
        6       16.43         15.04        12.90   OK
        8       16.96         16.74        15.68   OK
       12       17.63         17.46        16.98   OK

Peak median read bandwidth: 18.06 GB/s at 2 threads  <- denominator for bandwidth efficiency
Note: achievable software read bandwidth, not a DRAM hardware counter; expect it below the
      RAM's spec-sheet number. Run plugged in, performance power mode, apps closed.
```

## Quantized file vs the BF16 checkpoint (verify-quant)

```
$ /home/manas/ef-build/edgeforge verify-quant --qmodel /home/manas/edgeforge-data/tinyllama-q4.efq --model-dir /home/manas/edgeforge-data/tinyllama --golden tests/golden
EdgeForge verify-quant: /home/manas/edgeforge-data/tinyllama-q4.efq vs /home/manas/edgeforge-data/tinyllama
  [PASS] file structure valid: 201 tensors, 64-byte aligned, no overlap, sizes match types, every type explained by the policy
  [PASS] embedded config.json matches the source checkpoint
  settings: layers Q4_0, lm_head Q8_0, embed BF16, norms F32, block 32, scale mse-search
  policy   : tinyllama-1.1b-chat-v1.0
  override : model.layers.0.self_attn.k_proj.weight -> Q8_0
  override : model.layers.0.self_attn.v_proj.weight -> Q8_0
  override : model.layers.1.self_attn.v_proj.weight -> Q8_0
  override : model.layers.2.self_attn.v_proj.weight -> Q8_0
  override : model.layers.17.self_attn.v_proj.weight -> Q8_0
  override : model.layers.18.self_attn.v_proj.weight -> Q8_0
  [PASS] lossless tensors (embed/norms) bit-identical to the source: 46/46
  [PASS] no NaN/Inf after dequantization
  [PASS] per-tensor error stored in the header matches recomputation: 201/201
  [PASS] deterministic: 2541/2541 re-quantized sample rows byte-identical to the file

  type   tensors         params         size   rel RMSE (all)        worst  worst tensor
  BF16         1     65,536,000    125.0 MiB       0.0000e+00   0.0000e+00  model.embed_tokens.weight
  F32         45         92,160    360.0 KiB       0.0000e+00   0.0000e+00  model.norm.weight
  Q4_0       148    965,738,496    518.1 MiB       8.4352e-02   1.1707e-01  model.layers.0.self_attn.q_proj.weight
  Q8_0         7     68,681,728     69.6 MiB       5.8220e-03   8.9601e-03  model.layers.0.self_attn.k_proj.weight

  largest Q4_0 weight errors (bound 0.12):
    0.1171  model.layers.0.self_attn.q_proj.weight
    0.0973  model.layers.1.self_attn.k_proj.weight
    0.0924  model.layers.1.self_attn.q_proj.weight
    0.0914  model.layers.2.self_attn.k_proj.weight
    0.0907  model.layers.15.self_attn.v_proj.weight

  [PASS] Q4_0 reconstruction: worst tensor rel RMSE 0.1171 <= 0.12 (model.layers.0.self_attn.q_proj.weight)
  [PASS] Q8_0 reconstruction: worst tensor rel RMSE 0.0090 <= 0.01 (model.layers.0.self_attn.k_proj.weight)

  size on disk : 2.05 GiB -> 713.0 MiB  (2.94x smaller)
  read / token : 1.93 GiB -> 588.0 MiB  (3.36x less weight traffic per decoded token; decode speed follows
                 it when decoding is limited by memory traffic - measured in Stages 6/7)
  overrides    : +1.5 MiB of tensor data vs the uniform policy (+0.211% file, +0.256% read/token)
  (full reconstruction scan: 1.1 s, 12 threads)

  output error with quantized weights on real activations (golden):
    layer 0 vs the PyTorch layer-0 outputs (p0_factual, all positions):
      self_attn.q_proj   Q4_0  rel err 0.0177  cosine 0.999782
      self_attn.k_proj   Q8_0  rel err 0.0010  cosine 1.000000
      self_attn.v_proj   Q8_0  rel err 0.0049  cosine 0.999811
      self_attn.o_proj   Q4_0  rel err 0.0214  cosine 0.999232
      mlp.gate_proj      Q4_0  rel err 0.0407  cosine 0.998801
      mlp.up_proj        Q4_0  rel err 0.0341  cosine 0.997875
      mlp.down_proj      Q4_0  rel err 0.0356  cosine 0.998139
  [PASS] layer-0 GEMV outputs keep cosine >= 0.99 vs PyTorch (worst 0.997875, mlp.up_proj)

    activations for every layer, re-derived from the golden residual stream (exact BF16 weights):
      p0_factual     23 positions -> inputs of all 7 projections
      p1_technical   last position -> inputs of q/k/v
      p2_code        last position -> inputs of q/k/v
      p3_hindi       last position -> inputs of q/k/v
  [PASS] re-derived activations reproduce the golden layer outputs (22 layers x 23 positions: max rel err 4.02e-06 <= 1e-03, cosine 1.000000)

    154 layer matrices, worst 10 by GEMV cosine (min over prompts; 'worst pos' = single position):
      tensor                                     type      cosine    rel L2  worst pos  prompt
      model.layers.19.self_attn.v_proj.weight    Q4_0    0.990002    0.1424   0.908686  p3_hindi
      model.layers.16.self_attn.v_proj.weight    Q4_0    0.992453    0.1227   0.966953  p1_technical
      model.layers.8.self_attn.v_proj.weight     Q4_0    0.992735    0.1207   0.986366  p0_factual
      model.layers.15.self_attn.v_proj.weight    Q4_0    0.993225    0.1163   0.979565  p2_code
      model.layers.12.self_attn.v_proj.weight    Q4_0    0.993235    0.1163   0.980182  p1_technical
      model.layers.21.self_attn.v_proj.weight    Q4_0    0.993276    0.1165   0.970544  p2_code
      model.layers.11.self_attn.v_proj.weight    Q4_0    0.993775    0.1119   0.979881  p2_code
      model.layers.7.self_attn.v_proj.weight     Q4_0    0.993800    0.1118   0.945368  p2_code
      model.layers.20.self_attn.v_proj.weight    Q4_0    0.994047    0.1093   0.871466  p0_factual
      model.layers.9.self_attn.v_proj.weight     Q4_0    0.994061    0.1089   0.968723  p2_code
      worst per projection over all layers: q_proj 0.9987 k_proj 0.9992 v_proj 0.9900 o_proj 0.9945 gate_proj 0.9975 up_proj 0.9969 down_proj 0.9965
  [PASS] layer matrices keep GEMV cosine >= 0.99 on every prompt: 154/154 measured pass (worst 0.990002, model.layers.19.self_attn.v_proj.weight)
      note: the bound is per matrix and per prompt, over all of that prompt's positions together.
            It is not a per-position guarantee ('worst pos' can be lower) and not an end-to-end
            one: errors of 154 matrices compound through 22 layers (measured in Stage 6).
      (activation checks: 5.6 s)

  overrides vs the uniform policy (same tensor, both precisions measured):
    tensor                                     type           weight rel RMSE     GEMV cosine (min)       extra
    model.layers.0.self_attn.k_proj.weight     Q4_0 -> Q8_0   0.1256 -> 0.0090    0.999873 -> 1.000000    +256.0 KiB
    model.layers.0.self_attn.v_proj.weight     Q4_0 -> Q8_0   0.0870 -> 0.0056    0.978525 -> 0.999811    +256.0 KiB
    model.layers.1.self_attn.v_proj.weight     Q4_0 -> Q8_0   0.0883 -> 0.0057    0.947104 -> 0.999332    +256.0 KiB
    model.layers.2.self_attn.v_proj.weight     Q4_0 -> Q8_0   0.0849 -> 0.0055    0.935080 -> 0.999496    +256.0 KiB
    model.layers.17.self_attn.v_proj.weight    Q4_0 -> Q8_0   0.0871 -> 0.0057    0.988928 -> 0.999961    +256.0 KiB
    model.layers.18.self_attn.v_proj.weight    Q4_0 -> Q8_0   0.0948 -> 0.0061    0.986160 -> 0.999945    +256.0 KiB

    lm_head p0_factual   Q8_0  logits rel err 0.0031  cosine 0.999992
    lm_head p1_technical Q8_0  logits rel err 0.0032  cosine 0.999991
    lm_head p2_code      Q8_0  logits rel err 0.0036  cosine 0.999990
    lm_head p3_hindi     Q8_0  logits rel err 0.0048  cosine 0.999991
  [PASS] lm_head (Q8_0): next-token top-1 identical on 4/4 prompts, top-5 overlap 20/20

RESULT: PASS  (12 passed, 0 failed)
```

## Quality and speed of the quantized engine (eval-quant)

```
$ /home/manas/ef-build/edgeforge eval-quant --qmodel /home/manas/edgeforge-data/tinyllama-q4.efq --model-dir /home/manas/edgeforge-data/tinyllama --golden tests/golden --text tests/fixtures/eval_text.txt
EdgeForge eval-quant: the engine on packed weights (avx2 kernels, 6 threads)
  model : efq, policy tinyllama-1.1b-chat-v1.0: 148 Q4_0 + 7 Q8_0 matrices, embedding BF16
  file  : /home/manas/edgeforge-data/tinyllama-q4.efq (713.0 MiB), 588.0 MiB of weights read per token
  [PASS] packed Q4_0/Q8_0 dot (avx2) = dequantize-then-dot on the real weights: 2480 sampled rows, max rel err 1.8e-08 <= 1e-05 (model.layers.20.self_attn.k_proj)

  next-token distributions vs the PyTorch FP32 reference (teacher-forced), greedy continuation:
    p0_factual    25 positions: top-1 22/25, confident top-1 19/19, KL mean 0.1887 max 3.0568
                  greedy 8 tokens identical: "The capital of France is Paris."
    p1_technical   3 positions: top-1 3/3, confident top-1 3/3, KL mean 0.0201 max 0.0444
                  greedy 8 tokens identical: "A CPU cache, also known as a"
    p2_code        3 positions: top-1 3/3, confident top-1 3/3, KL mean 0.0693 max 0.2074
                  greedy 8 tokens identical: "Here's a Python function that checks"
    p3_hindi       3 positions: top-1 3/3, confident top-1 2/2, KL mean 0.0672 max 0.2010
                  greedy 8 tokens identical: "भारत की "

  text: eval_text.txt, 128 tokens, teacher-forced, every prediction scored
    quantized engine: PPL 16.591, 20.10 forward passes/s
    BF16 engine     : PPL 14.892, 8.31 forward passes/s
    per position vs BF16: top-1 113/127, confident top-1 54/55, KL mean 0.0707 max 3.0569
    perplexity: 14.892 -> 16.591 (+11.41%)

  speed (avx2 kernels, 6 threads; the full ladder: edgeforge bench-decode):
    BF16 engine     : 8.31 tok/s, 1.93 GiB/token -> 17.20 GB/s of weights
    quantized engine: 20.10 tok/s, 588.0 MiB/token -> 12.39 GB/s of weights (2.42x the BF16 tok/s)
  memory: quantized engine peak RSS 632.4 MiB (measured before the BF16 reference is loaded)

  all compared positions: 161 (34 golden vs PyTorch, 127 text vs BF16 engine)
    top-1 agreement 144/161 = 89.4% (reported, not a target: near-ties flip under any noise)
  [PASS] no NaN/Inf logits (161 positions)
  [PASS] mean KL(reference || quantized) 0.0880 <= 0.10 nats per position (max 3.0569)
  [PASS] top-1 agreement where the reference is confident (p >= 0.5): 81/82 = 98.8% >= 95%
  [PASS] text perplexity ratio 1.1141 <= 1.15 (14.892 -> 16.591)
    greedy continuations identical to PyTorch: 4/4 prompts (reported, not a target)

RESULT: PASS  (5 passed, 0 failed)
```

## Speed-up ladder, 4-bit engine (bench-decode)

```
$ /home/manas/ef-build/edgeforge bench-decode --qmodel /home/manas/edgeforge-data/tinyllama-q4.efq
EdgeForge bench-decode: efq, policy tinyllama-1.1b-chat-v1.0: 148 Q4_0 + 7 Q8_0 matrices, embedding BF16
  AMD Ryzen 5 7530U with Radeon Graphics | 6 physical cores, 12 logical | 588.0 MiB of weights per token
  prompt 23 tokens, then 32 greedy decode steps per configuration (EOS ignored)

  kernels threads  prefill tok/s  decode tok/s     GB/s % mbench  speed-up  outputs
  scalar        1           4.54          5.33     3.28      17%    1.00x  reference
  avx2          1          10.79         10.48     6.46      34%    1.97x  vs scalar: logits rel 6.1e-07, tokens 32/32
  avx2          2          15.40         13.25     8.17      43%    2.49x  bit-identical to 1 thread; vs scalar: logits rel 6.1e-07, tokens 32/32
  avx2          4          20.51         20.41    12.58      67%    3.83x  bit-identical to 1 thread; vs scalar: logits rel 6.1e-07, tokens 32/32
  avx2          6          23.24         21.92    13.52      72%    4.12x  bit-identical to 1 thread; vs scalar: logits rel 6.1e-07, tokens 32/32
  avx2         12          18.32         19.36    11.94      63%    3.64x  bit-identical to 1 thread; vs scalar: logits rel 6.1e-07, tokens 32/32
  scalar       12          12.98         15.00     9.25      49%    2.82x  bit-identical to 1 thread

  thread-count invariance: PASS (every thread count reproduces its 1-thread logits)
  GB/s = weight bytes per token x decode tok/s (estimated weight traffic)
  % mbench = share of the membench baseline 18.80 GB/s (a software read test, not the hardware limit)
  peak RSS 614.0 MiB
```

## Speed-up ladder, BF16 engine (bench-decode)

```
$ /home/manas/ef-build/edgeforge bench-decode --model-dir /home/manas/edgeforge-data/tinyllama --tokens 16
EdgeForge bench-decode: BF16 safetensors
  AMD Ryzen 5 7530U with Radeon Graphics | 6 physical cores, 12 logical | 1.93 GiB of weights per token
  prompt 23 tokens, then 16 greedy decode steps per configuration (EOS ignored)

  kernels threads  prefill tok/s  decode tok/s     GB/s % mbench  speed-up  outputs
  scalar        1           1.53          1.46     3.01      16%    1.00x  reference
  avx2          1           8.69          8.29    17.14      91%    5.69x  vs scalar: logits rel 1.0e-06, tokens 16/16
  avx2          2          10.51          9.73    20.13     107%    6.68x  bit-identical to 1 thread; vs scalar: logits rel 1.0e-06, tokens 16/16
  avx2          4          10.84         10.09    20.88     111%    6.92x  bit-identical to 1 thread; vs scalar: logits rel 1.0e-06, tokens 16/16
  avx2          6           9.85          9.31    19.26     102%    6.39x  bit-identical to 1 thread; vs scalar: logits rel 1.0e-06, tokens 16/16
  avx2         12           8.40          7.80    16.14      86%    5.35x  bit-identical to 1 thread; vs scalar: logits rel 1.0e-06, tokens 16/16
  scalar       12           4.39          3.97     8.22      44%    2.73x  bit-identical to 1 thread

  thread-count invariance: PASS (every thread count reproduces its 1-thread logits)
  GB/s = weight bytes per token x decode tok/s (estimated weight traffic)
  % mbench = share of the membench baseline 18.80 GB/s (a software read test, not the hardware limit)
  peak RSS 1.95 GiB
```

## Where the time goes (profile)

```
$ /home/manas/ef-build/edgeforge profile --qmodel /home/manas/edgeforge-data/tinyllama-q4.efq
EdgeForge profile: efq, policy tinyllama-1.1b-chat-v1.0: 148 Q4_0 + 7 Q8_0 matrices, embedding BF16
  avx2 kernels, 6 threads | 31 decode steps after a 23-token prompt (after one warm-up pass)

  op                 ms/token   share  weights/token      GB/s
  embedding             0.002    0.0%
  attn RMSNorm          0.095    0.2%
  q/k/v matmul          4.901   10.8%       63.4 MiB     13.56
  RoPE                  0.030    0.1%
  attention             0.426    0.9%
  o_proj matmul         3.872    8.6%       49.5 MiB     13.41
  residual adds         0.026    0.1%
  ffn RMSNorm           0.078    0.2%
  gate/up matmul       20.778   45.9%      272.2 MiB     13.74
  SwiGLU                0.817    1.8%
  down matmul          10.542   23.3%      136.1 MiB     13.54
  final RMSNorm         0.004    0.0%
  lm_head matmul        3.662    8.1%       66.4 MiB     19.02
  total (forward)      45.233  100.0%      587.7 MiB     13.62

  22.11 tok/s inside forward(); 22.10 tok/s wall clock (includes argmax between steps)
  matmul GB/s vs the membench baseline 18.80 GB/s (a software read test, not the hardware limit)
```
