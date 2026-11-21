# A replayed LoRA update inside Qwen

## Protocol fixed before execution

Use the pinned CPU profile and the mixed-vocabulary corpus from the
[feature replay trial](FEATURE_ORIGIN.md). Recreate the baseline features,
source normalization and the source-fitted head. Freeze the normalization,
head and Qwen base weights throughout this trial.

Apply rank-one LoRA to `blk.23.attn_output.weight`, the last block's
attention output projection in Qwen2.5-0.5B. Fix A to coordinate zero.
Fix B's direction from the source-fitted sports column. At the 32
sampled feature coordinates, set B to the corresponding integer head
weight multiplied by the trial coefficient and divided by 64. All
other B entries are zero. Load the two F32 factors as a GGUF adapter
with alpha 1 and scale 1. The resulting matrix update is rank one.
[Llama.cpp's adapter loader](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-adapter.cpp).

Replay the 16 adaptation texts for each trial coefficient in
`0, -1, 1, -2, 2, -4, 4, -8, 8`. Choose the smallest integer squared
error, breaking ties by smaller absolute coefficient and then smaller
coefficient. Evaluate the selected candidate once on the 16 held-out
texts using the frozen head. Release requires a nonzero update, lower
held-out squared error and no lost correct label relative to baseline.

Bind the CPU recipe, target tensor, each candidate's encoded adapter
and feature digests, exact adaptation scores and decision in retained
JSON. Verify by repeating the model runs and reconstructing that
record. Reject a different record before creating a release directory.
This transparent choice test updates one coefficient of one transformer
projection. Other coefficients, layers and gradients remain outside
the statement.

## Measured round

The Linux Ryzen 5 5500 server ran the pinned CPU profile. The adaptation
split selected coefficient 8, the upper endpoint of the fixed grid.
The held-out split was evaluated after this choice.

| Split | Frozen correct | With LoRA correct | Frozen squared error | With LoRA squared error |
| --- | ---: | ---: | ---: | ---: |
| Adaptation | 16/16 | 16/16 | 85,712 | 71,784 |
| Held out | 16/16 | 16/16 | 114,328 | 97,610 |

Held-out squared error fell by 14.6%. The 7,488-byte GGUF affected one
projection in the transformer, while the output head and quantization
of its features stayed frozen. The verifier replay used a fresh
process and performed the complete coefficient search again. The JSON
record is 2,278 bytes. Replaying it costs another run of the workload.

The measurement took 35.869 s and the verifier replay took 35.912 s.
Their JSON records and GGUF adapters matched byte for byte. Changing
the selected coefficient to its negative caused the verifier to exit
1 with no release directory. All 14 existing unit tests passed on
Linux and Windows. The old CPU and RTX head manifests still rebuilt
byte for byte from their recorded feature tables.
[Certificate](results/transformer-projection/certificate.json) ·
[GGUF adapter](results/transformer-projection/adapter.gguf) ·
[Replay timing](results/transformer-projection/verified.json) ·
[Altered certificate result](results/transformer-projection/altered-certificate.log.txt).

The [cross-host follow-up](CROSS_HOST_REPLAY.md) reproduced the record
and GGUF byte for byte on the workstation CPU with the same runtime.

The GGUF artifact contains the two generated LoRA factors. The
frozen base is Qwen2.5-0.5B-Instruct Q4_K_M. The model publisher
distributes the base under Apache-2.0.
[Qwen model card and license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).

The selected point sits on the boundary of the tested grid. The result
covers the pinned texts, model, native CPU profile and this first
projection update. Generalization, selection outside the grid and
gradient-based updates to a wider set of transformer weights need
separate experiments.

## Reproduce

Use the exact model, CPU recipe and native headers and libraries from
[the Qwen capture trial](FEATURE_ORIGIN.md#reproduce). Compile the base
probe as directed there. Build the GGUF writer against the same headers
and runtime library directory.

```sh
g++ -std=c++17 -O2 -I"$HEADERS/ggml/include" \
  experiments/write_projection_adapter.cpp -L"$LLAMA_LIB" -lggml-base \
  -o write-projection-adapter
```

With `PROBE` pointing to the locally built feature executable, run the
complete check from the repo root and supply a fresh output directory.

```sh
python3 -B experiments/transformer_projection.py \
  results/text-head-origin/cpu/recipe.json results/text-head-v2/corpus.tsv \
  "$MODEL" "$LLAMA_LIB" "$PROBE" ./write-projection-adapter projection-replay \
  --recipe-sha256 fe9ff83c6077430808d15972e41f18a168d74a5fae75b982d5e6816184398446 \
  --certificate results/transformer-projection/certificate.json
cmp projection-replay/adapter.gguf results/transformer-projection/adapter.gguf
```
