# An integer LoRA round on frozen Qwen features

## Protocol fixed before feature capture

The study asks whether the existing two-phase verified LoRA round improves
a small topic head attached to a frozen pretrained text model. The backbone
is Qwen2.5-0.5B-Instruct Q4_K_M, GGUF SHA-256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
The input is a self-contained, newly written English news-brief corpus.
The [fixed corpus](results/text-head-v1/corpus.tsv) has 16 examples in each
of three disjoint splits, with eight sports and eight weather labels per
split. The source split fits the frozen linear head. The adaptation split
drives the two LoRA phases. The held-out split measures utility.

One decoder call per text captures the last token's final hidden state,
with the KV memory cleared between texts. The probe saves 32 equally
spaced coordinates from its 896-wide row. For each coordinate, the source
split fixes the mean and population standard deviation. Multiply centered,
standardized values by four, round to nearest with ties to even and clip
to ±16. A zero standard deviation uses divisor one. The source sports
and weather centroids of the integer features fix opposite columns for
the frozen topic head. Each weight is half their difference rounded to
nearest with ties to even. Its other 30 columns are zero.

Use the existing 32-wide rank-4 round, step scale eight, and two sign-SGD
phases. Initialize B to zero and put a single one at input coordinate
0, 8, 16 or 24 in each respective row of A. Targets are `[96, -96]` for
sports and `[-96, 96]` for weather, followed by zeros. Each phase uses
two consecutive four-example adaptation slots. The manifest's squared
error ceiling follows the bounded arithmetic and permits certifying a
correct round irrespective of task utility. The prespecified utility
gate requires both an increase in held-out top-1 count and no increase
in held-out squared error. Ties in head score choose sports. A verified
round can fail this utility gate. No features, hyperparameters or texts
will be selected after examining held-out scores.

The verifier binds the classifier arithmetic and optimizer transitions
to the pinned integer feature manifest. Feature provenance requires
repeating the separate Qwen capture on the pinned model and runtime.

## Result

The preregistered two-phase optimizer failed the utility gate. The frozen
head already classified all 16 held-out texts correctly. The B update
preserved those labels and moved the scores closer to their target. The
subsequent A update reversed all 16 held-out decisions and all 16 decisions
on the adaptation split.

| Snapshot | Held-out correct | Held-out squared error | Adaptation correct |
| --- | ---: | ---: | ---: |
| Frozen head | 16/16 | 427,482 | 16/16 |
| After updating B | 16/16 | 108,378 | 16/16 |
| After updating A | 0/16 | 141,312,730 | 0/16 |

The baseline ceiling on top-1 accuracy leaves squared error as the
observable improvement in this split. With the permissive arithmetic
ceiling in the preregistered manifest, the verifier accepted the exact
failed optimizer transition. Rebuilding the round under a different
manifest pinned to the baseline error of 427,482 produced a separate
valid contribution set. The verifier returned exit code 1 with
`NO: held-out quality gate failed` and wrote no adapter. The result was
the same in Windows and Linux. Keep the frozen head for this fixture.
[Linux CPU/RTX report](results/text-head-v1/mixed-report.json) ·
[Windows replay and gate](results/text-head-v1/windows-report.json).

Two fresh RTX 5060 Ti feature captures produced the same bytes. The
Linux all-CPU, Linux CPU/RTX and Windows all-CPU executions produced
identical manifest, bundle and adapter bytes. The mixed bundle used CPU
for worker A and RTX for worker B in both phases. Its manifest is 7,642
bytes and its bundle is 203,425 bytes. Mixed assembly took 0.724 s,
including contributor process launches and output. Independent
verification took 11.1 ms on Linux and 15.8 ms on Windows. The ten
existing negative certificate controls were rejected on the new fixture,
and all twelve existing unit tests passed on both systems.
[Feature rows](results/text-head-v1/features.tsv) ·
[CPU-only Linux report](results/text-head-v1/linux-cpu-report.json) ·
[Environment and hashes](results/text-head-v1/environment.log.txt) ·
[Linux tests](results/text-head-v1/tests.log.txt).

The trial adapts a two-topic head on a 32-coordinate feature projection,
with a frozen backbone and a small self-contained corpus. Feature
extraction runs outside the integer proof. Generalization to new
datasets and updates to transformer weights remain open.

## Reproduce

Verify or replay from the retained integer inputs and feature table using
Python 3.12. Each output path should be new.

```sh
run=results/text-head-v1
python3 -B round_verifier.py "$run/manifest.json" "$run/bundle.json" \
  --manifest-sha256 79e808cd5a81aefcc38ab4b64a9f4176a4b876f2640e1d8a71cb37b0eb39679d \
  --adapter recovered-head.json
python3 -B experiments/text_head_round.py "$run/corpus.tsv" "$run/features.tsv" cpu-head-run
```

The replay also constructs the quality-gate control in its output
directory. Use `--lib "$LORA_LIB"` with the [CUDA proof library](README.md#layout--reproduce)
to run the second contributor on RTX. Recreate the feature table on the
selected RTX with the exact runtime libraries in `environment.log.txt`
and headers from llama.cpp `161755f29e415e2c33efe906e91843c068efd664`.
Set `HEADERS` to that source tree, `LLAMA_LIB` to the library directory,
`MODEL` to the pinned GGUF and `GPU_UUID` to the authorized RTX.

```sh
g++ -std=c++17 -O2 -I"$HEADERS/include" -I"$HEADERS/ggml/include" \
  experiments/text_head_features.cpp -L"$LLAMA_LIB" -lllama -lggml -lggml-base -o text-head-features
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export LD_LIBRARY_PATH="$LLAMA_LIB:$LLAMA_LIB/cuda_v13:${LD_LIBRARY_PATH:-}"
export GGML_BACKEND_PATH="$LLAMA_LIB/cuda_v13/libggml-cuda.so"
./text-head-features "$MODEL" "$LLAMA_LIB" "$run/corpus.tsv" > new-features.tsv
cmp "$run/features.tsv" new-features.tsv
```
