# Replay Qwen feature extraction before releasing a LoRA adapter

November 21, 2026. The verifier reruns frozen Qwen2.5-0.5B feature
extraction over the 48 pinned public texts, reconstructs the integer
round and checks the existing v2 certificate before writing an adapter.
It uses the same last-token position, 32 sampled coordinates and
source-only normalization as the [original head experiment](TEXT_HEAD.md).
The worker can supply the certificate bundle. The verifier controls
the probe, model, runtime libraries, corpus and the trusted recipe hash.

## Verification

1. Check the recipe against its trusted SHA-256. Verify the corpus, GGUF
   and the runtime libraries pinned by that recipe. The RTX profile pins
   eight libraries and the CPU profile pins four.
2. Start a fresh capture process on the selected RTX or CPU. Load the
   pinned backends explicitly. Check the complete fresh feature table
   against the recipe. The command has no feature-table input.
3. Rebuild the integer head from the source split and assign adaptation
   and held-out rows from the replay. Reapply the exact release bounds of
   the frozen head. The SHA-256 of the complete canonical manifest must
   match both the trusted recipe and the submitted certificate.
4. Replay the four existing round certificates, exact optimizer
   transitions and release gates. Write the adapter to a fresh path
   after all checks succeed.

The verifier assumes a trusted local build of the probe, local host and
system driver. The pinned runtime list identifies the libraries used
by the workload. File paths must remain under the verifier's control
between digest checking and execution. The statement covers this
feature projection and integer head on the two public corpora.
The [projection follow-up](TRANSFORMER_PROJECTION.md) tests one
rank-one weight update inside the transformer. The separate [cross-host
replay](CROSS_HOST_REPLAY.md) reproduced the CPU profile on two Ryzen
processors. Broader updates, independent operator identity and
cross-host RTX replay remain open.

## Evidence

Both profiles ran on the LAN host with Python 3.12.3. The RTX 5060 Ti
used driver 610.43.03 and the CPU profile used the Ryzen 5 5500. The
mixed-vocabulary corpus gave the following results.

| Profile | Held-out labels before and after | Squared error before and after | Full verification |
| --- | ---: | ---: | ---: |
| RTX | 16/16 and 16/16 | 130,474 and 80,010 | 33.634–35.207 s across three runs |
| CPU | 16/16 and 16/16 | 114,328 and 72,888 | 9.224 s on the end-to-end run |

All three RTX runs reproduced the original feature, manifest and
adapter digests. Three CPU captures matched each other and released
the separately certified CPU adapter. CPU and RTX had the same token
counts on all 48 texts. All 48 raw feature rows differed and 629 of
1,536 integer coordinates changed after source-only normalization.
Keep independent recipe and manifest hashes for the two backends.
Submitting the RTX certificate to the CPU recipe produced a manifest
refusal and no output.
[RTX runs 1](results/text-head-origin/mixed-1.json),
[2](results/text-head-origin/mixed-2.json) and
[3](results/text-head-origin/mixed-3.json) ·
[CPU replay](results/text-head-origin/cpu/cpu-verified.json) ·
[Feature comparison](results/text-head-origin/cpu-control.json).

The original dataset reproduced its features and manifest, then failed
the held-out correctness gate and released no adapter. Six input and
pin controls also refused output, covering recipe trust, corpus,
weights, runtime library, requested manifest and captured features.
All 14 existing unit tests passed on Linux and Windows. Windows
also verified the retained CPU certificate against its integer
manifest and reported the same squared error.
[Original refusal](results/text-head-origin/original.log.txt) ·
[Boundary controls](results/text-head-origin/controls.log.txt) ·
[Cross-profile refusal](results/text-head-origin/cpu/cpu-wrong-bundle.log.txt).

## Reproduce

From the repository root, build the trusted probe with the header and
library configuration in [the feature capture instructions](TEXT_HEAD.md#reproduce).
Set `MODEL` to the pinned GGUF, `LLAMA_LIB` to the installed library directory,
`GPU_UUID` to the selected RTX and `PROBE` to the local executable.
Use a fresh output path.

```sh
python3 -B experiments/verify_text_head_origin.py \
  results/text-head-origin/mixed-vocabulary-recipe.json \
  results/text-head-v2/corpus.tsv results/text-head-v2/mixed-vocabulary/bundle.json \
  "$MODEL" "$LLAMA_LIB" "$PROBE" recovered.json --gpu "$GPU_UUID" \
  --recipe-sha256 1156cf1fe807a3d2a4f1a790f176d9d8af3a4afaa73a3a4f4bc2a3450cf8d170
cmp recovered.json results/text-head-v2/mixed-vocabulary/adapter.json
```

The original corpus uses `results/text-head-origin/original-recipe.json`,
`results/text-head-v1/corpus.tsv`, and `results/text-head-v2/original/bundle.json`.
Its recipe pin is
`3751ac1eb55d85407b455d6e0d2ad9dce0221185d36b52854a1d092ef49e08a0`.
It is expected to exit 1 with `NO: held-out correctness gate failed`
and leave its adapter path absent.

The CPU recipe pins the same model and corpus with an independently
captured feature table, manifest and certificate. Omit `--gpu` for this
profile.

```sh
cpu=results/text-head-origin/cpu
python3 -B experiments/verify_text_head_origin.py \
  "$cpu/recipe.json" results/text-head-v2/corpus.tsv "$cpu/bundle.json" \
  "$MODEL" "$LLAMA_LIB" "$PROBE" recovered-cpu.json \
  --recipe-sha256 fe9ff83c6077430808d15972e41f18a168d74a5fae75b982d5e6816184398446
cmp recovered-cpu.json "$cpu/adapter.json"
```
