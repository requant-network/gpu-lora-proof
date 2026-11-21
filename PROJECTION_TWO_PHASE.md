# A two-phase rank-one LoRA update inside Qwen

## Protocol fixed before measuring

Use the frozen model, CPU recipe, corpus, feature normalization and output
head from the [single-projection experiment](TRANSFORMER_PROJECTION.md).
Keep the last attention output projection as the only trainable tensor.

1. Reproduce the original output-factor search. Select B using only the
   16 adaptation texts while A remains the unit vector at coordinate zero.
2. Freeze the selected B and keep A[0] equal to 1. Test the unchanged A
   and one additional coordinate at a time. The input positions are
   112, 224, 336, 448, 560, 672 and 784. For each position test −1/4 and
   +1/4. Select minimum squared error on the same adaptation texts.
   Break ties in favor of leaving A unchanged, then by increasing position
   and negative sign before positive sign.
3. Evaluate the selected adapters from both phases on the 16 held-out texts
   after both decisions. Release the two-phase adapter only for a nonzero
   update to A, a lower held-out squared error than both references and
   at least as many correct labels as each reference.

This is an extra coordinate-search step with rank one throughout. The record
binds the chosen first-phase adapter digest as the second-phase input,
the feature and GGUF digests for every evaluated point, and the final adapter
digest. A fresh process must reproduce the complete record before it publishes
an adapter. Use a changed input digest as a broken-link control and the v1
certificate and adapter as a compatibility control.

## Results

The input-factor search chose +1/4 at coordinate 784. All three variants
classified 16 of 16 held-out texts correctly.

| Adapter | Adaptation squared error | Held-out squared error |
| --- | ---: | ---: |
| Frozen baseline | 85,712 | 114,328 |
| Output factor B only | 71,784 | 97,610 |
| Output factor B followed by input factor A | 50,108 | 99,072 |

The second phase reduced adaptation error by 30.2% and increased held-out
error by 1.5% relative to the first phase. Retain the first-phase adapter
for this fixture. The release gate withheld the second-phase GGUF.

The LAN Ryzen 5 5500 produced the 6,309-byte record in 81.58 s. A fresh
replay took 82.80 s and reproduced the complete record byte for byte.
The original v1 certificate and adapter still replayed byte for byte.
All fourteen Python tests passed. The writer rejected five invalid
coordinates or signs before creating a GGUF.
[Certificate](results/projection-two-phase/certificate.json) ·
[Replay](results/projection-two-phase/replay-summary.json) ·
[v1 control](results/projection-two-phase/legacy-summary.json) ·
[Writer controls](results/projection-two-phase/writer-controls.log) ·
[Source and environment](results/projection-two-phase/environment.txt).

Changing only the second-phase input digest caused exit 1 and left the
output directory absent. Changing the digest and claiming release also
failed. [Broken link](results/projection-two-phase/wrong-phase-link.log) ·
[False release claim](results/projection-two-phase/tamper.log).

Both phases used the same CPU, corpus and projection. The record covers
coordinate selection and the linked optimizer inputs for this fixture.
Gradient-based changes across layers and evaluation on a new corpus remain
open.

The [signed two-host round](SHARDED_PROJECTION.md) distributes the
same adaptation scores and verifies both worker contributions per phase.

## Reproduce

Build the updated writer and use the pinned runtime as in the
[single-projection recipe](TRANSFORMER_PROJECTION.md#reproduce). Pass
`--two-phase` to the same command and use a fresh output directory.
For verification point `--certificate` at the retained v2 certificate.

```sh
python3 -B experiments/transformer_projection.py \
  results/text-head-origin/cpu/recipe.json results/text-head-v2/corpus.tsv \
  "$MODEL" "$LLAMA_LIB" "$PROBE" ./write-projection-adapter new-replay \
  --recipe-sha256 fe9ff83c6077430808d15972e41f18a168d74a5fae75b982d5e6816184398446 \
  --two-phase --certificate results/projection-two-phase/certificate.json
cmp new-replay/certificate.json results/projection-two-phase/certificate.json
test ! -e new-replay/adapter.gguf
```
