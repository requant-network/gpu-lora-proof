# A training-loss-selected integer LoRA transition

## Protocol fixed before the comparison

The [first frozen-Qwen trial](TEXT_HEAD.md) rejected the prescribed
full sign step on factor A. This follow-up evaluates a training-only
choice of how many coordinates to update in each factor. It retains
the same rank, step scale, initialization, targets, two phases and
two contribution slots per phase.

For each phase, verify the existing certificates and sum the two
certified gradients. Drop coordinates where the clipped full sign
step would leave the value unchanged. Sort the remaining coordinates
by descending absolute summed gradient, breaking ties by ascending
index. Evaluate all prefixes, including the unchanged factor. For
each prefix use the exact unclipped squared error over the eight
pinned training rows of that phase. Choose the shortest prefix attaining
the minimum. This rule permits a no-op and guarantees that each phase's
training loss cannot rise on its own eight inputs. Held-out examples
enter the evaluation after the optimizer has selected both transitions.

The rule is bound by the new `gpu-lora-round/v2` schema, including
the step seeds and the adapter and phase commitments. The existing
v1 schema still applies one full sign step. The compact proof paths
remain scoped to the original v1 statement.

Compare v1 and v2 on the original 48 texts and on a separately written,
balanced [mixed-vocabulary set](results/text-head-v2/corpus.tsv). In the
second set, the adaptation and held-out examples mention both topics
and the label follows the main subject of the brief. Recompute Qwen
features for that set using the prior frozen capture configuration.
Keep each corpus's source-only fitted head as its baseline. For this
follow-up, the utility criterion requires a lower held-out squared
error and at least as many correct held-out labels as the frozen head.
Evaluate both corpora regardless of outcome. A second manifest sets
the verifier's squared-error ceiling to the baseline error. Report
the verifier's adapter-release decision and the utility criterion for
each corpus.

## Measured outcome

| Corpus | Frozen head correct / SSE | Full sign step correct / SSE | Selected prefix correct / SSE |
| --- | ---: | ---: | ---: |
| Original | 16/16 · 427,482 | 0/16 · 141,312,730 | 15/16 · 148,826 |
| Mixed vocabulary | 16/16 · 130,474 | 11/16 · 22,034,378 | 16/16 · 80,010 |

The prefix chose six B coordinates and one A coordinate in the original
corpus, and four B coordinates and one A coordinate in the new corpus.
The second corpus met the prespecified utility gate, with a 38.7% decrease
in held-out squared error and no change in top-1 accuracy. The original
corpus improved squared error by 65.2% and lost one label. Keep its frozen
head. Both baselines had perfect top-1 accuracy on their small test splits.
[Original results](results/text-head-v2/original/mixed-report.json) ·
[Mixed-vocabulary results](results/text-head-v2/mixed-vocabulary/mixed-report.json) ·
[Full-step control on the new corpus](results/text-head-v2/mixed-vocabulary/v1-report.json).

## Adapter release

The scalar squared-error ceiling accepted both prefix adapters. That
exposed a gap between the verifier's release check and the prespecified
utility rule. In a separate manifest, enforce an exact integer error
ceiling one below baseline and a top-1 floor equal to the baseline
count. The floor uses the first maximum when scores tie, and requires
a unique maximum in each pinned classification target.

With both bounds, the original round returned `NO: held-out correctness
gate failed`, exit code 1, and wrote no adapter. The mixed-vocabulary
round returned exit code 0 and wrote the expected adapter. Both outcomes
matched on Linux and Windows. The verifier also rejected replacing the
selected prefix with the actual full sign transition under the same
certificates. The mixed CPU/RTX and all-CPU Linux runs produced identical
measurement and release bundles. Windows CPU reproduced both retained
release bundles. Two fresh captures of the new Qwen feature table matched
byte for byte. All 14 unit tests passed on both operating systems, and
the three retained v1 bundles still verified.
[Original release manifest](results/text-head-v2/original/manifest.json) ·
[Accepted release manifest](results/text-head-v2/mixed-vocabulary/manifest.json) ·
[Runtime and hashes](results/text-head-v2/environment.log.txt).

The CPU verifier took 89.7 ms for the original prefix run and 65.3 ms for
the mixed-vocabulary prefix run on the LAN host. The earlier ordinary
full-step certificates took about 11 ms. The extra cost covers exact
prefix replay, with four step certificates in each version. This trial
evaluates a head trained on frozen Qwen features and short authored
briefs. Broader held-out evaluation and transformer weight updates
remain open.

The [feature-origin replay](FEATURE_ORIGIN.md) reruns Qwen on the
original texts before checking the manifest and adapter. The same
procedure uses separately pinned recipes for the RTX and CPU captures.

## Reproduce

From the repository root, run the verifier against both retained bundles
with their distinct trusted manifest hashes. Choose fresh output paths.
The first command must exit with code 1 and leave its output absent.
The second must recover the checked-in adapter bytes.

```sh
run=results/text-head-v2
python3 -B round_verifier.py "$run/original/manifest.json" "$run/original/bundle.json" \
  --manifest-sha256 bf127ae5cc7ded34a44c0c0bea4654864eac637da1a51480ad6d2d70c18cc609 \
  --adapter original-release.json
python3 -B round_verifier.py "$run/mixed-vocabulary/manifest.json" "$run/mixed-vocabulary/bundle.json" \
  --manifest-sha256 d200e82458d79b44fc577480ba33b1e7fe79b6e935328bd7812cfdd05763c34a \
  --adapter mixed-release.json
cmp "$run/mixed-vocabulary/adapter.json" mixed-release.json
```

Rebuild each trial with `python3 -B experiments/text_head_round.py CORPUS
FEATURES NEW_OUTPUT --prefix`. Use `results/text-head-v1` for the original
corpus and feature table, and `results/text-head-v2` for the mixed-vocabulary
corpus and feature table. The generated `combined-gate` directory holds
the release manifest and bundle. Add `--lib` with the existing CUDA proof
library to move the second contributor to RTX. The [first Qwen head experiment](TEXT_HEAD.md#reproduce)
specifies the separate feature capture build and pinned runtime.
