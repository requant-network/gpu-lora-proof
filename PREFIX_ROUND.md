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
