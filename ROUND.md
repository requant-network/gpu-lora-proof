# A verified integer LoRA round

The November 21 experiment ran a complete synthetic round for each of two
fixed seeds. Each round uses a 32-wide rank-4 adapter, two optimizer phases,
two worker slots per phase and four training examples per slot. The existing
single-step Freivalds verifier checks every contribution. A separate round
verifier checked the final evidence on both Linux and Windows.

## Contract

The trusted manifest fixes the base weights, initial adapter, ordered training
batches and held-out examples. Publish its SHA-256 through a trusted channel
before accepting contributions. The phase and worker slot derive a distinct
single-step ID from that pin.

Each contribution carries the existing forward and backward certificate. The
round verifier ties its weights and features to the expected slot and computes
the clipped integer loss gradient from the certified prediction and pinned
target. Both workers in a phase start from the same adapter. The verifier
sums the checked gradients, then takes one bounded sign-SGD step on the selected
factor. The first phase updates B and the second updates A. The base weights
stay fixed. The next phase needs the exact A and B produced by that transition.

The Merkle tree covers all four ordered contribution records and their phase
input and output adapter digests. The verifier recomputes the tree, the adapter
and the held-out error. Worker names identify slots in the trusted fixture.
The experiment measures their execution on two hosts under one operator.
Independent operator identity, real-model training quality and private data
require further work.

The [compact proof comparison](COMPACT_ROUND.md) uses these same manifests and
adapters to compare direct replay, the retained round certificate and Groth16.

## Results

| Seed | Held-out rows | Initial → final squared error | Manifest + bundle | Linux verification | Two-host assembly |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261121 | 16 | 24,576 → 0 | 7,976 + 203,425 bytes | 11.66 ms | 1.176 s |
| 90817 | 16 | 24,576 → 0 | 7,924 + 203,427 bytes | 11.49 ms | 1.092 s |

The pinned targets come from the independent rank-1 CPU oracle. Each seed uses
a known synthetic teacher with an exact fit under the selected two-phase rule.
The all-CPU Linux run and the CPU/RTX run produced equal canonical bundles,
including the proof bytes. The Windows workstation generated worker A's two
contributions and retrieved worker B's two contributions from the Linux RTX
5060 Ti over a private lab connection. Both two-host bundles matched the
retained artifacts byte for byte. The Windows verifier recovered the same
adapter. Linux verification measures file parsing, arithmetic and adapter
write. Two-host assembly includes process launch, transport, contribution
verification and bundle write. [Mixed and CPU run report](results/round-v1/report.json) ·
[Two-host observations](results/round-v1/two-host.json).

All twenty negative controls were rejected. They include missing or duplicated
contributions, a re-labelled step, valid single-step proofs for a stale adapter
or a wrong upstream gradient, a re-sealed wrong matrix, a mismatched transition
and altered commitments. Twelve unit tests passed on both platforms. The new
CLI check refuses a wrong manifest pin, an incomplete round and overwriting a
recovered adapter. [Linux test output](results/round-v1/checks.log).

A clean Windows checkout with automatic line-ending conversion enabled also
preserved both manifest pins and recovered the same adapter bytes. The retained
round files use LF line endings to keep their exact bytes portable.

## Verify the retained round

```sh
python3 -B round_verifier.py \
  results/round-v1/20261121/manifest.json results/round-v1/20261121/mixed.json \
  --manifest-sha256 c645e19ef587a2b582ceba48719de32032e5584d33eacdbcf33b0db39421fe65 \
  --adapter recovered-20261121.json
```

Use the second fixture with manifest SHA-256
`95755b07cac5a522df5f94cbc721870ebf957c7a42944292fae197a6b0ade829`
to verify the other cohort. The output path must be new. [First manifest](results/round-v1/20261121/manifest.json) ·
[First proof bundle](results/round-v1/20261121/mixed.json) ·
[Second manifest](results/round-v1/90817/manifest.json) ·
[Second proof bundle](results/round-v1/90817/mixed.json).

Rebuild the two cohorts in a new directory:

```sh
python3 -B -m unittest discover -s tests -v
python3 -B round_experiment.py run --output ./cpu-round-run
```

The optional `--lib /path/to/liblora.so` uses the existing CUDA library for
worker B after the GPU identity check. Build it with the existing compiler
command and run the experiment with the selected RTX visible and the shared
lab lock held. To split the contributors across hosts, run
`round_experiment.py contribute MANIFEST PHASE WORKER` at each site, sending
the verified starting adapter as JSON on stdin. The second phase must wait
for both first-phase certificates and the verified optimizer transition.
[Single-step CUDA build](README.md#layout--reproduce).
