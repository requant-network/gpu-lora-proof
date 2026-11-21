# Private choice across three Qwen prefix templates

## Protocol fixed before proving

Start with the two previously approved Qwen prefix shapes and their
retained reference orders. Derive a third public shape by swapping the
first two differently labeled held-out rows of the CPU reference.
Swap both features and targets, so the sum of squared errors, label count,
optimizer inputs and reference adapter stay the same. Publish the three
admissible shape digests and the derived public shape.

Use three private Boolean selectors constrained to sum to one. Each
selector routes the private numeric order and adapter into one prefix
check and routes the public reference into the other checks. Use those
same selectors to choose the template digest in the salted order
commitment. The verifier key is shared and commitments remain the only
public proof inputs.

Create one private order per template using the existing private sign
transform. Shuffle delivery order and record no mapping from delivery
to template in the public package. Generate three proofs with independent
randomness using one disposable setup. Verify each delivery in a separate
read-only arbiter container without network access. The customer opens
each linked adapter independently.

Check that the fixture rejects duplicate templates, a choice outside the
allowlist, a changed selector on an issued order, a newly issued order
whose arithmetic conflicts with its selected template, and a proof
copied from the other order. Check the retained single-template and
two-template proofs under their original pins and compare the previous
two-template constraint count. Measure cold key preparation, prover
time, peak RSS, verifier time and transmitted proof bytes.

## Results

All three orders passed in fresh arbiter containers under the same key.
All customer openings matched. The verifier received two commitments and
a proof for each order, plus the key and trusted hashes. The chosen
template and openings stayed outside the arbiter containers. The earlier
single-template and two-template proofs passed under their original keys,
and the old two-template circuit still had 347,690 constraints.

| Ryzen 5 5500, two Rayon threads | Two templates, previous trial | Three templates |
| --- | ---: | ---: |
| Constraints | 347,690 | 477,025 |
| Setup wall time | 11.62 s | 14.20 s |
| Proving key | 71,338,640 bytes | 91,614,320 bytes |
| Cold wall time per proof | 41.19–41.37 s | 54.12–54.37 s |
| Cold key loading per proof | 35.07–35.08 s | 46.93–46.95 s |
| Peak prover RSS | 1.29 GiB | 1.65 GiB |
| Verifier time in the binary | 3.33–3.36 ms | 3.37–3.56 ms |
| Verification key | 328 bytes | 328 bytes |
| Commitments and proof per order | 192 bytes | 192 bytes |

The three deliveries and shared key total 904 bytes. This counts the
proof package separately from the already known shapes, the trust pins
and the verifier executable. Each cold run includes its own checked
proving-key load. The two-template measurements come from the
[previous experiment](PRIVATE_TEMPLATE_PAIR.md#results).

The five planned refusal controls returned errors. An out-of-range
choice failed before the prover opened its key. A newly issued
commitment with a mismatched template reached the arithmetic checks and
failed there. The seven Rust and fourteen Python tests passed.
[Report, proof bytes and pins](results/private-template-triple/report.json) ·
[Public reference shape](results/private-template-triple/third-shape.json) ·
[Runtime and source pins](results/private-template-triple/environment.txt).

This measures hidden selection among three known options, including
one permuted reference. The executing worker and the customer can see
the chosen template and numeric order. The arbiter can see the
allowlist and the public proof package. Side-channel resistance,
arbitrary templates and secrets hidden from the worker remain outside
this experiment. The added measured cost falls mainly on proving-key
handling and on proving.

The [shared proving-key follow-up](SHARED_TRIPLE_PROVING_KEY.md)
compares the same three orders with one checked key load.

## Reproduce

Use the pinned dependencies and a preloaded Ubuntu 24.04 Docker image.
Choose a new output directory on Linux.

```sh
cargo build --release --locked --manifest-path compact-round/Cargo.toml
python3 -B compact-round/private_pair_trial.py \
  compact-round/target/release/compact-lora-round NEW_OUTPUT --three-templates
```

The runner writes the verifier key, commitments, proofs and public
shape in NEW_OUTPUT/public. New runs use fresh salts and setup keys.
The protocol digest in the environment file covers the section fixed
before the trial.
