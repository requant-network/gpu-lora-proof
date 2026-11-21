# Private choice between two Qwen prefix templates

The [disclosed-template experiment](PRIVATE_PREFIX.md) uses one fixed
coordinate order and label layout per verification key. This follow-up
tests two previously verified Qwen profiles with different ordered
coordinates under one key. The exact numeric order and adapter remain
private from the arbiter. The two approved templates and their public
reference orders are known.

The circuit takes a private Boolean choice. It routes the private order
and adapter into the chosen prefix check. It routes the other branch's
known valid reference into that branch. Both branches run the full v2
prefix selection, squared-error and held-out accuracy checks. The same
Boolean selects the template digest in the salted order commitment.
The output commitment binds the returned adapter to that order.
The arithmetic circuit and proving key are fixed for the pair, with
exactly the two commitments as public inputs.

The test shuffles two private, sign-transformed orders derived from
the public CPU and GPU feature fixtures. It uses one disposable setup,
one proof per order and an empty read-only arbiter container for each
delivery. Each container receives only the commitments, proof,
verification key and separately pinned hashes. The customer opens
the adapter independently. Refusal controls swap the claimed template,
issue a fresh commitment for a mismatched template, and swap proofs
between orders. The test also checks a previous single-template proof
against its original issued order and key pins.

## Results

Both orders passed in separate offline arbiter containers under the
same 328-byte key. Both customer openings matched. The selector
change failed commitment checking. The fresh incompatible order
passed commitment checking and failed the arithmetic circuit. The
arbiter also refused a proof copied from the other order. The prior
single-template proof still verified with its original pins.

| Measurement on the LAN Ryzen 5 5500 | Result |
| --- | ---: |
| Constraints and public inputs | 347,690 and 2 |
| One setup, end to end | 11.62 s |
| Proving key | 71,338,640 bytes |
| Cold proof per order, end to end | 41.19–41.37 s |
| Cold key loading per order | 35.07–35.08 s |
| Peak prover RSS | 1.29 GiB |
| Arbiter verification inside the binary | 3.33–3.36 ms |
| Arbiter with container launch | 246–254 ms |
| Commitments plus proof per order | 192 bytes |
| Two orders plus one verification key | 712 bytes |

The measured single-template CPU setup used 193,153 constraints and
~23 s of cold proving per order. The hidden choice leaves the
arbiter's input count and proof size unchanged. The extra branch
raises setup and proving costs. Two Rayon threads ran the prover.
Seven Rust tests and fourteen Python tests passed. A fresh
single-template constraint census still returned 193,153.

[Public package, per-step timings and authenticated pins](results/private-template-pair/report.json).
The 712-byte figure excludes the verifier binary and the separately
distributed key and order fingerprints. The retained proof artifacts
and key contain no saved openings.

The [shared-key follow-up](SHARED_PROVING_KEY.md) measures two proofs
generated in one process using a single checked key load.

## Disclosure and reproducibility

The public arbiter knows the pair of admissible templates and the
authentic issued commitment. It receives no template choice, numeric
order, adapter or opening. The customer and worker know the selected
template and the order. The trial relies on the hiding of the salted
commitment and on the zero-knowledge property of Groth16 under
its setup assumptions. Side-channel resistance, a larger template set,
private input from the executing worker and payment privacy have
separate qualification scope.

Build with the pinned Rust lockfile and run the experiment using a
preloaded Ubuntu 24.04 Docker image.

```sh
cargo build --release --locked --manifest-path compact-round/Cargo.toml
python3 -B compact-round/private_pair_trial.py \
  compact-round/target/release/compact-lora-round NEW_OUTPUT
```
