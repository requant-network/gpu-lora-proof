# Reuse one proving key for three private orders

## Protocol fixed before execution

Create three ready orders under the same disposable key and fixed
[three-template roster](PRIVATE_TEMPLATE_TRIPLE.md). Run a cold process
for each order, then run one process on the same three orders and checked
key. Use the same issued order commitments and linked output commitments
in both paths. Exclude setup from the two prover clocks. Retain wall time,
checked key load, resident memory and proof bytes.

Validate each order before touching the proving key. Prove all three
statements with independent randomness. Publish one file only after all
three proofs have been produced, in request order. Split the file into
the three canonical proofs and verify each with its original issued pin,
commitments and verification key in a network-disabled container.

Use a newly issued incompatible order as the third item in a separate
batch. Supply an absent key path and check that validation refuses the
order before key access and leaves no output file. The two-order command
continues to write two canonical proofs.

## Results

| LAN Ryzen 5 5500, two Rayon threads | Three cold processes | One process |
| --- | ---: | ---: |
| Total prover wall time | 162.93 s | 66.58 s |
| Total checked key loading | 141.10 s | 46.95 s |
| Peak process RSS | 1.65 GiB | 1.65 GiB |
| Canonical proof size per order | 128 bytes | 128 bytes |

Wall time decreased by 96.34 s (59.1%) in this matched trial. Each
of the six proofs passed against its own issued order commitment in a
fresh offline arbiter. The proof file in the shared path was exactly
384 bytes. Each order still has its own proof and its original linked
output commitment.

The incompatible third item failed the arithmetic check before any
proving-key load. The requested output did not exist after refusal.
The retained single-template and pair proofs still verified. The pair
circuit retained 347,690 constraints. Seven Rust and fourteen Python
tests passed. [Measured results and proof files](results/shared-triple-proving-key/report.json) ·
[Source and runtime pins](results/shared-triple-proving-key/environment.txt).

The batch measurement starts with all three orders ready. The shared
process publishes the first proof with the other two at completion.
Queue wait and time to the first individual delivery remain separate
latency considerations. Each path uses 904 public bytes for the proofs,
commitments and shared verifier key. The retained comparison package
contains the alternate proofs from both paths.

## Reproduce

Build the pinned research binary and run the three-template harness
with key reuse enabled.

```sh
cargo build --release --locked --manifest-path compact-round/Cargo.toml
python3 -B compact-round/private_pair_trial.py \
  compact-round/target/release/compact-lora-round NEW_OUTPUT \
  --three-templates --compare-key-reuse
```

The environment file hashes the protocol as fixed before the trial.
