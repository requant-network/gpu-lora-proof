# One validated proving key for two private Qwen orders

The [two-template proof](PRIVATE_TEMPLATE_PAIR.md) spent most of the
cold prover time reading and validating its 71 MB arkworks proving key.
This follow-up reuses the decoded key in one process for two requests
with the same circuit. Both orders still undergo the existing pinned
manifest, commitment and circuit checks. Those checks complete for both
orders before key loading. The prover uses independent randomness for
each proof and writes the two canonical 128-byte proofs in request order
after both are generated. Each proof can be verified separately using
its original order pin, commitments and verification key.

The LAN comparison uses one setup and the same two private orders,
commitments and proving key for both paths. Run two separate prover
processes first, then run one process for both orders. Record elapsed
process time and peak RSS, including normal checked key decoding.
Verification runs in separate offline containers and is measured
outside those times. Submit a freshly committed incompatible order
second alongside a valid first order. Require an arithmetic refusal
before accessing a deliberately absent key and confirm that the pair
output is absent.

## Results

| Ryzen 5 5500, two Rayon threads | Separate processes | One process |
| --- | ---: | ---: |
| Total prover wall time | 82.82 s | 46.77 s |
| Elapsed time loading and checking the key | 70.25 s | 35.07 s |
| Peak process RSS | 1.290 GiB | 1.306 GiB |
| Proof bytes per order | 128 | 128 |

Total proving time fell by 36.05 s (43.5%) in the matched
two-order run. The measured shared-key path keeps the same prover and
offline per-order verifier. Both separately generated proofs and both
proofs from the shared process passed against the same two sets of
commitments. Both adapter openings matched. The incompatible second
order failed before key loading and produced no public output. Earlier
cross-order and template rejection controls still passed, as did a
retained single-template proof, seven Rust tests and fourteen Python
tests.

The separate path reports the sum of the two prover process durations.
The shared path reports one duration. The table uses the maximum of
the two separate RSS values and the peak of the shared process.
Both paths used checked key decoding and a local filesystem. The
measurement is one matched run with both orders ready, excluding
setup and verifier time. It measures batch completion time. Waiting
for another order and time to individual delivery in a live queue
remain separate scheduling questions.

[Measurements and both proof pairs](results/shared-proving-key/report.json).
The retained package includes both comparison paths. A delivery under
either one still needs 192 bytes per order with a cached verifier key.

## Reproduce

Use the locked Rust dependencies and the preloaded Ubuntu 24.04 image.
The CLI command `private-prefix-pair-prove-two` writes 256 bytes for the
two independent proofs. Run the comparison and independent checks with:

```sh
cargo build --release --locked --manifest-path compact-round/Cargo.toml
python3 -B compact-round/private_pair_trial.py \
  compact-round/target/release/compact-lora-round NEW_OUTPUT --compare-key-reuse
```
