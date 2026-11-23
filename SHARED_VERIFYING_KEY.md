# One prepared verification key for two private Qwen proofs

The two private Qwen orders in the [shared prover trial](SHARED_PROVING_KEY.md)
use a single verification key. The new verifier command pins and decodes
the key once and checks both proof equations independently. Each order
still has its own trusted order pin and both canonical encodings are checked.
The command reports success only after both checks pass. Each delivered
proof remains the original 128 bytes.

## Results

| Verification path | Median wall time for two proofs |
| --- | ---: |
| Two `private-prefix-verify` processes | 8.37 ms |
| One `private-prefix-verify-two` process | 6.72 ms |

The measured reduction was 1.65 ms, about 20%. The LAN Ryzen 5 5500 used
one pinned logical CPU and one Rayon thread. Each path had five warm-up
pairs and 64 measured pairs. The order of the two paths alternated. Each
measurement includes process startup and checked key decoding. It begins
with both proofs ready for verification, so it excludes queue wait.

Both earlier proof pairs passed. The shared path refused a wrong key pin,
a wrong second order pin, reusing the first proof for the second order,
an extra byte and a truncated second proof. All five failures exited with
code 1 and no success result. The offline verifier also passed in a
network-disabled Ubuntu container containing the binary, the pinned
verification key, and the two sets of public commitments and proofs.
Seven Rust tests passed. The retained public v1 proof and the CPU v2
proof also passed through the refactored key loader.
[Timings and refusals](results/shared-verifying-key/report.json) ·
[Offline verifier output](results/shared-verifying-key/offline.log) ·
[Rust tests](results/shared-verifying-key/tests.log) ·
[Public controls](results/shared-verifying-key/public-controls.log) ·
[Source and binary hashes](results/shared-verifying-key/environment.txt).

The comparison covers two ready proofs for one key on this LAN CPU.
Arrival schedules and verification on a phone remain unmeasured.

## Independent decisions for three ready proofs

`private-prefix-review-three VK VK_PIN ORDER_PROOF ORDER_PROOF ORDER_PROOF`
uses the same prepared key for three separate proof checks. Each
`ORDER_PROOF` contains the public commitments, the separately issued
order pin and the proof path. The result has one decision per input,
in input order, with its order commitment digest and a Boolean
`accepted` field. Applications use each order's decision.

An equation that evaluates to false produces a false decision for
that order. A missing file, wrong pin, malformed canonical encoding
or verifier error stops the command with exit code 1 and no decision
document. The original `private-prefix-verify-two` and
`private-prefix-verify-three` commands still require every proof
equation to pass before producing their success record.

Seven Rust tests and the retained two-proof CLI run passed on the
LAN Ryzen. The paired Worknet trial used 288 ready trios and checked
four input-error controls. This is a local public-artifact verifier
interface for the fixed shared-key workload.

## Independent decision for one ready proof

`private-prefix-review-one VK VK_PIN COMMITMENTS ORDER_PIN PROOF`
returns the same decision format with `proof_count` equal to one.
A well-formed proof that fails verification returns a false
decision. Input and verifier errors stop with no decision. This
allows a recipient to check a ready order when no trio is ready.

On LAN Ryzen, three single reviews of retained Worknet downloads
matched the trio's two accepts and one refusal. Wrong key and order
pins and an extra proof byte each stopped without a decision. All
seven Rust unit tests passed. The tested Linux binary SHA-256 was
`f679261a5121e1a24b7f37bec093442175b5633cf8828d8aad06f6afd31715b7`.

## Reproduce

Build the locked binary and verify the published pair. The pins below are
for the retained trial. Use the independently supplied pins for new orders.

```sh
cargo build --release --locked --manifest-path compact-round/Cargo.toml
bin=compact-round/target/release/compact-lora-round
kit=results/shared-proving-key
"$bin" private-prefix-verify-two \
  "$kit/verifying.key" 52945e1e41a7569e149e410ff8f84e848d7f6a6d0307887f226cf2a0f2fdecf5 \
  "$kit/order-1/round.commitments" f4d1bcbeab6c5a83f1ba843d2515864af1ad69ae92ef18943610a4996150b16e \
  "$kit/order-1/shared-key.proof" \
  "$kit/order-2/round.commitments" 734aafd5570516636cc8959f1691c0eb25c3f6208afb4f6f0d5e6b430d375468 \
  "$kit/order-2/shared-key.proof"
```
