# Private Qwen prefix release under a disclosed template

November 21, 2026. This trial extends the [private v1 order
experiment](PRIVATE_ROUND.md) to the exact [v2 prefix and quality
rules](COMPACT_PREFIX.md). Use the existing public Qwen CPU
[template](results/compact-prefix/cpu/shape.json) for both orders. It
discloses the active coordinate lists in gradient order and the
sixteen target label positions. Setup uses that template before creating
the two private variants. The variants apply independently chosen
feature-column signs to the CPU fixture and corresponding signs to
both the frozen head and factor A. This preserves the exact ordering
and predictions under the fixed template while changing the numeric
order and returned adapter.

## What each party receives

| Party | Data available in this trial |
| --- | --- |
| Customer | Its plaintext order, order salt, returned adapter and output salt through the private opening path. The customer authenticates its issued commitment and checks the returned adapter's opening. |
| Worker and prover | Plaintext order, output adapter and openings, disclosed template and proving key. |
| Public arbiter | The order and output commitments, proof and trusted verification-key pin. The fixed template is publicly available and the arbiter knows which contract is being verified. |
| Public observer | The same published artifacts and template as the arbiter, with public run timings. The operator used an off-chain verifier. |

The template constrains which inputs can be proved with this key. The
numeric weights, features, targets, squared-error ceiling, chosen
prefix lengths and adapter stay in the witness. The template's fixed
labels and active order are public knowledge. The run uses variants
of a published fixture to exercise the protocol. Real customer
confidentiality against inferences from public data or the disclosed
template requires its own evaluation.

## What the proof verifies

The order commitment absorbs the public template digest, the private
error ceiling, an independent salt and the packed numerical order.
The second commitment links the adapter to that order and adds its
own independent salt. Separate domain tags preserve the v1
commitment contract. The circuit checks input ranges and disjoint
held-out rows, then evaluates all eligible prefixes and applies the
earliest minimum in each phase. It checks the private squared-error
ceiling and 16/16 correct held-out labels.

The proof uses a lab setup key generated from the disclosed template.
It gives the arbiter an arithmetic decision on the committed order.
The worker sees the plaintext, and the trial assumes honest setup
and separate transport of the customer opening. Arbitrary hidden
coordinate orders or label positions need a circuit with different
witness routing and a new setup.
[Groth16 implementation](https://docs.rs/ark-groth16/0.5.0/ark_groth16/) ·
[Constrained Poseidon sponge](https://docs.rs/crate/ark-crypto-primitives/0.5.0/source/src/sponge/poseidon/constraints.rs).

## Observed run

The two variants reused one verification key and both passed inside
separate offline, read-only Ubuntu containers. Each arbiter container
mounted only its commitments, proof, verification key and verifier
executable. The customer opened both adapters outside the arbiter.
No witness files or salts remain in the published package.

| Item | Observation on Linux Ryzen 5 5500 |
| --- | ---: |
| Constraint count and public inputs | 193,153 and 2 |
| Commitments and proof per order | 64 and 128 bytes |
| Shared verifier key | 328 bytes |
| Two orders with one key | 712 bytes |
| One setup and size of proving key | 6.254 s and 38,652,112 bytes |
| Cold proving per order, including key load and check | 22.973–22.995 s |
| Arbiter verification inside the binary | 3.31–3.34 ms |
| Verification including container launch | 247–261 ms |
| Customer adapter opening | 2.15–2.17 ms |

The byte counts cover the commitments, proofs and key. Verifier
software, the public template and authenticated pins are distributed
separately. The seven refusal controls covered cross-order proof
and adapter links, changed output opening, changed ceiling, changed
template and newly issued orders that failed the accuracy or error gate.
For the accuracy control the prediction of one held-out row changed
while its squared error stayed within the revised ceiling. A fresh
salt changed the same order's commitment. The retained public v2 and
private v1 proof packages still verified. Seven Rust tests and fourteen
Python tests passed on Linux.
[Public package and timings](results/private-prefix/report.json).

## Reproduce

Build from the locked Rust manifest and run the public-only trial
using a locally available Ubuntu 24.04 Docker image.

```sh
cargo build --manifest-path compact-round/Cargo.toml --locked --release
bin=compact-round/target/release/compact-lora-round
python3 -B compact-round/private_prefix_trial.py "$bin" ./new-private-prefix-trial
```

The refreshed trial generates one key and two sets of salts. Verify
the retained artifacts with the order commitment pin and the key pin
issued separately. The pins in the retained report describe this
specific lab run.

```sh
root=results/private-prefix
vk=ee8df7762657b45a02ab9956a4359cf5052f5310bd47f43b68d49af1ec4838c3
"$bin" private-prefix-verify "$root/order-1/round.commitments" \
  34b31a6a624ea8f0f49711492819debd9697c23871c95ebf1816ec28c45108e7 \
  "$root/verifying.key" "$vk" "$root/order-1/round.proof"
"$bin" private-prefix-verify "$root/order-2/round.commitments" \
  9a08ac79859687c1f9d03c141ab43ea35d8b3f2f4b3f37f73f29d2bba88e3a59 \
  "$root/verifying.key" "$vk" "$root/order-2/round.proof"
```
