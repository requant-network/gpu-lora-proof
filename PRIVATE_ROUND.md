# A private order for a public arbiter

The November 21 trial verified one complete synthetic LoRA round while keeping
the numeric order and recovered adapter outside the public arbiter package.
An offline Ubuntu 24.04 container accepted a 128-byte Groth16 proof using two
public commitments. The customer separately opened the committed adapter.
[Public evidence](results/private-round/report.json).

## Disclosure and trust

| Role | Information available in this trial |
| --- | --- |
| Customer | Numeric order and its opening, returned adapter and its opening, public commitments and proof. The customer pins the issued order commitment and checks the returned adapter. |
| Worker or prover | Plaintext numeric order, starting weights, features, targets and resulting adapter. The prover uses both openings to build the proof. |
| Public arbiter | Both commitments, the proof, the verification key and the two expected pins from issuance and setup. The proof reveals whether the committed adapter satisfies the fixed round contract. |
| Public observer | The published package and its timing and repetition. The operation was tested off chain. |

The commitments use independent salts sampled by the owner and prover from the
BN254 scalar field. The setup holder uses the circuit-specific proving and
verification keys produced for this trial. The arbiter authenticates the
verification key pin separately. Repeated publication of one commitment
remains linkable. Independent custody of setup randomness, production key
distribution and privacy from the plaintext worker require separate work.

## What the circuit verifies

The input commitment binds the ordered numeric projection of the round:
base weights, starting factors, training features and targets, and held-out
features and targets. This fixed circuit supplies the schedule, dimensions and
zero-error gate. Raw JSON identifiers and formatting stay outside the numeric
commitment. A SHA-256 pin of the issued commitment identifies the order for
the arbiter and customer. The output commitment binds the recovered adapter
and the same input commitment.

The circuit checks private input ranges, exact separation of held-out rows from
training rows, the two bounded optimizer transitions and every held-out result.
After the range constraints, it packs 20 signed values into each scalar using
12-bit offset digits. The order and output use distinct Poseidon domain tags.
Each has its own salt, and the output sponge absorbs the order commitment.
The program uses Arkworks Poseidon with BN254 width 3, rate 2, capacity 1,
exponent 5, eight full rounds and 57 partial rounds. A unit test matches the
generated round constants and MDS matrix against the fingerprint of the
published table. [Arkworks constrained sponge](https://docs.rs/crate/ark-crypto-primitives/0.5.0/source/src/sponge/poseidon/constraints.rs) ·
[Pinned Poseidon parameter table](https://github.com/iden3/circomlibjs/blob/48b3ab37013c5ed21e9ff8a80a5b010795c97094/src/poseidon_constants.json).

## Observed run

| Item | Measurement |
| --- | ---: |
| Constraints / public inputs | 146,189 / 2 |
| Commitments / proof / verification key | 64 / 128 / 328 bytes |
| CPU proof generation, including key load | 15.05 s / 482,760 KiB peak RSS |
| CPU proof computation and write, after key load | 2.74 s |
| One disposable setup | 4.66 s / 31,268,912-byte proving key |
| Arbiter binary verification / including container launch | 3.29 ms / 246 ms |
| Customer opening inside the verifier binary | 2.15 ms |

The arbiter container ran as an unprivileged user with networking disabled,
a read-only root filesystem, and read-only mounts of the verifier executable
and three public files. The witness directory was temporary. Fresh salts
changed both commitments for the same private data. Eight controls rejected
swapped commitments, relabelling the order pin on an old proof, changed input
and output openings, a wrong key pin, relinking the adapter to another order,
and trailing proof bytes. Five circuit tests passed, including the private
range and held-out separation checks. The earlier public LoRA proof still
verified, and the revised binary generated another accepted proof with its
existing key. [Run and compatibility log](results/private-round/checks.log).

## Verify the published package

The verification command takes the order commitment pin and verifying key
pin separately from the proof. These pins correspond to the retained trial.
[Commitments](results/private-round/round.commitments) · [Proof](results/private-round/round.proof) ·
[Verifying key](results/private-round/verifying.key).

```sh
cargo build --manifest-path compact-round/Cargo.toml --locked --release
compact-round/target/release/compact-lora-round private-verify \
  results/private-round/round.commitments \
  f71d56e21da7d1028e30600db875e122a7d56e19b589b5db373b326dfe11cba2 \
  results/private-round/verifying.key \
  a2c79287f13ce07734d7d74c9ecc39f65d97070ab8449f8654adc3e72298a5ae \
  results/private-round/round.proof

python3 -B compact-round/private_trial.py \
  compact-round/target/release/compact-lora-round ./new-private-trial
```

The reproduction script uses a locally available `ubuntu:24.04` Docker image
and a new output directory. It generates a fresh synthetic witness and new
salted commitments. The public evidence and performance report are retained
after the private scratch directory is removed.
