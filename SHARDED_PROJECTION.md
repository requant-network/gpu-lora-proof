# Two-host signed LoRA projection round

This trial distributes the adaptation scoring for the
[two-phase Qwen projection](PROJECTION_TWO_PHASE.md).
The model, tokenizer, CPU runtime bytes, 48 public documents,
frozen head and candidate grids follow that experiment.
The adaptation split is divided into two fixed groups of eight
texts. The LAN Ryzen 5 5500 and the workstation Ryzen 7 8745HS
under WSL2 each run one group.

## Round and acceptance

A pinned order assigns one Ed25519 public key to each group. A signed
contribution binds the round, phase, document IDs, optimizer input and
candidate scores. Both workers score every candidate on their own
documents. Adding the integer loss totals reconstructs the adaptation
objective. The deterministic rules select B first and then one additional
A coordinate. The second contribution set binds the selected B adapter
and the digest of the complete accepted first transition.

The verifier requires exactly one signed contribution per slot per phase.
It recreates each individual score and feature digest, recomputes the
selections, then evaluates the held-out texts separately. It releases the
second adapter only if it improves on both the frozen and first-phase
references and preserves their correct-label count. If the second phase
fails this gate, the first adapter remains eligible on its own frozen-base
gate. Four ordered Merkle leaves bind the accepted contributions and
their phases into the final certificate.

Each host uses a separate fixture signing key. The replay verifier
receives the approved order digest and needs only the public keys.
The two hosts are operated by the same owner.

## Results

The signed shard totals reproduced all nine B candidate scores
and all fifteen A candidate scores of the archived monolithic
run. B selected coefficient 8. A selected +1/4 at coordinate 784.

| State | Adaptation squared error | Held-out squared error | Held-out labels |
| --- | ---: | ---: | ---: |
| Frozen baseline | 85,712 | 114,328 | 16/16 |
| Phase one, B | 71,784 | 97,610 | 16/16 |
| Phase two, B and A | 50,108 | 99,072 | 16/16 |

The verifier released phase one. Its 7,488-byte GGUF matched the
previous single-projection adapter byte for byte. The four signed
records total 13,576 bytes and the replay certificate is 18,741 bytes.

| Worker | Phase one | Phase two |
| --- | ---: | ---: |
| LAN Ryzen 5 5500 | 32.24 s | 38.01 s |
| Workstation Ryzen 7 8745HS under WSL2 | 54.04 s | 78.12 s |

The LAN verifier replayed all four contributions in 176.10 s
inside a network-disabled, read-only container. It received
only the public order, signed records and pinned model recipe.
The runtime and model were mounted read-only. The verifier
wrote the certificate and the eligible adapter in its result directory.
The WSL verifier took 240.74 s and produced identical certificate
and GGUF bytes, with each input pair supplied in the opposite file
order from the container run.

The duplicate slot, missing slot, contribution edited after signing
and contribution from another round each failed preflight. A signed phase-two input
with the wrong first-transition digest also failed preflight.
An altered loss with a valid slot signature passed preflight and
failed the independent model replay before publication. The first
attempt at that control reached the feature-capture timeout on
a different CPU affinity. Repeating it in the same isolated CPU
configuration as the honest replay reached the expected arithmetic
refusal. All fourteen existing Python tests passed.
[Pinned order](results/sharded-projection/order.json) ·
[Certificate and Merkle root](results/sharded-projection/certificate.json) ·
[Released adapter](results/sharded-projection/adapter.gguf) ·
[Container replay](results/sharded-projection/arbiter-summary.json) ·
[WSL replay](results/sharded-projection/arbiter-wsl-summary.json) ·
[Preflight controls](results/sharded-projection/early-controls.txt) ·
[Signed broken linkage](results/sharded-projection/signed-broken-link.txt) ·
[Signed false score](results/sharded-projection/false-score-replay.txt) ·
[Initial deadline](results/sharded-projection/initial-score-control-timeout.txt) ·
[Python tests](results/sharded-projection/python-tests.txt)

## Reproduce

Use the [pinned CPU projection runtime](TRANSFORMER_PROJECTION.md#reproduce)
with the writer that supports the optional input coordinate. The
[WSL profile](CROSS_HOST_REPLAY.md#reproduce) uses an identical native
runtime under a copied Linux loader. Both hosts need OpenSSL with
Ed25519 support. Supply the same order and its independently obtained
digest to each worker and verifier. The retained order uses disposable
public keys. For a fresh experiment, create two new keys and a new round
identifier, then prepare a corresponding pinned order.

```sh
openssl genpkey -algorithm ED25519 -out worker.pem
openssl pkey -in worker.pem -pubout -outform DER -out worker.der
```

Use the hex encoding of each DER public key in the order. Keep each
private PEM at its worker.

The worker command accepts `ORDER RECIPE CORPUS MODEL LIB PROBE WRITER OUTPUT`.
Use `--slot left` or `--slot right` with its private `--key`.
Start both workers with `--order-sha256 ORDER_SHA` to generate phase one.
Run both again with `--first LEFT_FIRST RIGHT_FIRST` and new output
paths to generate phase two. Run the same command in `verify` mode
with a fresh output directory, `--first LEFT_FIRST RIGHT_FIRST` and
`--second LEFT_SECOND RIGHT_SECOND`. The verifier writes the certificate
and the selected eligible GGUF after replay.
[LAN runtime pins](results/sharded-projection/lan-environment.txt) ·
[WSL runtime pins](results/sharded-projection/wsl-environment.txt)
