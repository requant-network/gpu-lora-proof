# Signed gradient updates in two Qwen blocks

The [sharded projection trial](SHARDED_PROJECTION.md) distributed
candidate scoring. This round uses reverse-mode differentiation to train
the A and B factors of the last two attention output projections in
Qwen2.5-0.5B-Instruct. The four trained tensors have 3,584 coordinates.
The base weights are frozen, and the ordered factors are A22, B22, A23,
and B23. The base is the publisher's [pinned Safetensors checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2), loaded in FP32.

## Ordered computation

Each slot calculates the sum of per-text binary cross-entropy gradients
on eight fixed adaptation texts. The two-class loss uses the logits of
the single tokens for sports and weather. The first step starts with
fixed nonzero A factors and zero B factors. The workers sign the gradient
and loss in ticks of 2^-18, binding their document IDs, step, order,
and the current factor digest. The coordinator sums the signed integers.
It applies an SGD step of 1/32 to the mean over all 16 texts using
round-to-nearest-even on the same grid. The second step binds the full
first transition, including both contributions. The 16 held-out texts
enter the release gate after the two SGD steps.

The CPU verifier recalculates the four gradients. Each coordinate and
the summed loss have a 32-tick absolute bound. The gradient also has a
relative L2 bound of 1/256 using a reference norm floor of 0.01 in gradient
units. The verifier derives both exact optimizer states from the accepted
signed integers. It releases the final state if held-out loss improves
and the count of correct labels holds. The certificate commits to the
four signed contributions, the released tensors and their loader
configuration. Gradient agreement within these bounds is part of this
experiment's numerical policy. Different CPU libraries can produce
different measured evaluation losses on the same released state.

## Measurements

| Slot | Physical executor | Step one | Step two |
| --- | --- | ---: | ---: |
| Left | Linux RTX 5060 Ti | 3.72 s | 3.06 s |
| Right | Windows Ryzen 7 8745HS CPU | 4.39 s | 3.99 s |

The first step changed 1,690 B coordinates. The second changed 2,892
coordinates, including 1,203 across both A factors. The four signed
records total 62,657 bytes. The released Safetensors file is 14,848
bytes.

The LAN Ryzen 5 5500 verified the round in 19.20 s in a fresh CPU
container with networking disabled and read-only model and contribution
mounts. The Windows CPU verification took 18.25 s and supplied each pair
in reversed file order. The largest gradient-coordinate difference in
either verifier was one tick. Both produced the same Merkle root, tensor
file, and adapter configuration.

| CPU verifier | Frozen loss ticks | Step one | Step two | Correct at every step |
| --- | ---: | ---: | ---: | ---: |
| LAN offline container | 1,939,308 | 1,937,465 | 1,935,294 | 13/16 |
| Windows | 1,939,298 | 1,937,452 | 1,935,286 | 13/16 |

The held-out loss dropped by about 0.21%. A fresh PEFT 0.19.0 load
matched all four released tensors and observed the same improvement
in its own causal-LM forward pass. Both executors were operated by the
owner. The evidence covers two SGD steps and the fixed binary objective.
Independent operator acceptance and broader evaluation remain open.

[Approved order](results/gradient-round/order.json) ·
[Linux certificate](results/gradient-round/certificate.json) ·
[Windows certificate](results/gradient-round/certificate-windows.json) ·
[Released tensors](results/gradient-round/adapter_model.safetensors) ·
[PEFT configuration](results/gradient-round/adapter_config.json) ·
[Linux replay](results/gradient-round/replay-linux.json) ·
[Windows replay](results/gradient-round/replay-windows.json) ·
[PEFT loading check](results/gradient-round/peft-check.json)

## Controls

An edited gradient failed signature verification. A properly signed
second-step record with the wrong first-transition digest failed
preflight. A properly signed gradient with one coordinate shifted by
1,024 ticks reached model replay and was refused. A duplicate slot
failed preflight. None of these runs wrote a release directory. Changing
the released LoRA scaling parameter failed the PEFT configuration pin.
All 14 existing unit tests passed on Windows and Linux.

[Round controls](results/gradient-round/controls.txt) ·
[Changed scaling control](results/gradient-round/peft-control.txt)

## Reproduce

Prepare the four files listed in the order in a dedicated model directory.
Obtain the file hashes and the order pin separately from the supplied
contributions. The weights and tokenizer files used here come from
Qwen revision `eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2`. The Linux
profile used Python 3.12.3, PyTorch 2.14.0+cu130, Transformers 5.20.2 and
Safetensors 0.8.0. The Windows profile used Python 3.12, PyTorch
2.14.0+cpu, Transformers 5.17.0 and Safetensors 0.8.0. Both use
Cryptography for the Ed25519 records. The CPU profile used three
model threads, MKL compatibility mode AVX2, and eager attention.

Prepare a fresh round ID and two Ed25519 keypairs for a new training
round. Keep each private PEM with its slot and replace the two pinned
public keys with the raw 32-byte public key hex values.

```sh
python -B experiments/gradient_round.py worker ORDER CORPUS MODEL LEFT_FIRST \
  --order-sha256 ORDER_SHA --device cuda --slot left --key LEFT_PEM
python -B experiments/gradient_round.py worker ORDER CORPUS MODEL RIGHT_FIRST \
  --order-sha256 ORDER_SHA --slot right --key RIGHT_PEM
```

Use the same commands with fresh output paths and
`--first LEFT_FIRST RIGHT_FIRST` for step two. On the GPU worker, expose
only the selected RTX 5060 Ti with `CUDA_VISIBLE_DEVICES`. On the CPU
worker and verifier, hide CUDA devices and set `MKL_CBWR=AVX2`.
All loaders use local files. The LAN verifier ran under Ubuntu 24.04
image `sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55`.

To replay the retained contributions, use the pinned
[corpus](results/text-head-v2/corpus.tsv), the same model directory,
and a new output directory.

```sh
r=results/gradient-round
pin=f1a863d61c5418713c48900c67f5384750fd6c0005e4200fd00c7ce415ccb47b
python -B experiments/gradient_round.py verify "$r/order.json" \
  results/text-head-v2/corpus.tsv "$MODEL" replay \
  --order-sha256 "$pin" --first "$r/left-first.json" "$r/right-first.json" \
  --second "$r/left-second.json" "$r/right-second.json"
python -B experiments/check_gradient_peft.py "$r/order.json" \
  results/text-head-v2/corpus.tsv "$MODEL" replay --order-sha256 "$pin"
```
