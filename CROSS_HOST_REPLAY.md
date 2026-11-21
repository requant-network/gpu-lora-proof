# Cross-host replay of the Qwen CPU adapters

November 21, 2026. The server ran the original [feature verifier](FEATURE_ORIGIN.md)
and [transformer projection trial](TRANSFORMER_PROJECTION.md) on an AMD
Ryzen 5 5500. The workstation replayed both on an AMD Ryzen 7 8745HS via
WSL2. The second machine received the same model, public corpus, trusted
recipe and existing certificate. Every inference and optimizer
decision in the replay ran on its own CPU.

The workstation used a temporary userspace populated from the server's
Ubuntu 24.04 binaries. It included the exact Haswell CPU backend,
four pinned model libraries, both native probe executables, Python
3.12.3, GNU libc 2.39 and the necessary runtime dependencies. The
workstation ran this userspace under its existing WSL Debian 12
installation using the copied dynamic loader, with no package or
system changes. The commands used a niceness adjustment of +12 and CPUs
0–3. The model was read from a Windows-mounted drive.

| Replay | Server wall time | Workstation wall time | Exact match |
| --- | ---: | ---: | --- |
| Frozen Qwen head with four v2 contributions | 9.224 s | 14.530 s | Full feature table, manifest and integer adapter |
| Rank-one update to the last attention output projection | 35.912 s | 67.552 s | Entire search record, held-out feature digest and GGUF adapter |

The workstation selected coefficient 8 again. The projected update
retained 16/16 held-out labels and reduced squared error from 114,328
to 97,610. The head replay recovered the existing 16/16 adapter with
squared error 72,888. The timings cover each complete run as executed
with its own scheduling and storage.
[Head replay](results/cross-host-replay/wsl-head.json) ·
[Projection replay](results/cross-host-replay/wsl-projection.json) ·
[Binary and environment fingerprints](results/cross-host-replay/wsl-environment.txt) ·
[Emitted artifact digests](results/cross-host-replay/wsl-output-sha256.txt).

The result covers two physical CPUs with matching backend and runtime
bytes and the same public workload. Independently built binaries,
different CPU instruction profiles and an independent operator remain
separate qualification gates.

## Reproduce

Use the [feature replay recipe and commands](FEATURE_ORIGIN.md#reproduce)
and the [projection replay command](TRANSFORMER_PROJECTION.md#reproduce)
on both hosts. Select the CPU recipe and compare the emitted feature,
manifest, certificate and adapter digests with the retained results.
Record each processor and the runtime component hashes.

On a WSL distribution with an older system libc, the verified run put
the matching runtime loader, shared libraries, local Python and its
stdlib in a temporary directory. The two probe launchers invoked the
copied loader directly. They set its library path to the copied system
and model libraries. The Python launcher used the copied stdlib and
the same system loader. The archived runtime fingerprint and the ELF
component digests are recorded with the results.
