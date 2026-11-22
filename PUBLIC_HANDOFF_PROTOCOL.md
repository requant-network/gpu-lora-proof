# Compare delivery of a public Qwen adapter claim

Use the two existing accepted Qwen v2 feature manifests and
their returned adapters. Compare three ways to validate the
same public final adapter and release gate. Direct replay
recomputes both optimizer steps from the pinned public
manifest. The transparent verifier also checks the supplied
step matrices, contributions and round root. Groth16
checks the adapter statement using a separately pinned
verification key. Record the extra information present in
the transparent certificate.

Generate a fresh disposable proof and key for each
manifest. Measure setup separately, then measure producer
proof construction and transparent assembly. Package the
manifest and claimed adapter in all three forms. Include
the extra certificate or proof and compare gzip package
sizes. Cover both a cached verification key and its first
download. Time unpacking and verification on Xeon in four
rotations per form. Use the same CPU service and interpreter
without network access. Reject a changed claimed adapter.

Compare the measured work and byte budgets before making
a recommendation for this public task. Treat any transfer
duration computed from byte counts and a hypothetical link
rate as a model. This trial covers the released feature
manifests. Feature extraction and executor identity are
separate parts of the original system.
