# An integer LoRA round on frozen Qwen features

## Protocol fixed before feature capture

The study asks whether the existing two-phase verified LoRA round improves
a small topic head attached to a frozen pretrained text model. The backbone
is Qwen2.5-0.5B-Instruct Q4_K_M, GGUF SHA-256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
The input is a self-contained, newly written English news-brief corpus.
The [fixed corpus](results/text-head-v1/corpus.tsv) has 16 examples in each
of three disjoint splits, with eight sports and eight weather labels per
split. The source split fits the frozen linear head. The adaptation split
drives the two LoRA phases. The held-out split measures utility.

One decoder call per text captures the last token's final hidden state,
with the KV memory cleared between texts. The probe saves 32 equally
spaced coordinates from its 896-wide row. For each coordinate, the source
split fixes the mean and population standard deviation. Multiply centered,
standardized values by four, round to nearest with ties to even and clip
to ±16. A zero standard deviation uses divisor one. The source sports
and weather centroids of the integer features fix opposite columns for
the frozen topic head. Each weight is half their difference rounded to
nearest with ties to even. Its other 30 columns are zero.

Use the existing 32-wide rank-4 round, step scale eight, and two sign-SGD
phases. Initialize B to zero and put a single one at input coordinate
0, 8, 16 or 24 in each respective row of A. Targets are `[96, -96]` for
sports and `[-96, 96]` for weather, followed by zeros. Each phase uses
two consecutive four-example adaptation slots. The manifest's squared
error ceiling follows the bounded arithmetic and permits certifying a
correct round irrespective of task utility. The prespecified utility
gate requires both an increase in held-out top-1 count and no increase
in held-out squared error. Ties in head score choose sports. A verified
round can fail this utility gate. No features, hyperparameters or texts
will be selected after examining held-out scores.

The verifier binds the classifier arithmetic and optimizer transitions
to the pinned integer feature manifest. Feature provenance requires
repeating the separate Qwen capture on the pinned model and runtime.
