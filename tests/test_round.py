import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prover
import round_experiment as experiment
import round_verifier as V


class TestRound(unittest.TestCase):
    def test_cli_requires_trusted_manifest_and_complete_round(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            manifest, teacher = experiment.fixture(20261121)
            expected, bundle, recovered = (root / f"{name}.json" for name in ("manifest", "bundle", "adapter"))
            V.save_json(expected, manifest)
            _, pin = V.read_json(expected, V.MANIFEST_BYTES)
            evidence, _ = experiment.assemble(expected, bundle, None)

            def verify(path, manifest_pin):
                return subprocess.run(
                    [sys.executable, "-B", str(ROOT / "round_verifier.py"), str(expected), str(path),
                     "--manifest-sha256", manifest_pin, "--adapter", str(recovered)],
                    capture_output=True, timeout=15,
                )

            self.assertEqual(verify(bundle, "0"*64).returncode, 1)
            self.assertFalse(recovered.exists())
            evidence["phases"].pop()
            truncated = root / "incomplete.json"
            V.save_json(truncated, evidence)
            self.assertEqual(verify(truncated, pin).returncode, 1)
            self.assertFalse(recovered.exists())
            self.assertEqual(verify(bundle, pin).returncode, 0)
            self.assertEqual(json.loads(recovered.read_bytes()), teacher)
            self.assertEqual(verify(bundle, pin).returncode, 1)
            self.assertEqual(json.loads(recovered.read_bytes()), teacher)

    def test_custom_inputs_reject_before_cuda_dispatch(self):
        d, r, b = V.D, V.R, V.B
        data = prover.gen_inputs(d, r, b, 42)
        data["W"].pop()
        with self.assertRaisesRegex(ValueError, "custom input W"):
            prover.build_proof(d, r, b, V.SCALE, 42, "gpu", reps=1, warmup=0, inputs=data)
