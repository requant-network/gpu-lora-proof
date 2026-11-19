#!/usr/bin/env python3
"""Stdlib self-tests for the gpu-lora-proof verifier (no GPU required).

Generates honest proofs with the CPU reference backend (tiny d=16 geometry,
two seeds, batch 1 and 3), then checks: honest proofs ACCEPT via the real
CLI (YES, exit 0); every negative control REJECTS (exit 1 / Reject);
malformed inputs reject; contract bounds hold.  Run:
    python3 -m unittest discover -s tests -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract as C
import prover
import tamper
import verifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_proof(seed, b, d=16, r=8):
    proof, expected = prover.build_proof(d, r, b, C.S_DEFAULT, seed, "cpu",
                                         None, reps=1, warmup=0)
    return proof, expected


class TestVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.p1, cls.e1 = make_proof(42, 1)
        cls.p2, cls.e2 = make_proof(42, 3)
        cls.p3, cls.e3 = make_proof(777, 1)
        cls.paths = {}
        for tag, (p, e) in (("p1", (cls.p1, cls.e1)), ("p2", (cls.p2, cls.e2)),
                            ("p3", (cls.p3, cls.e3))):
            pp = os.path.join(cls.tmp.name, f"{tag}_proof.json")
            ep = os.path.join(cls.tmp.name, f"{tag}_expected.json")
            with open(pp, "w") as f:
                json.dump(p, f)
            with open(ep, "w") as f:
                json.dump(e, f)
            cls.paths[tag] = (pp, ep)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cli(self, pp, ep):
        return subprocess.run([sys.executable, os.path.join(ROOT, "verifier.py"),
                               pp, ep], capture_output=True, text=True)

    def test_honest_accepts_cli(self):
        for tag in ("p1", "p2", "p3"):
            pp, ep = self.paths[tag]
            r = self.cli(pp, ep)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(r.stdout.strip(), "YES")

    def test_honest_check_counts(self):
        ok, _, stats = verifier.verify(self.p1, self.e1)
        self.assertTrue(ok)
        self.assertEqual(stats["checks"]["freivalds"], 18)
        self.assertEqual(stats["checks"]["response_bindings"], 54)
        self.assertEqual(stats["checks"]["leaf_hashes"], 18)
        self.assertEqual(stats["checks"]["input_pins"], 5)
        self.assertEqual(stats["checks"]["combines"], 4)

    def test_all_single_controls_reject(self):
        for name, fn in tamper.SINGLE_CONTROLS:
            with self.subTest(control=name):
                tp, _why = fn(self.p1)
                with self.assertRaises(verifier.Reject):
                    verifier.verify(tp, self.e1)

    def test_controls_reject_cli_exit1(self):
        tp, _ = tamper.wrong_dW_coherent(self.p1)
        pp = os.path.join(self.tmp.name, "tampered.json")
        with open(pp, "w") as f:
            json.dump(tp, f)
        r = self.cli(pp, self.paths["p1"][1])
        self.assertEqual(r.returncode, 1)
        self.assertTrue(r.stdout.startswith("NO:"))

    def test_stale_replay_rejects(self):
        tp, _ = tamper.stale_replay(self.p3)
        with self.assertRaises(verifier.Reject):
            verifier.verify(tp, self.e1)

    def test_coherent_tamper_keeps_hashes_valid(self):
        # Sanity: the coherent attacker really re-seals commitments, so the
        # rejection must come from arithmetic/pins, not stale hashes.
        tp, _ = tamper.wrong_dW_coherent(self.p1)
        sh = C.shapes(16, 8, 1)
        import array
        import hashlib
        for n in C.LEAF_ORDER:
            data = C.b64_decode_i64(tp["leaves"][n]["data_b64"])
            h = hashlib.sha256(C.leaf_bytes(n, sh[n],
                                            array.array("q", data).tobytes()))
            self.assertEqual(tp["leaf_hashes"][n], h.hexdigest())

    def test_garbage_rejects(self):
        pp = os.path.join(self.tmp.name, "garbage.json")
        with open(pp, "w") as f:
            f.write("{not json")
        r = self.cli(pp, self.paths["p1"][1])
        self.assertEqual(r.returncode, 1)
        self.assertTrue(r.stdout.startswith("NO:"))

    def test_missing_leaf_rejects(self):
        tp = json.loads(json.dumps(self.p1))
        del tp["leaves"]["dA"]
        with self.assertRaises(verifier.Reject):
            verifier.verify(tp, self.e1)

    def test_contract_bounds(self):
        for d in (256, 512, 1024):
            for r in (8, 16):
                for b in (1, 16):
                    C.assert_contract(d, r, b, C.S_DEFAULT)
        bd = C.bounds(1024, 16, 16, 8)
        self.assertLess(max(bd.values()), 1 << 40)
        self.assertLess(max(C.imma_int32_bounds(1024, 16).values()), 1 << 24)

    def test_soundness_bound_numeric(self):
        p = C.P
        self.assertLess(9 / (p * p), 2.0 ** -118)
        self.assertLess(2.0 ** -118, 2.0 ** -63)


if __name__ == "__main__":
    unittest.main()