#!/usr/bin/env python3
"""Host-side tests for tools/ornith_quality.py (no GPU, no model download).

Checks the parts of the quality suite that do not need an engine: the case list covers every
workload phase 23 names, the engine's greedy-output parser accepts the two shapes the CLI emits,
and - when a tokenizer has been extracted - each prompt encodes to a stable, non-empty token
sequence that decodes back to the same text.

    python tools/test_ornith_quality.py
    ORNITH_TOKENIZER_DIR=/work/tokenizer/ornith python tools/test_ornith_quality.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ornith_quality  # noqa: E402
import ornith_quality_cases  # noqa: E402


class SuiteTest(unittest.TestCase):
    def test_every_case_is_named_and_categorised(self):
        names = [c["name"] for c in ornith_quality_cases.CASES]
        self.assertEqual(len(names), len(set(names)), "case names must be unique")
        for case in ornith_quality_cases.CASES:
            self.assertTrue(case["name"] and case["category"] and case["prompt"].strip(),
                            "a case is missing a name, category or prompt")

    def test_covers_the_phase_23_workloads(self):
        """Phase 23 names six workloads; 18F adds the long repeated structure."""
        categories = {c["category"] for c in ornith_quality_cases.CASES}
        for required in ("code generation", "code repair", "repository reasoning", "structured JSON",
                         "prose reasoning", "instruction following", "long repeated structure"):
            self.assertIn(required, categories)


class ParserTest(unittest.TestCase):
    def test_comma_terminated_line(self):
        self.assertEqual(ornith_quality.parse_stream("16,220,16,220,\n"), [16, 220, 16, 220])

    def test_single_token(self):
        self.assertEqual(ornith_quality.parse_stream("7,\n"), [7])

    def test_empty_output(self):
        self.assertEqual(ornith_quality.parse_stream(""), [])

    def test_ignores_lines_after_the_first(self):
        """stdout also carries DONE/timing lines; only the first is the token stream."""
        self.assertEqual(ornith_quality.parse_stream("1,2,\nDONE 2 2 1.0 2.0 x 0 0 0\n"), [1, 2])


class TokenizerTest(unittest.TestCase):
    """Runs only when a tokenizer directory is supplied, so the suite stays a host-only test."""

    def setUp(self):
        self.directory = os.environ.get("ORNITH_TOKENIZER_DIR", "")
        if not self.directory:
            self.skipTest("ORNITH_TOKENIZER_DIR is not set")

    def test_prompts_encode_and_round_trip(self):
        tokenizer = ornith_quality.load_tokenizer(Path(self.directory))
        for case in ornith_quality_cases.CASES:
            ids = tokenizer.encode(case["prompt"])
            self.assertGreater(len(ids), 8, case["name"])
            # Encoding the decoded text again must be stable: the comparison is token equality.
            self.assertEqual(tokenizer.encode(tokenizer.decode(ids)), ids, case["name"])


class ParityParserTest(unittest.TestCase):
    """The GPU-parity log parser, exercised on the exact lines qwen35_gpu_parity prints."""

    LINE = (
        "layer 3 residual max_abs=1.2e-05 rms=3.4e-06 relative_rms=8.1e-07\n"
        "hidden max_abs=1e-05 rms=2e-06 relative_rms=9e-07\n"
        "logits max_abs=4.3e-04 rms=1.1e-04 relative_rms=2.2e-03\n"
        "distribution kl=1.7e-05 top10_overlap=10/10\n"
        "top1=42/42\n"
        "logits max_abs=9.9e-04 rms=2.2e-04 relative_rms=6.6e-03\n"
        "distribution kl=3.3e-05 top10_overlap=9/10\n"
        "top1=7/8\n"
        "GPU model parity: PASS\n")

    def test_parses_metrics_and_verdict(self):
        import ornith_model_parity
        parsed = ornith_model_parity.parse(self.LINE)
        self.assertTrue(parsed["passed"])
        self.assertEqual(len(parsed["tokens"]), 2)
        self.assertEqual(parsed["tokens"][0]["top10_overlap"], 10)
        self.assertTrue(parsed["tokens"][0]["top1_match"])
        self.assertFalse(parsed["tokens"][1]["top1_match"])
        self.assertAlmostEqual(parsed["tokens"][1]["logits_relative_rms"], 6.6e-03)
        self.assertEqual(parsed["worst_residual"]["layer"], 3)

    def test_reports_failure(self):
        import ornith_model_parity
        self.assertFalse(ornith_model_parity.parse("GPU model parity: FAIL\n")["passed"])


if __name__ == "__main__":
    unittest.main()
