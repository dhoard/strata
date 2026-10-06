"""Contracts for the Qwen and Swift container launchers.
Run: python -m unittest discover -s docker -p 'test_*.py'.

Pin model identity, tuning, 128K context and the 10 GiB ceiling without relying
on the deleted bench/results fixtures. GPU discovery and Docker are stubbed.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import unittest

from _stub_runtime import stub_python

ROOT = Path(__file__).resolve().parents[1]


def expected_env_pairs(release, model):
    """Explicit launch contract, including release identity over image defaults."""
    env = {
        'HIP_VISIBLE_DEVICES': '0', 'HSA_ENABLE_SDMA': '1',
        'STRATA_DOWNLOAD_MODEL': '1', 'STRATA_VRAM_BUDGET_MIB': '10240',
        'STRATA_VRAM_RUNTIME_RESERVE_MIB': '1024', 'STRATA_VRAM_SLACK_MIB': '256',
        'STRATA_MAX_CONTEXT': '131072', 'STRATA_POOL_WORKERS': '0',
        'STRATA_PREFILL': '2048', 'STRATA_PORT': '9931', 'STRATA_MODEL': model,
    }
    if release == 'swift':
        env.update(STRATA_HF_REPO='ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                   STRATA_PACK_DIR='/work/packs/swift-iq3_xxs',
                   STRATA_MODEL_NAME='swift-1.5-iq3_xxs')
    else:
        env.update(STRATA_HF_REPO='ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                   STRATA_PACK_DIR=f'/work/packs/{model.lower()}',
                   STRATA_MODEL_NAME=f'qwen3.8-flash-next-{model.lower()}')
    if (release, model) in (('qwen', 'IQ3_S'), ('swift', 'IQ3_XXS')):
        env.update(STRATA_EXPERT_CACHE='auto', STRATA_VRAM_LATER_MIB='700',
                   STRATA_PREFILL_RING='48')
    else:
        env.update(STRATA_EXPERT_CACHE='800', STRATA_VRAM_LATER_MIB='768')
    return sorted(f'{key}={value}' for key, value in env.items())


def launch(*args, env_extra=None, launcher='run.sh'):
    """Run ./run.sh --dry-run with docker/python (hipinfo/hfmodel) stubbed; returns (rc, out, err)."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        docker = root / 'docker'
        docker.write_text('#!/bin/sh\nexit 0\n')
        docker.chmod(0o755)
        python = root / 'python'
        stub_python(python)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
        env.update(PATH=str(root) + os.pathsep + env['PATH'], PYTHON=str(python), **(env_extra or {}))
        result = subprocess.run(
            ['bash', str(ROOT / launcher), '--dry-run', '--image', 'test:guard',
             '--hf-cache', directory, '--work', directory, *args],
            env=env, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr


def actual_env_pairs(stdout):
    line = next((l for l in stdout.splitlines() if 'docker run' in l), '')
    toks = line.split()
    return sorted(toks[i + 1] for i, t in enumerate(toks) if t == '-e' and i + 1 < len(toks))


class LauncherContract(unittest.TestCase):
    def test_both_launchers_exist(self):
        self.assertEqual(sorted(p.name for p in ROOT.glob('run*.sh')), ['run-swift.sh', 'run.sh'])

    def test_launcher_syntax(self):
        for launcher in ('run.sh', 'run-swift.sh'):
            subprocess.run(['bash', '-n', str(ROOT / launcher)], check=True)

    def test_swift_launcher_default(self):
        rc, out, err = launch(launcher='run-swift.sh')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out),
                         expected_env_pairs('swift', 'IQ3_XXS'))
        self.assertIn('STRATA_MODEL=IQ3_XXS', out)
        self.assertIn('STRATA_MODEL_NAME=swift-1.5-iq3_xxs', out)
        self.assertIn('STRATA_PACK_DIR=/work/packs/swift-iq3_xxs', out)
        self.assertIn('STRATA_EXPERT_CACHE=auto', out)
        self.assertIn('STRATA_VRAM_LATER_MIB=700', out)
        self.assertIn('STRATA_PREFILL_RING=48', out)
        self.assertIn('STRATA_MAX_CONTEXT=131072', out)

    def test_qwen_iq3_s_pin_survives_under_release_qwen(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_S')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), expected_env_pairs('qwen', 'IQ3_S'))

    def test_qwen_iq3xxs_pin_survives_under_release_qwen(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_XXS')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), expected_env_pairs('qwen', 'IQ3_XXS'))

    def test_swift_launcher_forwards_options_and_release_overrides(self):
        for args, env_extra, release, model in (
                (('--model', 'IQ3_XXS', '--release', 'qwen'), None, 'qwen', 'IQ3_XXS'),
                ((), {'STRATA_RELEASE': 'qwen'}, 'qwen', 'IQ3_S'),
                (('--model', 'IQ3_S'), {
                    'STRATA_HF_REPO': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF'},
                 'qwen', 'IQ3_S')):
            with self.subTest(args=args, env_extra=env_extra):
                rc, out, err = launch(*args, env_extra=env_extra, launcher='run-swift.sh')
                self.assertEqual(rc, 0, err)
                self.assertEqual(actual_env_pairs(out), expected_env_pairs(release, model))

    def test_qwen_launcher_default(self):
        rc, out, err = launch()
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out), expected_env_pairs('qwen', 'IQ3_S'))

    def test_expert_cache_numeric_overrides_auto_with_warning(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_S', '--expert-cache', '900')
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_EXPERT_CACHE=900', out)
        self.assertIn('overrides the tuned auto sizing', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_expert_cache_above_iq3xxs_pin_still_warns(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ3_XXS', '--expert-cache', '900')
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_EXPERT_CACHE=900', out)
        self.assertIn('tuned 800', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_ornith_refused_before_docker(self):
        rc, out, err = launch('--model', 'ornith')
        self.assertNotEqual(rc, 0)
        self.assertNotIn('docker run', out)
        self.assertIn('IQ3_XXS', re.sub(r'\x1b\[[0-9;]*m', '', err))


class ReleaseAxis(unittest.TestCase):
    """Release flags and environment overrides work through both launchers."""

    def test_release_swift_line_matches_the_gated_launch(self):
        rc, out, err = launch('--release', 'swift', '--model', 'IQ3_XXS')
        self.assertEqual(rc, 0, err)
        self.assertEqual(actual_env_pairs(out),
                         expected_env_pairs('swift', 'IQ3_XXS'))

    def test_release_flag_line_equals_the_manual_env_line(self):
        """What users typed yesterday (explicit -e) and what --release does today must give the
        same EFFECTIVE docker environment - docker takes the last -e of a key, so compare the
        last-wins dicts, duplicates included on the manual side."""
        rc1, old, err1 = launch('--model', 'IQ3_XXS',
                               '-e', 'STRATA_HF_REPO=ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                               '-e', 'STRATA_PACK_DIR=/work/packs/swift-iq3_xxs',
                               '-e', 'STRATA_MODEL_NAME=swift-1.5-iq3_xxs',
                               '-e', 'STRATA_EXPERT_CACHE=auto',
                               '-e', 'STRATA_VRAM_LATER_MIB=700',
                               '-e', 'STRATA_PREFILL_RING=48')
        rc2, new, err2 = launch('--release', 'swift', '--model', 'IQ3_XXS')
        self.assertEqual((rc1, rc2), (0, 0), err1 + err2)

        def effective(stdout):
            line = next((l for l in stdout.splitlines() if 'docker run' in l), '')
            toks = line.split()
            env = {}
            for i, t in enumerate(toks):
                if t == '-e' and i + 1 < len(toks) and '=' in toks[i + 1]:
                    k, _, v = toks[i + 1].partition('=')
                    env[k] = v
            return env
        self.assertEqual(effective(old), effective(new))
        self.assertEqual(sorted(f'{k}={v}' for k, v in effective(new).items()),
                         expected_env_pairs('swift', 'IQ3_XXS'))

    def test_release_qwen_line_keeps_the_qwen_pin(self):
        rc, qwen, _ = launch('--release', 'qwen', '--model', 'IQ3_S')
        rc2, default, _ = launch()
        self.assertEqual((rc, rc2), (0, 0))
        self.assertEqual(actual_env_pairs(qwen), actual_env_pairs(default))
        self.assertEqual(actual_env_pairs(qwen), expected_env_pairs('qwen', 'IQ3_S'))

    def test_coder_quant_still_resolves_coder(self):
        rc, out, err = launch('--model', 'IQ1_M')      # no release named: the quant names coder
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_MODEL_NAME=qwen3.8-flash-next-coder-iq1_m', out)
        self.assertIn('STRATA_PACK_DIR=/work/packs/coder-iq1_m', out)

    def test_coder_quant_under_qwen_is_refused(self):
        rc, out, err = launch('--release', 'qwen', '--model', 'IQ1_M')
        self.assertNotEqual(rc, 0)
        self.assertNotIn('docker run', out)
        self.assertIn('coder', re.sub(r'\x1b\[[0-9;]*m', '', err))

    def test_unmeasured_combination_warns_and_uses_conservative_pins(self):
        rc, out, err = launch('--release', 'swift', '--model', 'IQ2_XS')
        self.assertEqual(rc, 0, err)
        self.assertIn('not measured on this card', re.sub(r'\x1b\[[0-9;]*m', '', err))
        self.assertIn('STRATA_EXPERT_CACHE=680', out)
        self.assertNotIn('STRATA_PREFILL_RING', out)

    def test_repo_env_still_routes_the_release(self):
        rc, out, err = launch('--model', 'IQ3_XXS', env_extra={
            'STRATA_HF_REPO': 'ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF'})
        self.assertEqual(rc, 0, err)
        self.assertIn('STRATA_MODEL_NAME=swift-1.5-iq3_xxs', out)   # host env alone routes swift
        self.assertIn('STRATA_HF_REPO=ukisai/', out)                 # and is forwarded (it was not
                                                                    # before this change)


if __name__ == '__main__':
    unittest.main()
