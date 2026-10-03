"""Run with python -m unittest discover -s docker -p 'test_*.py'."""
from pathlib import Path
import subprocess
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeContract(unittest.TestCase):
    def reject(self, *args):
        result = subprocess.run(['bash', str(ROOT / 'run.sh'), '--dry-run', *args],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_context_is_fixed(self):
        self.assertIn('131072', self.reject('--max-context', '65536'))

    def test_budget_cannot_be_raised(self):
        self.assertIn('hard ceiling', self.reject('--budget', '10241'))

    def test_budget_must_leave_reserves(self):
        self.assertIn('hard ceiling', self.reject('--budget', '1280'))

    def test_launcher_syntax(self):
        subprocess.run(['bash', '-n', str(ROOT / 'run.sh'), str(ROOT / 'docker/entrypoint-hip.sh')], check=True)

    def test_tuned_defaults_and_cli_overrides(self):
        # Stub discovery only; exercise the actual launcher argument construction.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / 'docker'
            docker.write_text('#!/bin/sh\nexit 0\n')
            docker.chmod(0o755)
            python = root / 'python'
            python.write_text(f"#!{sys.executable}\n" + """import os, sys
if sys.argv[1].endswith('hipinfo.py'):
    if '--arch' in sys.argv: print('gfx1101')
    elif '--render-node' in sys.argv: print('/dev/dri/renderD128')
    elif '--vram' in sys.argv: print('12272 1500 10772')
    elif '--reserve-mib' in sys.argv: print('768')
elif sys.argv[1].endswith('hfmodel.py'):
    print('STRATA_CACHED=1; STRATA_ARENA_GB=43')
else:
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
""")
            python.chmod(0o755)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
            env.update(PATH=str(root) + os.pathsep + env['PATH'], PYTHON=str(python))
            args = ['bash', str(ROOT / 'run.sh'), '--dry-run', '--image', 'test:guard',
                    '--hf-cache', directory, '--work', directory]
            default = subprocess.run(args, env=env, capture_output=True, text=True, check=True).stdout
            for setting in ('STRATA_PREFILL=2048', 'STRATA_EXPERT_CACHE=800',
                            'STRATA_POOL_WORKERS=0', 'STRATA_MAX_CONTEXT=131072',
                            'STRATA_VRAM_LATER_MIB=768', 'STRATA_VRAM_RUNTIME_RESERVE_MIB=1024'):
                self.assertIn(setting, default)
            override = subprocess.run(args + ['--prefill', '1024', '--expert-cache', '512',
                                             '--pool-workers', '23'], env=env,
                                      capture_output=True, text=True, check=True).stdout
            for key, initial, explicit in [('PREFILL', '2048', '1024'),
                                           ('EXPERT_CACHE', '800', '512'),
                                           ('POOL_WORKERS', '0', '23')]:
                self.assertGreater(override.index(f'STRATA_{key}={explicit}'),
                                   override.index(f'STRATA_{key}={initial}'))


class OrnithRuntimeContract(unittest.TestCase):
    """What `./run3.sh` owes docs/ORNITH_QWEN35MOE.md: which tier runs by default, which KV layouts exist,
    what may be raised, and that a check never downloads.  `run.sh` above is Qwen3.8's own contract; the two
    launchers must not drift into each other's model, container or work directory."""

    ENTRYPOINT = ROOT / 'docker' / 'entrypoint-ornith.sh'

    def setUp(self):
        # Stub the two host probes run3.sh asks (the KFD reader and the artifact resolver) and docker itself,
        # so the contract is testable with no GPU, no image and no download.
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        docker = root / 'docker'
        docker.write_text('#!/bin/sh\nexit 0\n')
        docker.chmod(0o755)
        python = root / 'python'
        python.write_text(f"#!{sys.executable}\n" + """import os, sys
if sys.argv[1].endswith('hipinfo.py'):
    if '--arch' in sys.argv: print('gfx1101')
    elif '--render-node' in sys.argv: print('/dev/dri/renderD128')
    elif '--vram' in sys.argv: print('12272 1500 10772')
    elif '--reserve-mib' in sys.argv: print('768')
elif sys.argv[1].endswith('hfmodel.py'):
    print("STRATA_CACHED=1; STRATA_REPO=AtomicChat/Ornith-1.5-35B-A3B-GGUF; "
          "STRATA_MTP_REPO=EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF; "
          "STRATA_MTP_FILE=mtpdraft-Q8_0.gguf; STRATA_DOWNLOAD_GB=20.1; STRATA_MTP_DOWNLOAD_GB=2.0")
else:
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
""")
        python.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
        self.env.update(PATH=str(root) + os.pathsep + os.environ['PATH'], PYTHON=str(python))
        self.work = root

    def tearDown(self):
        self.directory.cleanup()

    def launch(self, *args):
        result = subprocess.run(['bash', str(ROOT / 'run3.sh'), '--dry-run', '--image', 'test:guard',
                                 '--hf-cache', str(self.work), '--work', str(self.work), *args],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        return result.stdout

    def reject(self, *args):
        result = subprocess.run(['bash', str(ROOT / 'run3.sh'), '--dry-run', '--image', 'test:guard',
                                 '--hf-cache', str(self.work), '--work', str(self.work), *args],
                                env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, 'expected a refusal for: ' + ' '.join(args))
        return result.stderr

    def test_default_is_the_split_tier_with_the_measured_settings(self):
        # f16 KV and auto expert cache are the measured fastest defaults (docs/ORNITH_QWEN35MOE.md); spec is
        # off until MTP is faster than plain decode.
        out = self.launch()
        for setting in ('STRATA_TIER=gpu', 'STRATA_KV=f16', 'STRATA_EXPERT_CACHE=auto',
                        'STRATA_MAX_CONTEXT=131072', 'STRATA_POOL_WORKERS=8', 'STRATA_SPEC=0',
                        'STRATA_VRAM_BUDGET_MIB=10240'):
            self.assertIn(setting, out)
        self.assertIn('entrypoint-ornith.sh', out)
        self.assertIn('strata-ornith-gfx1101', out)
        self.assertIn('127.0.0.1:9931:9931', out)
        self.assertIn(':/work', out)   # its own work tree, not Qwen3.8's

    def test_tier_cache_and_workers_are_forwarded(self):
        out = self.launch('--cpu', '--kv', 'f32', '--expert-cache', '2048', '--pool-workers', '10')
        for setting in ('STRATA_TIER=cpu', 'STRATA_KV=f32', 'STRATA_EXPERT_CACHE=2048',
                        'STRATA_POOL_WORKERS=10'):
            self.assertIn(setting, out)

    def test_only_the_implemented_kv_layouts_are_accepted(self):
        # int8/int4/int2 are Qwen3.8's layouts; the Qwen35MoE backend has f32 and f16 and parity for those.
        for bad in ('int8', 'q8', 'int4'):
            self.assertIn('f32 or f16', self.reject('--kv', bad))
        self.reject('--kv', '')

    def test_budget_and_context_bounds(self):
        self.assertIn('hard ceiling', self.reject('--budget', '10241'))
        self.assertIn('hard ceiling', self.reject('--budget', '1280'))
        self.assertIn('262144', self.reject('--max-context', '262145'))
        self.assertIn('--spec must be 0..4', self.reject('--spec', '5'))

    def test_publishing_beyond_localhost_needs_a_key(self):
        self.assertIn('--api-key', self.reject('--bind', '0.0.0.0'))
        self.assertIn('STRATA_API_KEY=secret', self.launch('--bind', '0.0.0.0', '--api-key', 'secret'))

    def test_a_check_never_downloads(self):
        out = self.launch('--check-only')
        self.assertNotIn('STRATA_AUTO_PREPARE', out)
        self.assertIn('STRATA_CHECK_ONLY=1', out)
        text = self.ENTRYPOINT.read_text()
        self.assertIn('STRATA_CHECK_ONLY:-0}" = "1" ] && FETCH+=(--offline)', text)

    def test_the_entrypoint_hands_the_tier_and_tunables_to_the_engine(self):
        # The launcher's whole purpose is that the engine is started on the chosen tier; the tier, KV layout
        # and cache have to reach the command line the server spawns.
        text = self.ENTRYPOINT.read_text()
        self.assertIn('"--kv", e["KV"], "--expert-cache", e["EXPERT_CACHE"]', text)
        self.assertIn('"--prefill", e["PREFILL"], "--" + e["TIER"]', text)
        self.assertIn('STRATA_PREFILL:-8', text)
        self.assertIn('--fit-report', text)
        self.assertIn('STRATA_KV:-f16', text)
        self.assertIn('STRATA_TIER:-gpu', text)
        self.assertNotIn('STRATA_KV:-int8', text)

    def test_launcher_syntax(self):
        subprocess.run(['bash', '-n', str(ROOT / 'run3.sh'), str(self.ENTRYPOINT)], check=True)

    def test_the_other_launchers_do_not_touch_the_ornith_path(self):
        for launcher in ('run.sh', 'run2.sh'):
            text = (ROOT / launcher).read_text()
            self.assertNotIn('entrypoint-ornith', text)
            self.assertNotIn('ornith', text.lower())


class ProcessAccounting(unittest.TestCase):
    def test_clients_are_deduplicated_and_device_filtered(self):
        import importlib.util
        import sys
        import tempfile
        sys.path.insert(0, str(ROOT / 'docker'))
        spec = importlib.util.spec_from_file_location('vram_guard', ROOT / 'docker/vram-guard.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / '123' / 'fdinfo'
            directory.mkdir(parents=True)
            fields = 'drm-driver: amdgpu\ndrm-client-id: 7\ndrm-pdev: 0000:03:00.0\ndrm-total-vram: 4096 KiB\n'
            (directory / '1').write_text(fields)
            (directory / '2').write_text(fields)
            (directory / '3').write_text(fields.replace('client-id: 7', 'client-id: 8').replace('4096', '1024'))
            (directory / '4').write_text(fields.replace('03:00', '09:00'))
            self.assertEqual(guard.process_vram(123, '0000:03:00.0', Path(d)), 5120 * 1024)
            self.assertIsNone(guard.process_vram(124, '0000:03:00.0', Path(d)))
            self.assertIsNone(guard.process_vram(123, '0000:05:00.0', Path(d)))


if __name__ == '__main__':
    unittest.main()
