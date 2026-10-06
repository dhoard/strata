"""Run with python -m unittest discover -s docker -p 'test_*.py'."""
from pathlib import Path
import json
import pty
import subprocess
import os
import sys
import tempfile
import unittest

from _stub_runtime import stub_python

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
            stub_python(python)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
            env.update(PATH=str(root) + os.pathsep + env['PATH'], PYTHON=str(python))
            args = ['bash', str(ROOT / 'run.sh'), '--dry-run', '--image', 'test:guard',
                    '--hf-cache', directory, '--work', directory]
            default = subprocess.run(args, env=env, capture_output=True, text=True, check=True).stdout
            for setting in ('STRATA_MODEL=IQ3_S', 'starting qwen IQ3_S',
                            'STRATA_MODEL_NAME=qwen3.8-flash-next-iq3_s',
                            'STRATA_PACK_DIR=/work/packs/iq3_s',
                            'STRATA_HF_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                            'STRATA_PREFILL=2048', 'STRATA_EXPERT_CACHE=auto',
                            'STRATA_POOL_WORKERS=0', 'STRATA_MAX_CONTEXT=131072',
                            'STRATA_VRAM_LATER_MIB=700', 'STRATA_VRAM_RUNTIME_RESERVE_MIB=1024',
                            'STRATA_PREFILL_RING=48'):
                self.assertIn(setting, default)
            self.assertNotIn('swift', default)
            swift = subprocess.run(args + ['--release', 'swift'], env=env,
                                   capture_output=True, text=True, check=True).stdout
            for setting in ('STRATA_MODEL=IQ3_XXS', 'STRATA_MODEL_NAME=swift-1.5-iq3_xxs',
                            'STRATA_PACK_DIR=/work/packs/swift-iq3_xxs',
                            'STRATA_HF_REPO=ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF'):
                self.assertIn(setting, swift)
            qwen = subprocess.run(args + ['--model', 'IQ3_XXS'], env=env,
                                  capture_output=True, text=True, check=True).stdout
            self.assertIn('starting qwen IQ3_XXS', qwen)
            self.assertIn('STRATA_EXPERT_CACHE=800', qwen)
            self.assertIn('STRATA_VRAM_LATER_MIB=768', qwen)
            self.assertNotIn('STRATA_PREFILL_RING', qwen)  # pins belong to their release+quant
            self.assertIn('STRATA_MODEL_NAME=qwen3.8-flash-next-iq3_xxs', qwen)
            self.assertIn('STRATA_PACK_DIR=/work/packs/iq3_xxs', qwen)
            self.assertIn('STRATA_HF_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', qwen)
            override = subprocess.run(args + ['--prefill', '1024', '--expert-cache', '512',
                                             '--pool-workers', '23', '--engine-silence', '900'], env=env,
                                      capture_output=True, text=True, check=True).stdout
            for key, initial, explicit in [('PREFILL', '2048', '1024'),
                                           ('EXPERT_CACHE', 'auto', '512'),
                                           ('POOL_WORKERS', '0', '23'),
                                           ('ENGINE_SILENCE_S', '', '900')]:
                self.assertGreater(override.index(f'STRATA_{key}={explicit}'),
                                   override.index(f'STRATA_{key}={initial}') if initial else 0)

    def test_launch_overrides_image_release_and_handles_stdin(self):
        # Exercise a real launch with Docker stubbed at its boundary. Simulate the
        # installed image's Swift default and Docker's rejection of piped -it.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / 'docker'
            docker.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
a = sys.argv[1:]
if a[:2] == ['image', 'inspect']:
    print('1')  # image has the HIP allocation guard
elif a and a[0] == 'run':
    if '-it' in a and not os.isatty(0):
        sys.exit('cannot attach stdin to a TTY-enabled container')
    env = {'STRATA_HF_REPO': 'ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF'}
    for i, arg in enumerate(a):
        if arg == '-e':
            key, _, value = a[i + 1].partition('=')
            env[key] = value
    if env['STRATA_HF_REPO'] != 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF':
        sys.exit('container selected Swift instead of the cached Qwen model')
    print(json.dumps(a))
''')
            docker.chmod(0o755)
            python = root / 'python'
            stub_python(python)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('STRATA_') and k != 'HSA_OVERRIDE_GFX_VERSION'}
            env.update(PATH=str(root) + os.pathsep + env['PATH'], PYTHON=str(python))
            args = ['bash', str(ROOT / 'run.sh'), '--image', 'test:swift-default',
                    '--hf-cache', directory, '--work', directory, '--port', '0']
            master, terminal = pty.openpty()
            try:
                for stdin, options, interactive, detached in (
                        (subprocess.DEVNULL, [], False, False),
                        (terminal, [], True, False),
                        (terminal, ['--detach'], False, True)):
                    with self.subTest(interactive=interactive, detached=detached):
                        result = subprocess.run(args + options, env=env, stdin=stdin,
                                                capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        line = next(l for l in result.stdout.splitlines() if l.startswith('["run"'))
                        docker_args = json.loads(line)
                        self.assertEqual('-it' in docker_args, interactive)
                        self.assertEqual('-d' in docker_args, detached)
            finally:
                os.close(master)
                os.close(terminal)


class EngineConfig(unittest.TestCase):
    """The engine config the entrypoint writes for the server (docker/entrypoint-hip.sh), without Docker or a GPU:
    the block is plain Python over the environment, so it can be run on its own."""

    def write(self, **env_extra):
        text = (ROOT / 'docker' / 'entrypoint-hip.sh').read_text()
        script = text.split('"$PY" - <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
        env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
        env.update(PACK='/work/packs/iq3_s', NATIVE='/hf-cache/qwen.gguf', PLE='/hf-cache/qwen2.gguf',
                   EXPERT_PROFILE='/opt/strata/data/expert-profile.bin', EXPERT_CACHE='auto',
                   RESERVE_MIB='700', PREFILL='2048', SPEC='4', MTP='/work/mtp/rt', MAX_CONTEXT='131072',
                   KV='int8', POOL_WORKERS='0', REPO='/opt/strata', MODEL_NAME='qwen3.8-flash-next-iq3_s',
                   LOG='/work/logs/strata-hip.log', MAX_TOKENS='32768', REASONING='high')
        env.setdefault('SILENCE', '')
        env.update(env_extra)
        with tempfile.TemporaryDirectory() as directory:
            env['CONFIG'] = os.path.join(directory, 'strata-hip.json')
            result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(Path(env['CONFIG']).read_text()), result.stdout

    def test_silence_reaches_the_config_only_when_asked(self):
        cfg, _ = self.write()
        self.assertNotIn('engine_silence_s', cfg)          # unset: the server's own default (300 s) stands
        cfg, out = self.write(SILENCE='0')
        self.assertEqual(cfg['engine_silence_s'], 0.0)     # #481: 0 = the server waits for the engine forever
        self.assertIn('engine_silence_s=0', out)
        self.assertEqual(self.write(SILENCE='1800')[0]['engine_silence_s'], 1800.0)

    def test_the_rest_of_the_config_is_untouched(self):
        cfg, _ = self.write(SILENCE='900')
        self.assertEqual(cfg['args'][:10], ['--pack', '/work/packs/iq3_s', '--native', '/hf-cache/qwen.gguf',
                                            '--ple-gguf', '/hf-cache/qwen2.gguf', '--mmap-experts',
                                            '--expert-profile', '/opt/strata/data/expert-profile.bin',
                                            '--expert-cache'])
        self.assertEqual(cfg['max_tokens'], 32768)
        self.assertEqual(cfg['reasoning_effort'], 'high')
        self.assertEqual(cfg['cwd'], '/opt/strata')


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
