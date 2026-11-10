"""Run the sealed 50% train / 30% test comparison sequentially and durably."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_paths import ROOT, REPORTS
REPO = ROOT / 'le-wm'
EXPERIMENT = ROOT / 'experiments/episode50-30-20261106-03'
HOST_OUTPUT = REPORTS / EXPERIMENT.name
MODELS = [('sigreg', 'lewm', 'tworoom-sigreg-episode50-20261106-03'),
          ('action_predictor', 'lewm_action', 'tworoom-action-episode50-20261106-03')]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_command(label, config, name, experiment=EXPERIMENT, epochs=1, dataset=None, batch_size=32):
    dataset = dataset or ROOT / 'storage-local/datasets/tworoom.h5'
    return [sys.executable, '-u', str(experiment / 'source/train_resumable.py'), '--config-name', config,
            'data=tworoom', f'data.dataset.name={dataset}', f'subdir={name}', f'output_model_name={name}',
            f'trainer.max_epochs={epochs}', f'loader.batch_size={batch_size}', 'num_workers=4',
            'loader.persistent_workers=true', 'loader.prefetch_factor=2', '+loader.multiprocessing_context=spawn', '+cpu_threads=1',
            'wandb.enabled=false', 'trainer.precision=bf16-mixed', 'trainer.devices=1', 'seed=3072',
            '+trainer.limit_val_batches=0.0', '+trainer.log_every_n_steps=10',
            '+trainer.enable_progress_bar=false', '+checkpoint_every_seconds=300', f'+control_file={experiment / "control.json"}',
            f'+trainer.default_root_dir={ROOT / "training" / name}',
            f'+episode_manifest={experiment / "manifest.json"}',
            f'+run_metadata_dir={experiment / "runs" / label}',
            f'hydra.run.dir={experiment / "runs" / label}']


def prepare():
    source = EXPERIMENT / 'source'
    if source.exists():
        raise FileExistsError(f'Frozen source already exists: {source}')
    source.mkdir()
    for name in ('train.py', 'train_resumable.py', 'resumable_training.py', 'jepa.py', 'module.py', 'utils.py', 'episode_data.py', 'experiment_tracking.py', 'training_compute.py'):
        shutil.copy2(REPO / name, source / name)
    shutil.copytree(REPO / 'config', source / 'config')
    (source / 'scripts').mkdir()
    for name in ('run_episode_comparison.py', 'prepare_episode_experiment.py', 'check_episode_training.py',
                 'evaluate_episode_experiment.py', 'audit_episode_experiment.py', 'check_resumable_training.py'):
        shutil.copy2(REPO / 'scripts' / name, source / 'scripts' / name)
    files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True).stdout
    (EXPERIMENT / 'environment.txt').write_text(packages)
    write_json(EXPERIMENT / 'source-snapshot.json', {'created_utc': now(), 'files_sha256': files,
               'python': sys.version, 'source': str(source)})
    manifest = read_json(EXPERIMENT / 'manifest.json')
    write_json(EXPERIMENT / 'plan.json', {
        'created_utc': now(), 'manifest_sha256': manifest['manifest_sha256'],
        'train_episodes': 5000, 'test_episodes': 3000, 'unused_episodes': 2000,
        'train_windows': 365657, 'test_windows': 218790, 'steps_per_model': 11427, 'training_batch_size': 32,
        'epochs': 1, 'fresh_initialization': True, 'same_training_order': True,
        'normalization': 'Training episodes only', 'loader_workers': 4, 'prefetch_batches_per_worker': 2, 'worker_start_method': 'spawn', 'cpu_threads': 1, 'validation_during_training': False,
        'checkpoint_interval_seconds': 300, 'pause_resume': True, 'training_backend': 'native_pytorch', 'precisions': ['bf16_mixed', 'fp32'], 'primary_metric': 'Next-embedding MSE',
        'training_commands': {label: train_command(label, cfg, name) for label, cfg, name in MODELS},
        'model_exports': {label: str(ROOT / 'storage/checkpoints' / name) for label, cfg, name in MODELS},
        'resume_policy': 'Restore atomic checkpoints with model, optimizer, scheduler, Python/NumPy/CPU/CUDA RNG and committed data cursor; verify split/config/order hashes before continuing.',
    })
    HOST_OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename in ('manifest.json', 'arrays.npz', 'plan.json', 'source-snapshot.json', 'environment.txt',
                     'preflight-gpu.json', 'dataset-recovery.json', 'restart-provenance.json', 'throughput-benchmark.json'):
        shutil.copy2(EXPERIMENT / filename, HOST_OUTPUT / filename)
    shutil.copytree(EXPERIMENT / 'diagnostics', HOST_OUTPUT / 'diagnostics', dirs_exist_ok=True)
    print(json.dumps({'prepared': True, 'experiment': str(EXPERIMENT), 'source_files': len(files)}), flush=True)


def verify_source(experiment=EXPERIMENT):
    snap = read_json(experiment / 'source-snapshot.json')
    for name, digest in snap['files_sha256'].items():
        if sha(experiment / 'source' / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')


class PauseRequested(Exception):
    pass


def pause_requested(experiment=EXPERIMENT):
    path = experiment / 'control.json'
    return path.exists() and read_json(path).get('request') == 'pause'


class Pipeline:
    experiment = EXPERIMENT
    host_output = HOST_OUTPUT

    def __init__(self):
        self.state = {'schema_version': 1, 'status': 'running', 'phase': 'starting',
                      'started_utc': now(), 'pid': os.getpid(), 'completed_phases': [],
                      'experiment': str(self.experiment), 'host_output': str(self.host_output)}
        self.publish()

    def publish(self):
        self.state['updated_utc'] = now()
        write_json(self.experiment / 'status.json', self.state)
        # Windows readers can briefly deny renaming the desktop status mirror.
        # WSL status above is authoritative. Retry the mirror at the next poll.
        try:
            write_json(self.host_output / 'status.json', self.state)
        except OSError as error:
            if not getattr(self, '_mirror_delayed', False):
                print(f'Windows status mirror delayed: {error}', file=sys.stderr, flush=True)
            self._mirror_delayed = True
        else:
            self._mirror_delayed = False

    def child(self, phase, command, log, env, progress=None):
        if pause_requested(self.experiment):
            raise PauseRequested('Paused before the next stage')
        log.parent.mkdir(parents=True, exist_ok=True)
        self.state.update(phase=phase, command=command, log=str(log), phase_started_utc=now())
        self.state.pop('child_progress', None)
        with log.open('a', buffering=1) as stream:
            stream.write(f'\nPipeline invocation {now()}: {json.dumps(command)}\n')
            process = subprocess.Popen(command, cwd=self.experiment / 'source', env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            self.state['child_pid'] = process.pid
            self.publish()
            while process.poll() is None:
                if progress is not None and progress.exists():
                    self.state['child_progress'] = read_json(progress)
                self.publish()
                if pause_requested(self.experiment) and not phase.startswith('train_'):
                    process.terminate()
                    process.wait(timeout=30)
                    raise PauseRequested('Paused evaluation; completed phases are preserved')
                time.sleep(2)
            if phase.startswith('train_'):
                (log.parent / 'training-exit.code').write_text(str(process.returncode) + '\n')
            if process.returncode == 75:
                raise PauseRequested('Training checkpoint saved; GPU released')
            if process.returncode:
                raise RuntimeError(f'{phase} failed with exit code {process.returncode}; see {log}')
        self.state['completed_phases'].append(phase)
        self.state.pop('child_pid', None)
        self.publish()

    def run(self):
        verify_source()
        env = os.environ.copy()
        env['LEWM_REPO'] = str(EXPERIMENT / 'source')
        env['PYTHONPATH'] = str(EXPERIMENT / 'source')
        env['PYTHONUNBUFFERED'] = '1'
        env['OMP_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        for label, config, name in MODELS:
            logdir = EXPERIMENT / 'runs' / label
            done = logdir / 'runcompletion.json'
            if not done.exists():
                command = train_command(label, config, name)
                checkpoint = logdir / 'checkpoints/last.ckpt'
                if checkpoint.exists():
                    command += [f'+resume_from={checkpoint}']
                elif (logdir / 'initialization.json').exists():
                    raise RuntimeError(f'{label} has initialization but no resumable checkpoint; inspect startup failure')
                runenv = env.copy()
                runenv['SPT_CACHE_DIR'] = str(ROOT / 'training' / name)
                self.child(f'train_{label}', command, logdir / 'console.log', runenv, logdir / 'progress.json')
            completion = read_json(done)
            if not completion['passed'] or completion['global_step'] != 11427:
                raise ValueError(f'{label} did not complete all 11,427 updates')
            command = [sys.executable, '-u', str(EXPERIMENT / 'source/scripts/audit_episode_experiment.py'),
                       '--run-dir', str(logdir),
                       '--manifest', str(EXPERIMENT / 'manifest.json'),
                       '--expected-objective', 'sigreg' if label == 'sigreg' else 'action_prediction']
            if label == 'action_predictor':
                command += ['--paired-run-dir', str(EXPERIMENT / 'runs/sigreg')]
            self.child(f'audit_{label}', command, logdir / 'audit-console.log', env)
            if not read_json(logdir / 'final-audit.json')['passed']:
                raise ValueError(f'{label} final audit did not pass')
        output = EXPERIMENT / 'evaluation'
        output.mkdir(exist_ok=True)
        command = [sys.executable, '-u', str(EXPERIMENT / 'source/scripts/evaluate_episode_experiment.py'),
                   '--experiment-dir', str(EXPERIMENT), '--output-dir', str(output),
                   '--sigreg-model', MODELS[0][2], '--action-model', MODELS[1][2],
                   '--batch-size', '8', '--num-workers', '4', '--cpu-threads', '1', '--precisions', 'bf16_mixed', 'fp32']
        self.child('evaluate_next_embedding', command, output / 'console.log', env, output / 'progress.json')
        results = read_json(output / 'results.json')
        if not results['passed'] or results['test_episodes'] != 3000 or results['test_windows'] != 218790:
            raise ValueError('Evaluation did not cover the full sealed test set')
        lines = ['Two-Room: fresh 50% training / 30% test comparison',
                 'Training: 5,000 episodes; 365,657 windows; batch32; 11,427 optimizer steps per model.',
                 'Test: 3,000 disjoint episodes; 218,790 windows; 656,370 predicted transitions per model/precision.',
                 'Unused: 2,000 episodes. Training-only normalization. Identical shared initial weights and training order.',
                 'Metric: next-embedding mean squared error; auxiliary losses excluded.', '']
        for precision in results['requested_precisions']:
            c = results['comparisons'][precision]['original_validation_arithmetic']
            lines += [precision, f"  Original SIGReg: {c['sigreg_mse']:.12g}",
                      f"  Action predictor: {c['action_predictor_mse']:.12g}",
                      f"  Action predictor relative reduction: {c['action_relative_reduction_percent']:.6f}%", '']
        (EXPERIMENT / 'report.txt').write_text('\n'.join(lines) + '\n')
        for label, config, name in MODELS:
            shutil.copytree(EXPERIMENT / 'runs' / label, HOST_OUTPUT / 'runs' / label, dirs_exist_ok=True, ignore=shutil.ignore_patterns('checkpoints'))
            saved_config = ROOT / 'storage/checkpoints' / name / 'config.yaml'
            shutil.copy2(saved_config, HOST_OUTPUT / 'runs' / label / 'model-config.yaml')
        shutil.copytree(output, HOST_OUTPUT / 'evaluation', dirs_exist_ok=True)
        shutil.copy2(EXPERIMENT / 'report.txt', HOST_OUTPUT / 'report.txt')
        self.state.update(status='complete', phase='complete', finished_utc=now(),
                          results=str(output / 'results.json'), report=str(HOST_OUTPUT / 'report.txt'))
        self.state.pop('child_progress', None)
        self.publish()


def run_locked(pipeline_type):
    with (pipeline_type.experiment / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (pipeline_type.experiment / 'status.json').exists() and read_json(pipeline_type.experiment / 'status.json')['status'] == 'complete':
            print('Experiment already complete; no changes made.', flush=True)
            return
        pipeline = pipeline_type()
        try:
            pipeline.run()
        except PauseRequested as error:
            pipeline.state.update(status='paused', paused_utc=now(), message=str(error))
            pipeline.state.pop('child_pid', None)
            pipeline.publish()
        except BaseException as error:
            pipeline.state.update(status='failed', failed_utc=now(), error_type=type(error).__name__, error=str(error))
            pipeline.publish()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        return
    run_locked(Pipeline)


if __name__ == '__main__':
    main()
