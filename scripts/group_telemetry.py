"""Sample device-wide NVML readings and the trainer's committed group-step counter."""
import csv
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import threading
import time

from run_episode_comparison import read_json, write_json


class TrainingTelemetry:
    def __init__(self, experiment, interval=5):
        self.root = Path(experiment)
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="training-telemetry", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=10)

    def _sample(self):
        smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        history = deque(maxlen=37)
        path = self.root / "gpu-telemetry.csv"
        with path.open("a", newline="", buffering=1) as stream:
            csvout = csv.writer(stream)
            if not stream.tell():
                csvout.writerow(["sampled_utc","group_step","gpu_util_percent","used_mib","free_mib","power_w"])
            while not self.stop.is_set():
                try:
                    p = read_json(self.root/"group-progress.json")
                    step = int(p["optimizer_steps"])
                    if p.get("status") == "running" and step > 0:
                        query = subprocess.run([smi,"-i","0","--query-gpu=utilization.gpu,memory.used,memory.free,power.draw",
                                                "--format=csv,noheader,nounits"], text=True, capture_output=True,
                                               timeout=4, check=True)
                        vals = [x.strip() for x in query.stdout.strip().split(",")]
                        util, used, free = map(int, vals[:3])
                        power = float(vals[3]) if len(vals)>3 and vals[3] not in ("[N/A]","N/A") else None
                        stamp = datetime.now(timezone.utc).isoformat()
                        sampled = time.monotonic()
                        history.append((sampled,step,util,used,free))
                        csvout.writerow([stamp, step, util, used, free, power])
                        elapsed = sampled - history[0][0]
                        moved = step - history[0][1]
                        rate = moved/elapsed if elapsed >= 15 and moved>0 else None
                        remaining = max(0, p["expected_steps"]-step)
                        write_json(self.root/"gpu-telemetry.json", {
                            "sampled_utc":stamp, "sample_count":len(history), "samples_window_seconds":elapsed,
                            "group_step":step, "gpu_util_percent":util,
                            "gpu_util_percent_mean":sum(h[2] for h in history)/len(history),
                            "gpu_util_percent_min":min(h[2] for h in history), "gpu_util_percent_max":max(h[2] for h in history),
                            "memory_used_mib":used, "memory_used_mib_peak":max(h[3] for h in history),
                            "memory_free_mib_min":min(h[4] for h in history), "power_w":power,
                            "group_steps_per_second":rate, "eta_seconds":remaining/rate if rate else None})
                except FileNotFoundError:
                    pass  # Initial durable checkpoint and progress may still be in preparation.
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                    # Telemetry is independent of the optimizer and its recovery protocol.
                    if not getattr(self, "_reported_error", False):
                        print(f"GPU telemetry sample unavailable: {exc}", flush=True)
                    self._reported_error = True
                self.stop.wait(self.interval)
