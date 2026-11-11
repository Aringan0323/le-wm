"""Local LeWM training dashboard. Run with Python; open http://127.0.0.1:8765."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, re, secrets, shutil, subprocess, tempfile, threading, time, urllib.request, urllib.error, urllib.parse, webbrowser
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STATE_ROOT=Path(os.environ.get("LEWM_DESKTOP_ROOT", Path.home()/"Desktop/AI"))/"lewm-wsl"
POINTER=STATE_ROOT/"active-experiment.json"
HTML_PATH=ROOT/"dashboard/index.html"
HIDDEN=getattr(subprocess,"CREATE_NO_WINDOW",0)
LABELS={"sigreg":"SIGReg","action_predictor":"Action predictor","combined":"SIGReg 0.09 + action 0.09","mse_only":"MSE only",
        "combined_s06":"SIGReg 0.06 + action 0.09","combined_s03":"SIGReg 0.03 + action 0.09",
        "combined_a100":"SIGReg 0.09 + action 1.0","action_w50":"Action 50 (no SIGReg)",
        "delta_l0_s09":"LDAD 0 + SIGReg 0.09","delta_l50_s09":"LDAD 50 + SIGReg 0.09",
        "delta_l50_s0":"LDAD 50 + SIGReg 0"}
DELTA_MODELS=("delta_l0_s09","delta_l50_s09","delta_l50_s0")
PAIRED_MODELS=("sigreg","action_predictor")
SINGLE_MODELS=tuple(key for key in LABELS if key not in PAIRED_MODELS)

def utc(): return datetime.now(timezone.utc).isoformat()
def read_json(path):
    try: return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError,ValueError,TypeError): return {}
def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError): return None
def timestamp(value):
    try:
        stamp=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if stamp.tzinfo is None: stamp=stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    except (ValueError,TypeError): return None
def age(value):
    stamp = timestamp(value)
    return max(0, time.time() - stamp) if stamp is not None else None
def host_path(value,experiment):
    value=str(value or "")
    if value.startswith("/") and str(experiment).startswith("\\\\"):
        parts=str(experiment).split("\\")
        return Path("\\\\"+parts[2]+"\\"+parts[3]+value.replace("/","\\"))
    return Path(value)
def tail(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0,2); length=stream.tell(); stream.seek(max(0,length-16384)); text=stream.read(16384).decode("utf-8","replace")
        return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]","","\n".join(text.splitlines()[-65:]))
    except OSError: return ""

def replay_asset(experiment, relative):
    """Resolve only JSON/PNG assets beneath the selected planning replay directory."""
    if not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", relative):
        raise ValueError("Invalid replay asset path")
    if any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError("Invalid replay asset path")
    if Path(relative).suffix.lower() not in (".json", ".png"):
        raise ValueError("Unsupported replay asset")
    selected = read_json(experiment / "planning" / "active.json").get("directory")
    if not isinstance(selected, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", selected):
        raise ValueError("No active planning replay")
    planning = (experiment / "planning").resolve(strict=True)
    directory = (planning / selected).resolve(strict=True)
    root = (directory / "replay").resolve(strict=True)
    target = (root / relative).resolve(strict=True)
    if directory.parent != planning or root.parent != directory or not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Replay asset is outside the selected replay directory")
    return target


def planning_snapshot(experiment, linux_experiment=None):
    directory = experiment / "planning"
    selected = read_json(directory / "active.json").get("directory")
    if selected:
        if not isinstance(selected, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", selected):
            return {}
        directory = directory / selected
    planning = read_json(directory / "results.json") or read_json(directory / "progress.json")
    if not planning or not selected:
        return planning
    live = False
    pid = planning.get("pid")
    if planning.get("status") == "running" and type(pid) is int and pid > 0:
        try:
            args = host_path(f"/proc/{pid}/cmdline", experiment).read_bytes().decode().split("\0")
            index = args.index("--experiment-dir")
            live = (args[index + 1] == str(linux_experiment or experiment)
                    and any(Path(arg).name in ("plan_delta_checkpoints.py", "plan_delta_episodes.py") for arg in args))
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            pass
    try:
        replay_asset(experiment, "manifest.json")
        replay_available = True
    except (OSError, ValueError, RuntimeError):
        replay_available = False
    return {**planning, "process_live": live, "directory": str(directory),
            "replay_available": replay_available, "replay_url": "/replay" if replay_available else None}


def evaluation_phases(directory, status, progress, results, test_windows, models=PAIRED_MODELS):
    """Read the requested phase summaries and attribute partial metrics by phase ID."""
    rows = []
    total = results.get("test_windows") or test_windows or progress.get("total_windows")
    for model in models:
        for precision in ("bf16_mixed", "fp32"):
            phase = f"{model}--{precision}"
            saved = read_json(directory / "phases" / phase / "summary.json")
            aggregate = results.get("models", {}).get(model, {}).get(precision, {})
            summary = saved if saved.get("complete") else {}
            if not summary and results.get("passed") and number(aggregate.get("validation_arithmetic_mse")) is not None:
                summary = aggregate
            if summary.get("model", model) != model or summary.get("precision", precision) != precision:
                summary = {}
            mse = number(summary.get("validation_arithmetic_mse"))
            if mse is None and results.get("passed"):
                comparison = results.get("comparisons", {}).get(precision, {}).get("original_validation_arithmetic", {})
                mse = number(comparison.get(model + "_mse"))
            complete = bool(summary) or mse is not None
            windows = summary.get("windows", total)
            rows.append({"model": model, "precision": precision, "state": "complete" if complete else "waiting",
                         "mse": mse, "completed_windows": windows if complete else 0,
                         "total_windows": windows, "eta_seconds": 0 if complete else None})

    pending = {f"{row['model']}--{row['precision']}": row for row in rows if row["state"] == "waiting"}
    if not pending:
        return rows
    evaluating = status.get("phase") == "evaluate_next_embedding"
    pipeline_state = status.get("status")
    live = progress.get("phase")
    # On resume, an unfinished phase restarts from zero. Its old progress file
    # persists until the new evaluator publishes its first batch.
    started = timestamp(status.get("phase_started_utc"))
    sampled = timestamp(progress.get("updated_utc"))
    if evaluating and pipeline_state == "running" and started is not None and sampled is not None and sampled < started:
        live = None
    row = pending.get(live)
    if row is not None:
        row.update(mse=number(progress.get("validation_arithmetic_mse_so_far")),
                   completed_windows=progress.get("completed_windows", 0),
                   total_windows=progress.get("total_windows", row["total_windows"]))
    elif evaluating:
        # The previous phase can have published its summary while progress.json
        # still names it. The next phase has no measured MSE until its first batch.
        row = next(iter(pending.values()))
    if row is not None:
        if evaluating:
            row["state"] = {"running": "running", "paused": "paused",
                            "failed": "interrupted", "interrupted": "interrupted"}.get(pipeline_state, "waiting")
        else:
            row["state"] = "interrupted"
        if row["state"] == "running" and live in pending:
            row["eta_seconds"] = number(progress.get("estimated_remaining_phase_seconds"))
    return rows

def reference_sources(reference_path, comparisons):
    sources = {key: Path(reference_path) for key in PAIRED_MODELS} if reference_path else {}
    for entry in comparisons:
        key = entry.get("key")
        if key in SINGLE_MODELS and entry.get("windows_path"):
            sources[key] = Path(entry["windows_path"])
    return sources

def same_training_protocol(current, previous, *, allow_relocation):
    same_batches = previous.get("training_order") == current.get("training_order")
    if same_batches and current.get("manifest_sha256") and current["manifest_sha256"] == previous.get("manifest_sha256"):
        return True
    if not allow_relocation:
        return False
    if not same_batches:
        orders = []
        for manifest in (current, previous):
            order = manifest.get("training_order", {})
            batch = order.get("batch_size")
            windows = manifest.get("counts", {}).get("train", {}).get("windows")
            if (type(batch) is not int or batch < 1 or type(windows) is not int or windows < 1
                    or order.get("drop_last") is not False or order.get("steps_per_epoch") != math.ceil(windows / batch)):
                return False
            orders.append({key: val for key, val in order.items() if key not in ("batch_size", "steps_per_epoch")})
        if orders[0] != orders[1]:
            return False
    # Hash each sealed manifest before accepting relocated paths or a different batch grouping.
    # reference_training additionally verifies the actual sample-order prefix and epoch trace.
    for manifest in (current, previous):
        content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if manifest.get("manifest_sha256") != hashlib.sha256(encoded).hexdigest():
            return False
    protocol = ("schema_version", "split", "window", "counts", "normalization", "array_hashes")
    geometry = ("columns", "episode_count", "row_count", "episode_lengths_sha256", "episode_offsets_sha256")
    return (all(field in current and current[field] == previous.get(field) for field in protocol)
            and all(field in current.get("dataset", {}) and current["dataset"][field] == previous.get("dataset", {}).get(field)
                    for field in geometry))

def reference_scores(reference_path, manifest, comparisons=()):
    expected = manifest.get("array_hashes", {})
    rows = []
    verified_results = {}
    for model, reference in reference_sources(reference_path, comparisons).items():
        if reference not in verified_results:
            recorded = read_json(reference / "manifest.json").get("array_hashes", {})
            same_test = all(expected.get(key) and expected[key] == recorded.get(key)
                            for key in ("test_episode_ids", "test_clip_indices"))
            verified_results[reference] = read_json(reference / "evaluation/results.json") if same_test else {}
        results = verified_results[reference]
        if not results.get("passed"):
            continue
        for precision in ("bf16_mixed", "fp32"):
            summary = results.get("models", {}).get(model, {}).get(precision, {})
            mse = number(summary.get("validation_arithmetic_mse"))
            if summary.get("complete") and mse is not None:
                rows.append({"model": model, "precision": precision, "mse": mse, "experiment": reference.name})
    experiments = list(dict.fromkeys(row["experiment"] for row in rows))
    return {"experiment": ", ".join(experiments), "rows": rows} if rows else None

METRIC_FIELDS=("pred_loss", "loss", "sigreg_loss", "action_loss", "ldad_loss",
               "latent_cross_example_rms_std", "latent_cross_example_var",
               "latent_cross_example_max_range", "latent_global_max_range", "latent_cross_example_count")


class LongMetricHistory:
    """Consume appended native CSV rows once; keep a bounded, evenly spaced full-run history."""
    max_points = 3072

    def __init__(self):
        self.stamp = self.result = None
        self.offset = self.rows = 0
        self.sequential = True
        self.last_line = b""
        self.last_point = None
        self.stride = 1
        self.samples = []

    def update(self, path, stat, generation):
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, generation)
        if stamp == self.stamp:
            return self.result
        with path.open("rb", buffering=1024 * 1024) as source:
            if self.stamp:
                reset = (stamp[0] != self.stamp[0] or generation != self.stamp[3]
                         or stat.st_size <= self.stamp[1] or stat.st_size < self.offset)
                if not reset and self.last_line:
                    source.seek(self.offset - len(self.last_line))
                    reset = source.read(len(self.last_line)) != self.last_line
                if reset:
                    self.__init__()
            source.seek(self.offset)
            if self.offset == 0:
                header = source.readline()
                if not header.endswith(b"\n"):
                    return {"points": [], "path": str(path), "truncated": False, "logged_steps": 0}
                fields = next(csv.reader([header.decode("utf-8-sig")]))
                self.columns = [(fields.index("fit/" + f), f) for f in METRIC_FIELDS if "fit/" + f in fields]
                self.step_column = fields.index("step")
                self.offset = len(header)
            while self.offset < stat.st_size:
                line = source.readline()
                if not line.endswith(b"\n") or self.offset + len(line) > stat.st_size:
                    break
                # Native MetricsWriter writes only scalar numbers and empty fields.
                fields = line.rstrip(b"\r\n").split(b",")
                point = {"step": int(fields[self.step_column])}
                for index, name in self.columns:
                    if fields[index]:
                        value = number(fields[index])
                        if value is not None:
                            point[name] = value
                self.rows += 1
                self.sequential &= point["step"] == self.rows and "pred_loss" in point
                if point["step"] == 1 or point["step"] % self.stride == 0:
                    self.samples.append(point)
                if len(self.samples) >= self.max_points:
                    self.stride *= 2
                    self.samples = [p for p in self.samples if p["step"] == 1 or p["step"] % self.stride == 0]
                self.last_point, self.last_line = point, line
                self.offset += len(line)
        points = self.samples.copy()
        if self.last_point and (not points or points[-1] is not self.last_point):
            points.append(self.last_point)
        self.result = {"points": points, "path": str(path), "truncated": False,
                       "sample_stride": self.stride, "logged_steps": self.rows, "sequential": self.sequential}
        self.stamp = stamp
        return self.result


class Monitor:
    def __init__(self,pointer,helpers=ROOT):
        self.pointer,self.helpers=Path(pointer),Path(helpers)
        self.token=secrets.token_urlsafe(28)
        self.lock=threading.RLock(); self.stop=threading.Event()
        self.payload={"available":False,"sampled_utc":utc(),"message":"Preparing the active experiment."}
        self.metric_cache={}; self.reference_curve_cache={}; self.rates={}; self.rate_activity={}; self.rate_baselines={}; self.gpu_cache={}; self.gpu_time=0; self.control_job=None
        self.long_histories={}; self.prefix_hashes={}
    def metrics(self,path,all_steps=False, multi_epoch=False, generation=0):
        try: stat=path.stat(); stamp=(stat.st_mtime_ns,stat.st_size)
        except OSError: return {"points":[],"path":str(path),"truncated":False}
        if multi_epoch:
            history = self.long_histories.setdefault(path, LongMetricHistory())
            return history.update(path, stat, generation)
        key = (str(path), all_steps)
        cached=self.metric_cache.get(key)
        if cached and cached[0]==stamp: return cached[1]
        try:
            limit=16*1024*1024
            with path.open("rb") as stream:
                header=stream.readline(); truncated=stat.st_size>limit
                if truncated: stream.seek(max(len(header),stat.st_size-limit)); stream.readline()
                body=stream.read(limit)
            rows={}
            for row in csv.DictReader(io.StringIO((header+body).decode("utf-8-sig","replace"))):
                step=number(row.get("step"))
                if step is None: continue
                point=rows.setdefault(int(step),{"step":int(step)})
                for field in METRIC_FIELDS:
                    value = number(row.get("fit/" + field))
                    if value is not None: point[field] = value
                # Retain the earlier single-auxiliary API for the paired models.
                auxiliary = [point[key] for key in ("sigreg_loss", "action_loss") if key in point]
                if len(auxiliary) == 1: point["aux_loss"] = auxiliary[0]
                else: point.pop("aux_loss", None)
            points=[rows[k] for k in sorted(rows) if "pred_loss" in rows[k]]
            if not all_steps and len(points)>1400:
                last=points[-1]; points=points[::math.ceil(len(points)/1400)]
                if points[-1] is not last: points.append(last)
            result={"points":points,"path":str(path),"truncated":truncated}
        except (OSError,csv.Error):
            result=cached[1] if cached else {"points":[],"path":str(path),"truncated":False}
        self.metric_cache[key]=(stamp,result)
        return result
    def prefix_hash(self, path, windows, key):
        """Advance a SHA state over newly committed rows of one trace generation."""
        length = windows * 8
        previous, digest = self.prefix_hashes.get(key, (0, hashlib.sha256()))
        if length < previous:
            previous, digest = 0, hashlib.sha256()
        if previous != length:
            digest = digest.copy()
            with path.open("rb") as source:
                source.seek(previous)
                remaining = length - previous
                while remaining:
                    chunk = source.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        return None
                    digest.update(chunk)
                    remaining -= len(chunk)
            self.prefix_hashes[key] = length, digest
        return digest.hexdigest()

    def reference_curve(self, reference, model, manifest):
        """Cache a complete trace and its logged losses after checking the training audit."""
        run = reference / "runs" / model
        trace_path = run / "training-indices.i64"
        audit_path = run / "final-audit.json"
        metrics_path = run / "metrics.csv"
        manifest_hash = manifest["manifest_sha256"]
        key = (str(run), manifest_hash)
        try:
            stats = [p.stat() for p in (audit_path, trace_path, metrics_path)]
            stamp = tuple((s.st_size, s.st_mtime_ns) for s in stats)
            cached = self.reference_curve_cache.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
            audit = read_json(audit_path)
            details = audit.get("audit", {})
            data = details.get("data_provenance", {})
            if not audit.get("passed") or details.get("manifest_sha256") != manifest_hash:
                return None
            epoch_windows = manifest["counts"]["train"]["windows"]
            epochs = data.get("epochs", 1)
            if type(epochs) is not int or epochs < 1:
                return None
            windows = epoch_windows * epochs
            steps = manifest["training_order"]["steps_per_epoch"] * epochs
            if data.get("windows") != windows or data.get("completed_steps") != steps or stats[1].st_size != 8 * windows:
                return None
            identity = (str(trace_path), stamp)
            full_hash = self.prefix_hash(trace_path, windows, ("audit", identity))
            if full_hash != data.get("order_sha256"):
                return None
            first_hash = full_hash if epochs == 1 else self.prefix_hash(trace_path, epoch_windows, ("first", identity))
            trace = {"path": trace_path, "windows": windows, "full": full_hash, "first": first_hash}
            metrics = self.metrics(metrics_path, all_steps=True, multi_epoch=model in DELTA_MODELS)
            points = metrics["points"]
            aligned = (metrics.get("sequential") and metrics["logged_steps"] == steps if model in DELTA_MODELS
                       else len(points) == steps and all(p["step"] == i for i, p in enumerate(points, 1)))
            if metrics["truncated"] or not aligned:
                return None
        except (OSError, KeyError, TypeError):
            return None
        curve = {"key": model, "label": LABELS[model], "experiment": reference.name, "epochs": epochs,
                 "batch_size": manifest["training_order"].get("batch_size"), "metrics": metrics}
        # Bind the client token to the verified source generation and exact
        # serialized curve. This cost is paid only when the files change.
        signature = hashlib.sha256(json.dumps([manifest_hash, stamp, curve], separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
        verified = (trace, curve, signature)
        self.reference_curve_cache[key] = (stamp, verified)
        return verified

    def reference_training(self, reference_path, manifest, progress, comparisons=(), active_trace=None, multi_epoch=False):
        if not manifest.get("manifest_sha256"):
            return [], None
        sources = reference_sources(reference_path, comparisons)
        seen = progress.get("windows_seen", 0)
        if not isinstance(seen, int) or seen < 0 or progress.get("manifest_sha256", manifest["manifest_sha256"]) != manifest["manifest_sha256"]:
            return [], None
        curves = []
        signatures = []
        first_epoch_hash = None
        reference_hashes = {}
        planned = read_json(active_trace.parent / "order-protocol.json") if multi_epoch and active_trace else {}
        manifests = {}
        for model, reference in sources.items():
            if reference not in manifests:
                manifests[reference] = read_json(reference / "manifest.json")
            previous = manifests[reference]
            if not same_training_protocol(manifest, previous, allow_relocation=multi_epoch):
                continue
            verified = self.reference_curve(reference, model, previous)
            if verified is None:
                continue
            trace, curve, signature = verified
            # A historical single pass overlaps only the current run's first epoch.
            # Equally long runs must agree on the full audited sequence as well.
            if first_epoch_hash is not None and first_epoch_hash != trace["first"]:
                return [], None
            if reference_hashes.setdefault(trace["windows"], trace["full"]) != trace["full"]:
                return [], None
            first_epoch_hash = trace["first"]
            if planned and (planned.get("manifest_sha256") != manifest["manifest_sha256"]
                            or (planned.get("epoch_sha256") or [None])[0] != first_epoch_hash
                            or (curve["epochs"] > 1 and planned.get("order_sha256") != trace["full"])):
                continue
            if seen:
                try:
                    if seen <= trace["windows"]:
                        expected = self.prefix_hash(trace["path"], seen, ("live prefix", trace["full"]))
                        if expected != progress.get("order_sha256"):
                            continue
                    else:
                        if not active_trace or not multi_epoch:
                            continue
                        key = ("completed reference", str(active_trace), active_trace.stat().st_ino, progress.get("resume_count", 0))
                        if self.prefix_hash(active_trace, trace["windows"], key) != trace["full"]:
                            continue
                except OSError:
                    continue
            curves.append(curve)
            signatures.append(signature)
        token = hashlib.sha256("".join(signatures).encode()).hexdigest() if curves and (multi_epoch or len(curves) == len(sources)) else None
        return curves, token
    def queued_training(self, entries, manifest, active_keys):
        queued = []
        seen = set(active_keys)
        for entry in entries:
            key = entry.get("key")
            if key not in SINGLE_MODELS or key in seen or not entry.get("windows_path"):
                continue
            seen.add(key)
            folder = Path(entry["windows_path"])
            run = folder / "runs" / key
            queued_manifest = read_json(folder / "manifest.json")
            matched = manifest.get("manifest_sha256") and queued_manifest.get("manifest_sha256") == manifest["manifest_sha256"]
            progress = read_json(run / "progress.json") if matched else {}
            metrics = self.metrics(run / "metrics.csv", all_steps=True, multi_epoch=key in DELTA_MODELS,
                                   generation=progress.get("resume_count", 0)) if matched else {"points": [], "truncated": False}
            queued.append({"key": key, "label": entry.get("label", LABELS[key]), "experiment": folder.name, "queue_position": len(queued) + 1,
                           "state": progress.get("status") or ("running" if metrics["points"] else "queued"), "metrics": metrics})
        return queued
    def gpu(self):
        if time.monotonic()-self.gpu_time<5: return self.gpu_cache
        self.gpu_time=time.monotonic()
        executable=shutil.which("nvidia-smi")
        query=["--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw","--format=csv,noheader,nounits"]
        command=[executable,*query] if executable else ["wsl.exe","-d","Ubuntu-24.04","--","nvidia-smi",*query]
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=5,creationflags=HIDDEN,check=True)
            row=next(csv.reader(io.StringIO(process.stdout)))
            self.gpu_cache={"available":True,"name":row[0].strip(),"utilization":number(row[1]),"memory_used_mib":number(row[2]),"memory_total_mib":number(row[3]),"power_watts":number(row[4]),"sampled_utc":utc()}
        except (OSError,subprocess.SubprocessError,StopIteration,IndexError):
            self.gpu_cache={"available":False,"sampled_utc":utc()}
        return self.gpu_cache
    def rate(self,key,progress,running=False):
        steps=number(progress.get("optimizer_steps")) or 0
        windows=number(progress.get("windows_seen")) or 0
        elapsed=number(progress.get("elapsed_seconds")) or 0
        samples=self.rates.setdefault(key,deque(maxlen=20))
        previous_active=self.rate_activity.get(key,False)
        self.rate_activity[key]=running
        empty={"steps_per_second":None,"windows_per_second":None,"basis":"warming up"}
        if not running:
            samples.clear()
            self.rate_baselines[key]=steps
            return empty
        if not previous_active:
            samples.clear()
            self.rate_baselines[key]=steps
        # First new progress after startup/resume anchors the timing window.
        # A second progress report measures work without worker-startup/paused time.
        if steps<20 or steps<=self.rate_baselines.get(key,steps):
            return empty
        if samples and (steps<samples[-1][1] or elapsed<samples[-1][0]):
            samples.clear()
        if not samples or samples[-1][1]!=steps:
            samples.append((elapsed,steps,windows))
        recent=[x for x in samples if elapsed-x[0]<=180]
        if len(recent)<2 or recent[-1][0]<=recent[0][0]:
            return empty
        dt=recent[-1][0]-recent[0][0]
        return {"steps_per_second":(recent[-1][1]-recent[0][1])/dt,
                "windows_per_second":(recent[-1][2]-recent[0][2])/dt,"basis":"recent"}
    def checkpoint(self,run,progress,completion,experiment):
        info=read_json(run/"checkpoint.json") or read_json(run/"latest-checkpoint.json")
        candidates=[info.get("path"),info.get("checkpoint_path"),info.get("checkpoint"),progress.get("checkpoint_path"),progress.get("last_checkpoint_path"),completion.get("final_ckpt_path")]
        for value in candidates:
            if not value: continue
            path=host_path(value,experiment)
            try:
                stat=path.stat()
                return {"available":True,"path":str(path),"age_seconds":max(0,time.time()-stat.st_mtime),"size_bytes":stat.st_size,"step":info.get("global_step",info.get("step"))}
            except OSError: pass
        saved=info.get("saved_utc",info.get("updated_utc"))
        return {"available":bool(saved),"age_seconds":age(saved),"step":info.get("global_step")}
    def collect(self):
        pointer=read_json(self.pointer)
        if not pointer.get("windows_path"):
            return {"available":False,"sampled_utc":utc(),"message":"Preparing the active experiment.","gpu":self.gpu(),"pointer_path":str(self.pointer)}
        experiment=Path(pointer["windows_path"])
        status=read_json(experiment/"status.json"); plan=read_json(experiment/"plan.json"); manifest=read_json(experiment/"manifest.json")
        control=read_json(experiment/"control.json"); names=pointer.get("model_names",pointer.get("models",{}))
        if not isinstance(names,dict): names={}
        model_keys = tuple(key for key in LABELS if key in names) or PAIRED_MODELS
        shared = set(model_keys) == set(DELTA_MODELS) and (pointer.get("shared_batch") is True
                    or plan.get("training_backend") == "native_pytorch_shared_batch" or (experiment / "members").is_dir())
        group_progress = read_json(experiment / "group-progress.json") if shared else {}
        group_checkpoint = read_json(experiment / "group-checkpoint.json") if shared else {}
        group_telemetry = read_json(experiment / "gpu-telemetry.json") if shared else {}
        models=[]
        for key in model_keys:
            decoder_free = shared and key == "delta_l0_s09"
            label = "SIGReg 0.09 (no LDAD)" if decoder_free else LABELS[key]
            member = experiment / "members" / key if shared else experiment
            run=member/"runs"/key; progress=read_json(run/"progress.json"); completion=read_json(run/"runcompletion.json")
            if shared:
                # One optimizer cursor commits all three members. Each objective's
                # loss and CSV stay local to its own model.
                own_losses = group_progress.get("members", {}).get(key, {}).get("last_losses", progress.get("last_losses", {}))
                progress = {**progress, **group_progress, "last_losses": own_losses}
            name=names.get(key,"")
            if isinstance(name,dict): name=name.get("name",name.get("model_name",""))
            metrics_path=run/"metrics.csv"
            if not metrics_path.exists() and completion.get("registry_run"): metrics_path=host_path(completion["registry_run"],experiment)/"metrics.csv"
            metrics=self.metrics(metrics_path, all_steps=(key in SINGLE_MODELS), multi_epoch=key in DELTA_MODELS,
                                 generation=progress.get("resume_count", 0))
            active=status.get("phase")==("train_shared" if shared else "train_"+key)
            state="complete" if completion.get("passed") or progress.get("status")=="complete" else progress.get("status","waiting")
            if active and status.get("status") in ("paused", "failed"): state=status["status"]
            epochs = int(progress.get("epochs") or plan.get("epochs") or (50 if key in DELTA_MODELS else 1))
            epoch_steps = manifest.get("training_order", {}).get("steps_per_epoch")
            epoch_windows = manifest.get("counts", {}).get("train", {}).get("windows")
            progress = {**progress, "epochs": epochs, "steps_per_epoch": epoch_steps,
                        "expected_steps": progress.get("expected_steps", epoch_steps * epochs if epoch_steps else plan.get("steps_per_model")),
                        "expected_windows": progress.get("expected_windows", epoch_windows * epochs if epoch_windows else plan.get("train_windows"))}
            latest = metrics["points"][-1] if metrics["points"] else {}
            last_losses = {}
            loss_fields = ("pred_loss", "loss", "sigreg_loss", "action_loss")
            for field in (*loss_fields, "ldad_loss") if key in DELTA_MODELS and not decoder_free else loss_fields:
                observed = number(progress.get("last_losses", {}).get(field))
                last_losses[field] = observed if observed is not None else number(latest.get(field))
            pred = last_losses["pred_loss"]
            rate=self.rate(str(experiment)+key,progress,active and status.get("status")=="running")
            speed=rate.get("steps_per_second")
            eta=max(0,(number(progress.get("expected_steps")) or 0)-(number(progress.get("optimizer_steps")) or 0))/speed if speed and speed>0 else None
            if shared:
                saved = group_checkpoint.get("saved_utc")
                checkpoint = {"available": bool(saved), "age_seconds": age(saved), "step": group_checkpoint.get("global_step")}
                telemetry_age = age(group_telemetry.get("sampled_utc"))
                if active and number(group_telemetry.get("eta_seconds")) is not None and telemetry_age is not None and telemetry_age < 120:
                    eta = number(group_telemetry["eta_seconds"])
                elif eta is None:
                    eta = number(group_progress.get("eta_seconds"))
            else:
                checkpoint = self.checkpoint(run,progress,completion,member)
            models.append({"key":key,"label":label,"ldad_enabled":key in DELTA_MODELS and not decoder_free,"name":name,"state":state,"active":active,"progress":progress,"last_pred_loss":pred,"last_losses":last_losses,"metrics":metrics,"throughput":rate,"recent_eta_seconds":eta,"checkpoint":checkpoint})
        log=host_path(status.get("log",""),experiment) if status.get("log") else None
        if log is None or not log.is_file():
            active=next((m for m in models if m["active"]),None)
            log=(experiment/"training.log" if shared else experiment/"runs"/(active["key"] if active else model_keys[0])/"console.log")
        test_windows = manifest.get("counts", {}).get("test", {}).get("windows")
        if shared:
            results, evaluation_progress, phases = {}, {}, []
            for key in model_keys:
                directory = experiment / "members" / key / "evaluation"
                complete = read_json(directory / "results.json")
                live = read_json(directory / "progress.json")
                member_status = dict(status)
                if status.get("phase") == f"evaluate_{key}":
                    member_status["phase"] = "evaluate_next_embedding"
                    evaluation_progress = live
                phases.extend(evaluation_phases(directory, member_status, live, complete, test_windows, (key,)))
        else:
            evaluation_dir = experiment / "evaluation"
            results = read_json(evaluation_dir / "results.json")
            evaluation_progress = read_json(evaluation_dir / "progress.json")
            phase_progress = evaluation_progress or (status.get("child_progress", {}) if status.get("phase") == "evaluate_next_embedding" else {})
            phases = evaluation_phases(evaluation_dir, status, phase_progress, results, test_windows, model_keys)
        comparisons = [entry for entry in (pointer.get("comparison_experiments") or ()) if entry.get("key") not in model_keys]
        reference = reference_scores(pointer.get("reference_experiment_windows_path"), manifest, comparisons)
        reference_training_models = []
        reference_training_token = None
        if shared or (len(model_keys) == 1 and model_keys[0] in SINGLE_MODELS):
            key = model_keys[0]
            member = experiment / "members" / key if shared else experiment
            reference_training_models, reference_training_token = self.reference_training(
                pointer.get("reference_experiment_windows_path"), manifest, models[0]["progress"], comparisons,
                member / "runs" / key / "training-indices.i64", key in DELTA_MODELS)
        queued = self.queued_training(pointer.get("queued_experiments") or (), manifest, model_keys)
        batch_size = manifest.get("training_order", {}).get("batch_size") or plan.get("batch_size") or pointer.get("training_batch_size")
        return {"available":True,"sampled_utc":utc(),"label":pointer.get("label",experiment.name),"experiment_path":str(experiment),"host_output":pointer.get("host_output"),"status":status,"control":control,"plan":plan,"counts":manifest.get("counts",{}),"batch_size":batch_size,"models":models,"shared_batch":shared,"group_telemetry":group_telemetry,"group_progress":group_progress,"queued_training_models":queued,"gpu":self.gpu(),"planning":planning_snapshot(experiment, pointer.get("linux_experiment")),"evaluation":results,"evaluation_progress":evaluation_progress,"evaluation_phases":phases,"reference_evaluation":reference,"reference_training_models":reference_training_models,"reference_training_token":reference_training_token,"log_tail":tail(log),"log_path":str(log),"controls_available":all((self.helpers/n).is_file() for n in ("Pause-LeWM-Training.ps1","Resume-LeWM-Training.ps1"))}
    def refresh(self):
        try: payload=self.collect()
        except Exception as error:
            with self.lock: payload={**self.payload,"collector_error":str(error),
                                     "reference_training_models": [], "reference_training_token": None}
        with self.lock: self.payload=payload
    def run(self):
        while not self.stop.is_set(): self.refresh(); self.stop.wait(2)
    def snapshot(self, reference_token=None):
        with self.lock:
            result = {**self.payload, "control_token": self.token, "control_job": self.control_job}
        current = result.setdefault("reference_training_token", None)
        curves = result.setdefault("reference_training_models", [])
        omit = bool(current and reference_token == current and curves)
        result["reference_training_omitted"] = omit
        if omit:
            del result["reference_training_models"]
        return result
    def request_control(self,action):
        with self.lock:
            if self.control_job and self.control_job.get("status")=="running": raise ValueError("A control request is already in progress.")
            pointer = read_json(self.pointer)
            if action == "resume" and pointer.get("windows_path") and planning_snapshot(
                    Path(pointer["windows_path"]), pointer.get("linux_experiment")).get("process_live"):
                raise ValueError("The checkpoint planning test is running. Resume training after it finishes.")
            script=self.helpers/("Pause-LeWM-Training.ps1" if action=="pause" else "Resume-LeWM-Training.ps1")
            if not script.is_file(): raise ValueError("Training control helper is not available.")
            self.control_job={"action":action,"status":"running","requested_utc":utc()}
        def execute():
            try:
                # A detached training launcher can inherit pipe handles on Windows.
                # Regular files let the short control helper finish independently.
                with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as output, \
                        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errors:
                    process=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",str(script)],
                                           stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                           timeout=60, creationflags=HIDDEN)
                    output.seek(0)
                    errors.seek(0)
                    stdout, stderr = output.read().strip(), errors.read().strip()
                try: response=json.loads(stdout)
                except ValueError: response={"message":stdout or stderr}
                result={"action":action,"status":"accepted" if process.returncode==0 else "failed","response":response,"finished_utc":utc()}
            except (OSError,subprocess.SubprocessError) as error: result={"action":action,"status":"failed","error":str(error),"finished_utc":utc()}
            with self.lock: self.control_job=result
            self.refresh()
        threading.Thread(target=execute,daemon=True).start()
        return {"accepted":True,"action":action,"message":"Request sent. The confirmed state will appear in the monitor."}

def handler_for(monitor,html_path=HTML_PATH):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def local(self):
            port=self.server.server_address[1]
            return self.headers.get("Host","") in ("127.0.0.1:"+str(port),"localhost:"+str(port))
        def send(self,status,content,mime="application/json; charset=utf-8"):
            body=content if isinstance(content,bytes) else content.encode("utf-8") if isinstance(content,str) else json.dumps(content,allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type",mime); self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store"); self.send_header("X-Content-Type-Options","nosniff"); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if not self.local(): self.send(403,{"error":"Local host only"})
            elif self.path in ("/","/index.html"):
                try: self.send(200,Path(html_path).read_text(encoding="utf-8"),"text/html; charset=utf-8")
                except OSError: self.send(503,{"error":"Dashboard HTML is missing"})
            elif url.path == "/replay":
                try: self.send(200,Path(html_path).with_name("replay.html").read_text(encoding="utf-8"),"text/html; charset=utf-8")
                except OSError: self.send(404,{"error":"Replay viewer is not available"})
            elif url.path == "/api/replaymanifest" or url.path.startswith("/replay-assets/"):
                try:
                    relative = "manifest.json" if url.path == "/api/replaymanifest" else urllib.parse.unquote(url.path[len("/replay-assets/"):], errors="strict")
                    pointer = read_json(monitor.pointer)
                    if not pointer.get("windows_path"):
                        raise ValueError("No active experiment")
                    asset = replay_asset(Path(pointer["windows_path"]), relative)
                    self.send(200,asset.read_bytes(),"image/png" if asset.suffix.lower() == ".png" else "application/json; charset=utf-8")
                except (OSError, ValueError, RuntimeError): self.send(404,{"error":"Replay asset is not available"})
            elif url.path == "/api/status":
                tokens = urllib.parse.parse_qs(url.query).get("reference_token", [])
                self.send(200, monitor.snapshot(tokens[0] if len(tokens) == 1 else None))
            elif self.path=="/favicon.ico": self.send(204,"")
            else: self.send(404,{"error":"Not found"})
        def do_POST(self):
            if not self.local() or not secrets.compare_digest(self.headers.get("X-Control-Token",""),monitor.token): self.send(403,{"error":"Invalid local control token"}); return
            port=self.server.server_address[1]; origin=self.headers.get("Origin")
            if origin and origin not in ("http://127.0.0.1:"+str(port),"http://localhost:"+str(port)): self.send(403,{"error":"Cross-origin controls are not allowed"}); return
            if self.path not in ("/api/control/pause","/api/control/resume"): self.send(404,{"error":"Unknown control"}); return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if length<0 or length>1024: raise ValueError("Invalid control request size")
                self.rfile.read(length)
                self.send(202,monitor.request_control(self.path.rsplit("/",1)[-1]))
            except ValueError as error: self.send(409,{"error":str(error)})
    return Handler

def test_evaluation_phases(directory):
    total = 100
    running = {"status": "running", "phase": "evaluate_next_embedding"}
    def rows(status=running, progress=None, results=None):
        return evaluation_phases(directory, status, progress or {}, results or {}, total)

    initial = rows({"status": "running", "phase": "train_sigreg"})
    assert len(initial) == 4 and all(row["state"] == "waiting" and row["mse"] is None for row in initial)
    first = directory / "phases" / "sigreg--bf16_mixed"
    first.mkdir(parents=True)
    (first / "summary.json").write_text(json.dumps({"complete": True, "model": "sigreg", "precision": "bf16_mixed",
                                                    "windows": total, "validation_arithmetic_mse": 0.04}))
    old = {"phase": "sigreg--bf16_mixed", "completed_windows": total, "validation_arithmetic_mse_so_far": 0.99}
    transition = rows(progress=old)
    assert [row["state"] for row in transition] == ["complete", "running", "waiting", "waiting"]
    assert [row["mse"] for row in transition] == [0.04, None, None, None]

    progress = {"phase": "sigreg--fp32", "completed_windows": 32, "total_windows": total,
                "validation_arithmetic_mse_so_far": 0.03, "estimated_remaining_phase_seconds": 12,
                "updated_utc": "2026-11-07T05:01:00Z"}
    assert rows(progress=progress)[1] == {"model": "sigreg", "precision": "fp32", "state": "running",
                                          "mse": 0.03, "completed_windows": 32, "total_windows": total, "eta_seconds": 12}
    for pipeline_state, expected in (("paused", "paused"), ("failed", "interrupted")):
        paused = rows({**running, "status": pipeline_state}, progress)
        assert paused[1]["state"] == expected and paused[1]["mse"] == 0.03 and paused[1]["eta_seconds"] is None
        between = rows({**running, "status": pipeline_state}, old)
        assert between[0]["state"] == "complete" and between[1]["state"] == expected and between[1]["mse"] is None
    restarted = rows({**running, "phase_started_utc": "2026-11-07T05:02:00Z"}, progress)
    assert restarted[1]["state"] == "running" and restarted[1]["completed_windows"] == 0 and restarted[1]["mse"] is None
    results = {"passed": True, "models": {"action_predictor": {"fp32": {"validation_arithmetic_mse": 0.0, "windows": total}}},
               "comparisons": {"bf16_mixed": {"original_validation_arithmetic": {"action_predictor_mse": 0.02}}}}
    finished = rows({"status": "complete", "phase": "complete"}, {}, results)
    assert [row["mse"] for row in finished] == [0.04, None, 0.02, 0.0]
    assert [row["state"] for row in finished] == ["complete", "waiting", "complete", "complete"]

def test_delta_dashboard(root):
    root.mkdir()
    monitor = Monitor(root / "pointer.json", root)
    monitor.gpu = lambda: {}
    # A relocated dataset's manifest can have a different hash but still use
    # identical training windows, statistics and data-order protocol.
    def sealed(path, normalization, batch=2, seed=42, drop_last=False):
        value = {"schema_version": 1, "split": {"seed": 32}, "window": {"span": 20}, "counts": {"train": {"windows": 3}},
                 "training_order": {"seed": seed, "batch_size": batch, "steps_per_epoch": math.ceil(3 / batch), "drop_last": drop_last}, "normalization": normalization, "array_hashes": {"train_clip_indices": "clips"},
                 "dataset": {"path": path, "columns": ["pixels", "action"], "episode_count": 2, "row_count": 21,
                             "episode_lengths_sha256": "lengths", "episode_offsets_sha256": "offsets"}}
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {**value, "manifest_sha256": digest}
    assert same_training_protocol(sealed("/a", {}), sealed("/b", {}), allow_relocation=True)
    assert not same_training_protocol(sealed("/a", {}), sealed("/b", {"action": [0.1]}), allow_relocation=True)
    assert same_training_protocol(sealed("/a", {}, 2), sealed("/b", {}, 3), allow_relocation=True)
    assert not same_training_protocol(sealed("/a", {}, 2), sealed("/b", {}, 3), allow_relocation=False)
    assert not same_training_protocol(sealed("/a", {}, 2), sealed("/b", {}, 3, seed=7), allow_relocation=True)
    assert not same_training_protocol(sealed("/a", {}, 2), sealed("/b", {}, 3, drop_last=True), allow_relocation=True)
    # A native append stream retains its first point, handles unfinished lines,
    # and reconciles a resume that replaces the abandoned tail.
    path = root / "long.csv"
    header = "step,fit/pred_loss,fit/ldad_loss,fit/loss\n"
    size = LongMetricHistory.max_points * 4 + 7
    with path.open("w", newline="") as stream:
        stream.write(header)
        for i in range(1, size + 1):
            stream.write(f"{i},.1,.03,1.6\n")
        stream.write(f"{size + 1},.1")
    first = monitor.metrics(path, multi_epoch=True)
    assert first["sample_stride"] > 1 and len(first["points"]) <= LongMetricHistory.max_points
    assert first["points"][0]["step"] == 1 and first["points"][-1]["step"] == first["logged_steps"] == size
    assert first["sequential"] and first["points"][-1]["ldad_loss"] == .03
    with path.open("a", newline="") as stream:
        stream.write(",.04,2.1\n")
    appended = monitor.metrics(path, multi_epoch=True)
    assert appended["logged_steps"] == size + 1 and appended["points"][-1]["loss"] == 2.1
    path.write_text(header + "1,.4,.2,10.4\n")
    assert monitor.metrics(path, multi_epoch=True, generation=1)["points"] == [
        {"step": 1, "pred_loss": .4, "ldad_loss": .2, "loss": 10.4}]

    active, old, delta = (root / name for name in ("active", "old", "delta"))
    manifest = {"manifest_sha256": "fixture", "counts": {"train": {"windows": 4}, "test": {"windows": 3}},
                "training_order": {"steps_per_epoch": 2, "batch_size": 2}}
    for folder in (active, old, delta):
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(manifest))
    first_epoch = b"".join(i.to_bytes(8, "little") for i in [3, 4, 6, 5])
    schedule = first_epoch + b"".join(i.to_bytes(8, "little") for i in [5, 3, 4, 6])
    checksum = lambda data: hashlib.sha256(data).hexdigest()
    for folder, key, epochs, trace in ((old, "combined", 1, first_epoch), (delta, "delta_l0_s09", 2, schedule)):
        run = folder / "runs" / key
        run.mkdir(parents=True)
        (run / "training-indices.i64").write_bytes(trace)
        (run / "metrics.csv").write_text(header + "".join(f"{s},.1,.2,.3\n" for s in range(1, 2*epochs+1)))
        audit = {"manifest_sha256": "fixture", "data_provenance": {"windows": 4*epochs,
                 "completed_steps": 2*epochs, "epochs": epochs, "order_sha256": checksum(trace)}}
        (run / "final-audit.json").write_text(json.dumps({"passed": True, "audit": audit}))
    key = "delta_l50_s0"
    run = active / "runs" / key
    run.mkdir(parents=True)
    (root / "pointer.json").write_text(json.dumps({"windows_path": str(active), "model_names": {key: key},
        "comparison_experiments": [{"key": "combined", "windows_path": str(old)}, {"key": "delta_l0_s09", "windows_path": str(delta)}],
        "queued_experiments": [{"key": "delta_l50_s09", "windows_path": str(delta)}]}))
    (run / "training-indices.i64").write_bytes(schedule[:48])
    protocol = {"manifest_sha256": "fixture", "epoch_sha256": [checksum(first_epoch), checksum(schedule[32:])],
                "order_sha256": checksum(schedule)}
    (run / "order-protocol.json").write_text(json.dumps(protocol))
    progress = {"windows_seen": 6, "optimizer_steps": 3, "completed_epochs": 1, "current_epoch": 2,
                "epochs": 2, "order_sha256": checksum(schedule[:48])}
    (run / "progress.json").write_text(json.dumps(progress))
    (run / "metrics.csv").write_text(header + "1,.1,.03,1.6\n2,.1,.03,1.6\n3,.1,.02,1.1\n")
    (active / "status.json").write_text(json.dumps({"status":"running","phase":"train_"+key}))
    (active / "planning").mkdir()
    (active / "planning/progress.json").write_text(json.dumps({"completed_trials": 3, "expected_trials": 1500}))
    monitor.refresh()
    status = monitor.snapshot()
    model, = status["models"]
    assert model["key"] == key and model["active"] and model["last_losses"]["ldad_loss"] == .02
    assert model["progress"]["current_epoch"] == 2 and model["progress"]["expected_steps"] == 4
    assert [m["key"] for m in status["reference_training_models"]] == ["combined", "delta_l0_s09"]
    assert [m["epochs"] for m in status["reference_training_models"]] == [1, 2]
    assert all(m["batch_size"] == 2 for m in status["reference_training_models"])
    assert monitor.snapshot(status["reference_training_token"])["reference_training_omitted"]
    assert status["queued_training_models"][0]["key"] == "delta_l50_s09"
    assert status["planning"]["completed_trials"] == 3
    checkpoint_planning = active / "planning" / "checkpoint-18411"
    checkpoint_planning.mkdir()
    (active / "planning/active.json").write_text(json.dumps({"directory": "checkpoint-18411"}))
    checkpoint_progress = {"status": "running", "pid": 2147483647, "checkpoint_step": 18411,
        "completed_trials": 3, "expected_trials": 60, "active_model": key,
        "models": {key: {"completed_trials": 3, "expected_trials": 20, "successes": 1, "success_rate_percent": 100/3}}}
    (checkpoint_planning / "progress.json").write_text(json.dumps(checkpoint_progress))
    monitor.refresh()
    assert monitor.snapshot()["planning"]["models"][key]["successes"] == 1
    assert monitor.snapshot()["planning"]["process_live"] is False
    from unittest.mock import patch
    process_args = checkpoint_planning / "cmdline"
    process_args.write_bytes(b"python\0/code/scripts/plan_delta_checkpoints.py\0--experiment-dir\0/test/group\0")
    with patch(__name__ + ".host_path", return_value=process_args):
        assert planning_snapshot(active, "/test/group")["process_live"]
        assert not planning_snapshot(active, "/another/group")["process_live"]
        process_args.write_bytes(b"python\0/code/scripts/other.py\0--experiment-dir\0/test/group\0")
        assert not planning_snapshot(active, "/test/group")["process_live"]
    final = {**checkpoint_progress, "status": "complete", "passed": True, "completed_trials": 60}
    (checkpoint_planning / "results.json").write_text(json.dumps(final))
    assert planning_snapshot(active)["passed"] and not planning_snapshot(active)["process_live"]
    # Continuous episode evaluation has separate outputs and separate denominators.
    continuous = active / "planning" / "episodes-checkpoint-18411"
    continuous.mkdir()
    (active / "planning/active.json").write_text(json.dumps({"directory": continuous.name}))
    episode_progress = {"status": "running", "pid": 2147483647, "checkpoint_step": 18411,
        "evaluation_mode": "continuous_episodes", "completed_episodes": 2, "expected_episodes": 50,
        "completed_tasks": 7, "expected_tasks": 150,
        "models": {key: {"status": "running", "completed_episodes": 2, "expected_episodes": 50,
            "episode_successes": 1, "episode_success_rate_percent": 50,
            "completed_tasks": 7, "expected_tasks": 150, "task_successes": 6,
            "task_success_rate_percent": 600/7, "zero_action_successes": 1}}}
    (continuous / "progress.json").write_text(json.dumps(episode_progress))
    with patch(__name__ + ".host_path", return_value=process_args):
        process_args.write_bytes(b"python\0/code/scripts/plan_delta_episodes.py\0--experiment-dir\0/test/group\0")
        snapshot = planning_snapshot(active, "/test/group")
        assert snapshot["process_live"] and snapshot["evaluation_mode"] == "continuous_episodes"
        assert snapshot["models"][key]["episode_successes"] == 1
        assert snapshot["models"][key]["task_successes"] == 6
        assert snapshot["models"][key]["zero_action_successes"] == 1
        assert not planning_snapshot(active, "/another/group")["process_live"]
        process_args.write_bytes(b"python\0/code/scripts/not_plan_delta_episodes.py\0--experiment-dir\0/test/group\0")
        assert not planning_snapshot(active, "/test/group")["process_live"]
        process_args.write_bytes(b"python\0/code/scripts/plan_delta_episodes.py.bak\0--experiment-dir\0/test/group\0")
        assert not planning_snapshot(active, "/test/group")["process_live"]
    episode_final = {**episode_progress, "status": "complete", "passed": True}
    (continuous / "results.json").write_text(json.dumps(episode_final))
    assert planning_snapshot(active)["passed"] and not planning_snapshot(active)["process_live"]
    assert read_json(checkpoint_planning / "results.json") == final
    (active / "planning/active.json").write_text(json.dumps({"directory": "../elsewhere"}))
    assert planning_snapshot(active) == {}
    (active / "planning/active.json").unlink()
    # Reject an unmatched later epoch while retaining a verified first-pass comparator.
    (run / "order-protocol.json").write_text(json.dumps({**protocol, "order_sha256": "wrong"}))
    assert [m["key"] for m in monitor.collect()["reference_training_models"]] == ["combined"]
    (run / "order-protocol.json").write_text(json.dumps({**protocol, "epoch_sha256": ["wrong"]}))
    assert monitor.collect()["reference_training_models"] == []

    group = root / "shared"
    group.mkdir()
    (group / "manifest.json").write_text(json.dumps(manifest))
    (group / "control.json").write_text(json.dumps({"request": "run"}))
    (group / "plan.json").write_text(json.dumps({"epochs": 2, "training_backend": "native_pytorch_shared_batch"}))
    (group / "status.json").write_text(json.dumps({"status": "running", "phase": "train_shared"}))
    (group / "group-checkpoint.json").write_text(json.dumps({"global_step": 2, "saved_utc": utc()}))
    (group / "group-progress.json").write_text(json.dumps({**progress, "status": "running", "expected_steps": 4, "eta_seconds": 10}))
    (group / "gpu-telemetry.json").write_text(json.dumps({"sampled_utc": utc(), "gpu_util_percent_mean": 79,
                                                        "sample_count": 12, "group_steps_per_second": 2, "eta_seconds": .5}))
    (root / "pointer.json").write_text(json.dumps({"shared_batch": True, "windows_path": str(group),
        "model_names": {key: key + "-shared" for key in DELTA_MODELS},
        "comparison_experiments": [{"key":"combined", "windows_path":str(old)}]}))
    for i, key in enumerate(DELTA_MODELS):
        member = group / "members" / key
        run = member / "runs" / key
        run.mkdir(parents=True)
        (run / "training-indices.i64").write_bytes(schedule[:48])
        (run / "order-protocol.json").write_text(json.dumps(protocol))
        if key == "delta_l0_s09":
            (run / "metrics.csv").write_text("step,fit/pred_loss,fit/sigreg_loss,fit/loss\n" + "".join(f"{s},.{i+1},.2,.4\n" for s in range(1, 4)))
        else:
            (run / "metrics.csv").write_text(header + "".join(f"{s},.{i+1},.2,.4\n" for s in range(1, 4)))
        (member / "evaluation").mkdir()
    shared = monitor.collect()
    assert shared["shared_batch"] and shared["group_telemetry"]["gpu_util_percent_mean"] == 79
    assert [m["key"] for m in shared["models"]] == list(DELTA_MODELS)
    assert [m["last_pred_loss"] for m in shared["models"]] == [.1, .2, .3]
    control_model = shared["models"][0]
    assert not control_model["ldad_enabled"] and "ldad_loss" not in control_model["last_losses"]
    assert all("ldad_loss" not in p for p in control_model["metrics"]["points"])
    assert all(m["active"] and m["progress"]["optimizer_steps"] == 3 and m["recent_eta_seconds"] == .5
               and m["checkpoint"]["step"] == 2 for m in shared["models"])
    assert [m["key"] for m in shared["reference_training_models"]] == ["combined"]
    assert len(shared["evaluation_phases"]) == 6 and all(r["state"] == "waiting" for r in shared["evaluation_phases"])
    done, scoring, queued = DELTA_MODELS
    (group / "status.json").write_text(json.dumps({"status":"running", "phase":"evaluate_"+scoring}))
    (group / "members" / done / "evaluation/results.json").write_text(json.dumps({"passed":True,"models":{done:{
        p:{"complete":True, "windows":3, "validation_arithmetic_mse":.01} for p in ("bf16_mixed","fp32")}}}))
    (group / "members" / scoring / "evaluation/progress.json").write_text(json.dumps({"phase":scoring+"--bf16_mixed",
        "completed_windows":2,"total_windows":3,"validation_arithmetic_mse_so_far":.02}))
    assessed = monitor.collect()
    assert not any(m["active"] for m in assessed["models"])
    assert [(r["model"], r["state"], r["mse"]) for r in assessed["evaluation_phases"]] == [
        (done,"complete",.01),(done,"complete",.01),(scoring,"running",.02),(scoring,"waiting",None),
        (queued,"waiting",None),(queued,"waiting",None)]


def test_detached_control_output(root):
    """A resumed launcher may outlive its helper while retaining output handles."""
    import sys
    from unittest.mock import patch

    root.mkdir(parents=True)
    pointer = root / "pointer.json"
    pointer.write_text("{}")
    for name in ("Pause-LeWM-Training.ps1", "Resume-LeWM-Training.ps1"):
        (root / name).touch()
    child = root / "child.py"
    child.write_text(
        "import sys, time\nfrom pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "(root / 'child-ready').touch()\n"
        "deadline = time.monotonic() + 20\n"
        "while not (root / 'release-child').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(.02)\n"
        "(root / 'child-finished').touch()\n"
    )
    helper = root / "helper.py"
    helper.write_text(
        "import json, subprocess, sys\nfrom pathlib import Path\n"
        "root = Path(__file__).parent\n"
        "if sys.argv[1] == 'resume':\n"
        "    subprocess.Popen([sys.executable, str(root / 'child.py'), str(root)],\n"
        "        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,\n"
        "        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n"
        "print(json.dumps({'status': 'accepted', 'action': sys.argv[1]}), flush=True)\n"
    )
    monitor = Monitor(pointer, helpers=root)
    completed = threading.Event()
    monitor.refresh = completed.set
    original_run = subprocess.run

    def run_fixture(command, **kwargs):
        assert command[0] == "powershell.exe"
        action = "resume" if Path(command[-1]).name.startswith("Resume-") else "pause"
        return original_run([sys.executable, str(helper), action], **kwargs)

    try:
        with patch.object(subprocess, "run", side_effect=run_fixture):
            monitor.request_control("resume")
            assert completed.wait(5), "Resume helper waited for its detached descendant"
            assert monitor.control_job["status"] == "accepted"
            deadline = time.monotonic() + 5
            while not (root / "child-ready").exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert (root / "child-ready").exists() and not (root / "child-finished").exists()
            completed.clear()
            monitor.request_control("pause")
            assert completed.wait(5), "Pause was blocked by the completed resume helper"
            assert monitor.control_job["action"] == "pause" and monitor.control_job["status"] == "accepted"
    finally:
        (root / "release-child").touch()
        deadline = time.monotonic() + 5
        while not (root / "child-finished").exists() and time.monotonic() < deadline:
            time.sleep(.02)
        completed.wait(5)


def test_replay_routes(root):
    experiment = root / "experiment"
    planning = experiment / "planning"
    directory = planning / "episodes-checkpoint-100"
    directory.mkdir(parents=True)
    (planning / "active.json").write_text(json.dumps({"directory": directory.name}))
    (directory / "results.json").write_text(json.dumps({"status": "complete", "passed": True}))
    assert not planning_snapshot(experiment)["replay_available"]
    replay = directory / "replay"
    frames = replay / "135" / "ground_truth"
    frames.mkdir(parents=True)
    manifest = {"episodes": [{"episode": 135}], "checkpoint_step": 100}
    (replay / "manifest.json").write_text(json.dumps(manifest))
    image = b"\x89PNG\r\n\x1a\nfixture"
    (frames / "000000.png").write_bytes(image)
    outside = root / "secret.json"
    outside.write_text('{"private":true}')
    (replay / "private.txt").write_text("not an allowed asset type")
    snapshot = planning_snapshot(experiment)
    assert snapshot["replay_available"] and snapshot["replay_url"] == "/replay"
    pointer = root / "pointer.json"
    pointer.write_text(json.dumps({"windows_path": str(experiment)}))
    monitor = Monitor(pointer, root)
    html = root / "index.html"
    html.write_text("Dashboard fixture")
    (root / "replay.html").write_text("Replay viewer fixture")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(monitor, html))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:" + str(server.server_address[1])
    def missing(path):
        try:
            urllib.request.urlopen(url + path, timeout=5)
            raise AssertionError("Unsafe or absent replay asset was served: " + path)
        except urllib.error.HTTPError as error:
            assert error.code == 404
    try:
        assert urllib.request.urlopen(url + "/replay", timeout=5).read() == b"Replay viewer fixture"
        for endpoint in ("/api/replaymanifest", "/replay-assets/manifest.json"):
            with urllib.request.urlopen(url + endpoint, timeout=5) as response:
                assert response.headers.get_content_type() == "application/json"
                assert json.load(response) == manifest
        with urllib.request.urlopen(url + "/replay-assets/135/ground_truth/000000.png", timeout=5) as response:
            assert response.headers.get_content_type() == "image/png" and response.read() == image
        for path in ("../secret.json", "%2e%2e/secret.json", "%252e%252e/secret.json", "135/../manifest.json",
                     "/secret.json", "C%3a/secret.json", "%5c%5cserver/secret.json", "private.txt", "missing.png"):
            missing("/replay-assets/" + path)
        # Exercise actual symlink containment where the platform permits creating them.
        link = replay / "outside.json"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pass
        else:
            missing("/replay-assets/outside.json")
        (replay / "manifest.json").unlink()
        missing("/api/replaymanifest")
        assert not planning_snapshot(experiment)["replay_available"]
    finally:
        server.shutdown()
        server.server_close()


def self_test():
    with tempfile.TemporaryDirectory(prefix="lewm-dashboard-") as folder:
        root=Path(folder); experiment=root/"experiment"; run=experiment/"runs/sigreg"; run.mkdir(parents=True)
        test_evaluation_phases(root/"phase-fixture")
        test_delta_dashboard(root/"delta-fixture")
        test_detached_control_output(root/"control-fixture")
        test_replay_routes(root/"replay-fixture")
        (experiment/"status.json").write_text(json.dumps({"status":"running","phase":"train_sigreg"}))
        (experiment/"manifest.json").write_text(json.dumps({"training_order":{"batch_size":32,"steps_per_epoch":11427},"counts":{"train":{"episodes":5000,"windows":365657},"test":{"episodes":3000,"windows":218790}}}))
        (run/"progress.json").write_text(json.dumps({"optimizer_steps":100,"windows_seen":3200,"expected_steps":11427,"elapsed_seconds":20,"last_losses":{"pred_loss":.012}}))
        (run/"metrics.csv").write_text("step,epoch,fit/pred_loss,fit/loss,fit/sigreg_loss\n10,0,.03,.2,1.8\n20,0,.02,.18,1.7\n")
        (experiment/"control.json").write_text(json.dumps({"request":"pause"}))
        pointer=root/"pointer.json"; pointer.write_text(json.dumps({"windows_path":str(experiment),"label":"Fixture","model_names":{"sigreg":"fixture-model","action_predictor":"fixture-action"}}))
        monitor=Monitor(pointer,root); monitor.gpu=lambda:{"available":False}; monitor.refresh(); data=monitor.snapshot()
        assert data["batch_size"]==32 and data["models"][0]["progress"]["optimizer_steps"]==100
        assert data["models"][0]["metrics"]["points"][-1]["pred_loss"]==.02
        assert len(data["models"]) == 2 and len(data["evaluation_phases"]) == 4
        assert data["models"][0]["metrics"]["points"][-1]["aux_loss"] == 1.7
        assert data["models"][0]["throughput"]["windows_per_second"] is None
        fixture={"optimizer_steps":20,"windows_seen":640,"elapsed_seconds":100}
        assert monitor.rate("rate-fixture",fixture,True)["steps_per_second"] is None
        fixture.update(optimizer_steps=40,windows_seen=1280,elapsed_seconds=104)
        assert monitor.rate("rate-fixture",fixture,True)["steps_per_second"] is None
        fixture.update(optimizer_steps=60,windows_seen=1920,elapsed_seconds=108)
        assert monitor.rate("rate-fixture",fixture,True)["windows_per_second"]==160
        assert monitor.rate("rate-fixture",fixture,False)["steps_per_second"] is None
        fixture.update(optimizer_steps=80,windows_seen=2560,elapsed_seconds=200)
        assert monitor.rate("rate-fixture",fixture,True)["steps_per_second"] is None
        assert data["control"]["request"]=="pause"
        combined = root / "combined"
        combined_run = combined / "runs/combined"
        combined_run.mkdir(parents=True)
        reference = root / "reference"
        (reference / "evaluation").mkdir(parents=True)
        hashes = {"test_episode_ids": "episodes", "test_clip_indices": "clips"}
        manifest = {"manifest_sha256": "fixture-manifest", "array_hashes": hashes,
                    "training_order": {"batch_size": 32, "steps_per_epoch": 2, "seed": 3072},
                    "counts": {"train": {"windows": 64}, "test": {"windows": 50}}}
        for directory in (combined, reference):
            (directory / "manifest.json").write_text(json.dumps(manifest))
        old_run = reference / "runs/sigreg"
        old_run.mkdir(parents=True)
        trace = b"".join(i.to_bytes(8, "little") for i in range(64))
        (old_run / "training-indices.i64").write_bytes(trace)
        (old_run / "final-audit.json").write_text(json.dumps({"passed": True, "audit": {"manifest_sha256": "fixture-manifest",
            "data_provenance": {"windows": 64, "completed_steps": 2, "order_sha256": hashlib.sha256(trace).hexdigest()}}}))
        (old_run / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/sigreg_loss\n1,.4,.5,.6\n2,.3,.4,.5\n")
        progress = {"manifest_sha256": "fixture-manifest", "windows_seen": 32, "order_sha256": hashlib.sha256(trace[:256]).hexdigest()}
        (combined_run / "progress.json").write_text(json.dumps(progress))
        (reference / "evaluation/results.json").write_text(json.dumps({"passed": True, "models": {"sigreg": {
            "bf16_mixed": {"complete": True, "validation_arithmetic_mse": 0.0}}}}))
        (combined / "status.json").write_text(json.dumps({"status": "running", "phase": "train_combined"}))
        (combined_run / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/sigreg_loss,fit/action_loss\n1,.02,.11,.7,.3\n")
        combined_pointer = root / "combined-pointer.json"
        combined_pointer.write_text(json.dumps({"windows_path": str(combined), "model_names": {"combined": "fixture-combined"},
            "reference_experiment_windows_path": str(reference), "training_batch_size": 32}))
        combined_monitor = Monitor(combined_pointer, root)
        combined_monitor.gpu = lambda: {"available": False}
        single = combined_monitor.collect()
        assert single["batch_size"] == 32 and [m["key"] for m in single["models"]] == ["combined"]
        assert single["models"][0]["active"] and single["models"][0]["last_losses"] == {
            "pred_loss": .02, "loss": .11, "sigreg_loss": .7, "action_loss": .3}
        assert "aux_loss" not in single["models"][0]["metrics"]["points"][0]
        assert [(r["model"], r["precision"], r["state"]) for r in single["evaluation_phases"]] == [
            ("combined","bf16_mixed","waiting"), ("combined","fp32","waiting")]
        assert single["reference_evaluation"]["rows"][0]["mse"] == 0.0
        curve, = single["reference_training_models"]
        assert curve["key"] == "sigreg" and [p["pred_loss"] for p in curve["metrics"]["points"]] == [.4, .3]
        assert combined_monitor.collect()["reference_training_models"][0] is curve
        assert single["reference_training_token"] is None
        second = reference / "runs/action_predictor"
        second.mkdir()
        for name in ("training-indices.i64", "final-audit.json"):
            shutil.copyfile(old_run / name, second / name)
        (second / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/action_loss\n1,.3,.4,.5\n2,.2,.3,.4\n")
        full = combined_monitor.collect()
        token = full["reference_training_token"]
        assert token and len(full["reference_training_models"]) == 2
        assert combined_monitor.collect()["reference_training_token"] == token
        mse = root / "mse-experiment"
        mse_run = mse / "runs/mse_only"
        mse_run.mkdir(parents=True)
        (mse / "manifest.json").write_text(json.dumps(manifest))
        combined_pointer_data = read_json(combined_pointer)
        followups = [{"key": key, "label": LABELS[key], "windows_path": str(root / key)}
                     for key in ("combined_s06", "combined_s03", "combined_a100", "action_w50")]
        combined_pointer_data["queued_experiments"] = [{"key": "mse_only", "label": "MSE only", "windows_path": str(mse)}, *followups]
        combined_pointer.write_text(json.dumps(combined_pointer_data))
        queued = combined_monitor.collect()
        assert [m["key"] for m in queued["models"]] == ["combined"]
        assert [(row["key"], row["queue_position"]) for row in queued["queued_training_models"]] == [
            ("mse_only", 1), ("combined_s06", 2), ("combined_s03", 3), ("combined_a100", 4), ("action_w50", 5)]
        assert all(row["state"] == "queued" and row["metrics"]["points"] == [] for row in queued["queued_training_models"])
        # Promote the same fixture to live, with the combined run added as the third reference.
        for name in ("training-indices.i64", "final-audit.json"):
            shutil.copyfile(old_run / name, combined_run / name)
        (combined_run / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/sigreg_loss,fit/action_loss\n1,.02,.11,.7,.3\n2,.01,.07,.5,.2\n")
        (mse_run / "progress.json").write_text(json.dumps(progress))
        (mse_run / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/latent_cross_example_rms_std,fit/latent_cross_example_count\n1,.02,.02,0,32\n2,.01,.01,,1\n")
        (mse / "status.json").write_text(json.dumps({"status": "running", "phase": "train_mse_only"}))
        mse_pointer = root / "mse-pointer.json"
        mse_pointer.write_text(json.dumps({"windows_path": str(mse), "model_names": {"mse_only": "fixture-mse"},
            "reference_experiment_windows_path": str(reference), "comparison_experiments": [{"key": "combined", "windows_path": str(combined)}],
            "queued_experiments": followups}))
        promoted_monitor = Monitor(mse_pointer, root)
        promoted_monitor.gpu = lambda: {"available": False}
        promoted_monitor.refresh()
        promoted = promoted_monitor.snapshot()
        assert [m["key"] for m in promoted["models"]] == ["mse_only"]
        assert [(m["key"],m["queue_position"]) for m in promoted["queued_training_models"]] == [("combined_s06",1),("combined_s03",2),("combined_a100",3),("action_w50",4)]
        assert promoted["models"][0]["active"] and [m["key"] for m in promoted["reference_training_models"]] == [*PAIRED_MODELS, "combined"]
        pts = promoted["models"][0]["metrics"]["points"]
        assert pts[0]["latent_cross_example_rms_std"] == 0 and "latent_cross_example_rms_std" not in pts[1]
        assert promoted["reference_training_token"] != token and promoted_monitor.snapshot(promoted["reference_training_token"])["reference_training_omitted"]
        for filename in ("training-indices.i64", "final-audit.json"):
            shutil.copyfile(old_run / filename, mse_run / filename)
        prior = read_json(mse_pointer)["comparison_experiments"] + [{"key": "mse_only", "windows_path": str(mse)}]
        prior_token = promoted["reference_training_token"]
        for index, entry in enumerate(followups):
            key, folder = entry["key"], Path(entry["windows_path"])
            next_run = folder / "runs" / key
            next_run.mkdir(parents=True)
            (folder / "manifest.json").write_text(json.dumps(manifest))
            (folder / "status.json").write_text(json.dumps({"status": "running", "phase": "train_" + key}))
            (next_run / "progress.json").write_text(json.dumps(progress))
            for filename in ("training-indices.i64", "final-audit.json", "metrics.csv"):
                shutil.copyfile(combined_run / filename, next_run / filename)
            if key == "combined_a100":
                (next_run / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/sigreg_loss,fit/action_loss,fit/latent_cross_example_rms_std,fit/latent_cross_example_count\n1,.02,.11,.7,.3,.15,32\n2,.01,.07,.5,.2,.12,32\n")
            if key == "action_w50":
                (next_run / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/action_loss,fit/latent_cross_example_rms_std,fit/latent_cross_example_count\n1,.02,1.52,.03,.12,32\n2,.01,.51,.01,.09,32\n")
            mse_pointer.write_text(json.dumps({"windows_path": str(folder), "model_names": {key: key}, "comparison_experiments": prior,
                "reference_experiment_windows_path": str(reference), "queued_experiments": followups[index + 1:]}))
            promoted_monitor.refresh()
            stage = promoted_monitor.snapshot(prior_token)
            assert not stage["reference_training_omitted"] and stage["reference_training_token"] != prior_token
            assert stage["models"][0]["key"] == key and stage["models"][0]["active"]
            historical = {m["key"]: m for m in stage["reference_training_models"]}
            assert list(historical) == [*PAIRED_MODELS, *(p["key"] for p in prior)]
            assert historical["mse_only"]["metrics"]["points"][0]["latent_cross_example_rms_std"] == 0
            assert len(stage["queued_training_models"]) == len(followups) - index - 1
            prior_token = stage["reference_training_token"]
            assert promoted_monitor.snapshot(prior_token)["reference_training_omitted"]
            prior.append(entry)
            if key == "combined_a100":
                assert stage["models"][0]["label"] == "SIGReg 0.09 + action 1.0"
                assert stage["models"][0]["last_losses"] == {"pred_loss": .01, "loss": .07, "sigreg_loss": .5, "action_loss": .2}
        assert stage["models"][0]["label"] == "Action 50 (no SIGReg)"
        assert stage["models"][0]["last_losses"] == {"pred_loss": .01, "loss": .51, "sigreg_loss": None, "action_loss": .01}
        assert [point["latent_cross_example_rms_std"] for point in stage["models"][0]["metrics"]["points"]] == [.12, .09]
        assert [point["latent_cross_example_rms_std"] for point in historical["combined_a100"]["metrics"]["points"]] == [.15, .12]
        assert historical["mse_only"]["metrics"]["points"][0]["latent_cross_example_rms_std"] == 0
        assert list(historical) == [*PAIRED_MODELS, "combined", "mse_only", "combined_s06", "combined_s03", "combined_a100"]
        evaluation = folder / "evaluation"
        evaluation.mkdir()
        (folder / "status.json").write_text(json.dumps({"status": "running", "phase": "evaluate_next_embedding"}))
        (evaluation / "progress.json").write_text(json.dumps({"phase": "action_w50--bf16_mixed", "completed_windows": 8,
            "total_windows": 50, "validation_arithmetic_mse_so_far": .015}))
        live, waiting = promoted_monitor.collect()["evaluation_phases"]
        assert (live["model"], live["state"], live["mse"], live["completed_windows"]) == ("action_w50", "running", .015, 8)
        assert waiting["precision"] == "fp32" and waiting["state"] == "waiting"
        (evaluation / "results.json").write_text(json.dumps({"passed": True, "models": {"action_w50": {
            "bf16_mixed": {"complete": True, "windows": 50, "validation_arithmetic_mse": .012},
            "fp32": {"complete": True, "windows": 50, "validation_arithmetic_mse": .013}}}}))
        final = promoted_monitor.collect()["evaluation_phases"]
        assert [(p["state"], p["mse"]) for p in final] == [("complete", .012), ("complete", .013)]
        (second / "metrics.csv").write_text("step,fit/pred_loss,fit/loss,fit/action_loss\n1,.3,.4,.5\n2,.202,.3,.4\n")
        changed = combined_monitor.collect()
        assert changed["reference_training_token"] and changed["reference_training_token"] != token
        (combined_run / "progress.json").write_text(json.dumps({**progress, "order_sha256": "different"}))
        ineligible = combined_monitor.collect()
        assert ineligible["reference_training_models"] == [] and ineligible["reference_training_token"] is None
        (combined_run / "progress.json").write_text(json.dumps(progress))
        (reference / "manifest.json").write_text(json.dumps({**manifest, "manifest_sha256": "different",
                                                             "array_hashes": {**hashes, "test_clip_indices": "different"}}))
        unmatched = combined_monitor.collect()
        assert unmatched["reference_evaluation"] is None and unmatched["reference_training_models"] == []
        html=root/"index.html"; html.write_text("<h1>Next-embedding loss</h1>")
        server=ThreadingHTTPServer(("127.0.0.1",0),handler_for(monitor,html)); threading.Thread(target=server.serve_forever,daemon=True).start()
        url="http://127.0.0.1:"+str(server.server_address[1])
        try:
            assert json.load(urllib.request.urlopen(url+"/api/status",timeout=5))["label"]=="Fixture"
            assert "Next-embedding loss" in urllib.request.urlopen(url,timeout=5).read().decode()
            def poll(reference_token=None):
                query = urllib.parse.urlencode({"reference_token": reference_token}) if reference_token else ""
                with urllib.request.urlopen(url + "/api/status" + ("?" + query if query else ""), timeout=5) as response:
                    return json.load(response)
            with monitor.lock: monitor.payload = full
            default = poll()
            assert len(default["reference_training_models"]) == 2 and default["reference_training_omitted"] is False
            same = poll(token)
            assert same["reference_training_token"] == token and same["reference_training_omitted"] is True
            assert "reference_training_models" not in same and same["models"] == default["models"]
            with monitor.lock: monitor.payload = changed
            stale = poll(token)
            assert stale["reference_training_token"] != token and not stale["reference_training_omitted"]
            assert len(stale["reference_training_models"]) == 2
            with monitor.lock: monitor.payload = ineligible
            missing = poll(token)
            assert missing["reference_training_token"] is None and not missing["reference_training_omitted"]
            assert missing["reference_training_models"] == []
            try: urllib.request.urlopen(urllib.request.Request(url+"/api/control/pause",data=b"{}",method="POST"),timeout=5); raise AssertionError("Tokenless control accepted")
            except urllib.error.HTTPError as error: assert error.code==403
        finally: server.shutdown(); server.server_close()
    print(json.dumps({"passed":True,"checks":["shared progress and ETA, three independent loss histories, checkpoint and six precision passes","three Delta-JEPA models, bounded incremental curves and verified multi-epoch comparisons","ordered queue and all eight historical model histories through promotion","live and historical across-example latent variation","matched reference scores and full training curves",
        "evaluation phase states and MSE attribution","single-task and continuous-episode planning schemas, process identity and retained history","HTTP status/UI and opt-in reference caching","control token rejection","replay routes, MIME types, missing manifests and path/symlink containment","completed resume permits pause while detached launcher keeps output handles"]}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer",type=Path,default=POINTER); parser.add_argument("--port",type=int,default=8765); parser.add_argument("--open-browser",action="store_true"); parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test(); return
    monitor=Monitor(args.pointer); threading.Thread(target=monitor.run,daemon=True).start()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),handler_for(monitor))
    print("Training dashboard: http://127.0.0.1:"+str(args.port),flush=True)
    if args.open_browser: webbrowser.open("http://127.0.0.1:"+str(args.port))
    try: server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt: pass
    finally: monitor.stop.set(); server.server_close()
if __name__=="__main__": main()
