"""Wait for a queued experiment's predecessor and hand off the local dashboard."""
import argparse
from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ACTIVE = Path(os.environ.get("LEWM_DESKTOP_ROOT", Path.home() / "Desktop/AI")) / "lewm-wsl/active-experiment.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def registered(pointer, target):
    return any(row.get("linux_experiment") == target
               for row in pointer.get("queued_experiments", []))


def disposition(pointer, queue, prior_state, prior_control, next_state, next_control):
    target = queue["target"]["linux_experiment"]
    prior = queue["predecessor"]["linux_experiment"]
    active = pointer.get("linux_experiment")
    promoted = active == target
    if not promoted and not registered(pointer, target):
        return "superseded"
    if active not in (prior, target):
        waiting = [row.get("linux_experiment") for row in pointer.get("queued_experiments", [])]
        if prior in waiting and waiting.index(prior) < waiting.index(target):
            return "waiting_for_turn"
        return "superseded"
    if next_state != "queued":
        return "already_started"
    if prior_control != "run" or next_control != "run" or prior_state == "paused":
        return "waiting_for_resume"
    if prior_state == "failed":
        return "waiting_for_predecessor_recovery"
    if prior_state != "complete":
        return "waiting"
    return "start" if promoted else "promote"


def execute(queue_file, register):
    queue = read_json(queue_file)
    target = Path(queue["target"]["windows_path"])
    predecessor = Path(queue["predecessor"]["windows_path"])
    previous_notice = None

    def announce(state, **extra):
        nonlocal previous_notice
        if state != previous_notice or extra:
            data = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                        pid=os.getpid(), **extra)
            write_json(queue_file.with_name("queue-status.json"), data)
            print(json.dumps(data), flush=True)
            previous_notice = state

    with queue_file.with_name("queue.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("A queue worker already holds the lock.", flush=True)
            return 0

        pointer = read_json(ACTIVE)
        prior = queue["predecessor"]["linux_experiment"]
        if register and (pointer.get("linux_experiment") == prior or registered(pointer, prior)):
            if not registered(pointer, queue["target"]["linux_experiment"]):
                pointer["queued_experiments"] = [*pointer.get("queued_experiments", []), queue["target"]]
                write_json(ACTIVE, pointer)
        while True:
            try:
                pointer = read_json(ACTIVE)
                decision = disposition(pointer, queue, read_json(predecessor / "status.json")["status"],
                                       read_json(predecessor / "control.json")["request"],
                                       read_json(target / "status.json")["status"],
                                       read_json(target / "control.json")["request"])
            except (OSError, ValueError, KeyError) as error:
                announce("waiting_for_state")
                print(f"State temporarily unavailable: {error}", flush=True)
                time.sleep(10)
                continue
            if decision in ("promote", "start") and not (target / "source-snapshot.json").is_file():
                announce("waiting_for_source")
                time.sleep(5)
                continue
            if decision == "promote":
                next_pointer = dict(queue["pointer"])
                remaining = [entry for entry in pointer.get("queued_experiments", [])
                             if entry.get("linux_experiment") != queue["target"]["linux_experiment"]]
                if remaining:
                    next_pointer["queued_experiments"] = remaining
                write_json(ACTIVE, next_pointer)
                announce("promoted")
                # Re-read statuses and pause controls with the target pointer active.
                continue
            if decision == "start":
                announce("launching")
                command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(ROOT / "Run-LeWM-Experiment.ps1"),
                           "-RunScript", queue["pointer"]["linux_run_script"]]
                with queue_file.with_name("queue-pipeline.log").open("a", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW)
                status = read_json(target / "status.json")
                announce("dispatched", pipeline_state=status.get("status"), exit_code=result.returncode)
                return result.returncode
            announce(decision)
            if decision in {"already_started", "superseded"}:
                return 0
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--register", action="store_true", help="Add the new run to its predecessor's queue.")
    args = parser.parse_args()
    sys.exit(execute(args.queue, args.register))
