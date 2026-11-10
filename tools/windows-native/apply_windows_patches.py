"""Apply the unsupported-POSIX-signals fix to stable-pretraining 0.1.8 in the selected venv."""
import argparse
from importlib import metadata
import os
from pathlib import Path
import sysconfig

ORIGINAL = "    for sig in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGCONT, signal.SIGTERM):"
PATCHED = '''    # Windows lacks the POSIX user/continue signals; log supported signals only.
    for sig in (getattr(signal, name) for name in
                ("SIGUSR1", "SIGUSR2", "SIGCONT", "SIGTERM") if hasattr(signal, name)):'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check whether the fix is already applied")
    check_only = parser.parse_args().check
    if os.name != "nt":
        raise RuntimeError("This patch applies to the native Windows environment")
    version = metadata.version("stable-pretraining")
    if version != "0.1.8":
        raise RuntimeError(f"Unexpected stable-pretraining version {version}")
    file = Path(sysconfig.get_path("purelib")) / "stable_pretraining" / "manager.py"
    source = file.read_text(encoding="utf-8")
    if PATCHED in source:
        print(f"Windows signal patch present in {file}")
        return
    if source.count(ORIGINAL) != 1:
        raise RuntimeError(f"Unrecognized stable-pretraining source in {file}")
    if check_only:
        raise RuntimeError("Windows signal patch has not been applied")
    file.write_text(source.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print(f"Applied Windows signal patch in {file}")


if __name__ == "__main__":
    main()
