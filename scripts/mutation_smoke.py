"""Apply a critical association mutation and require its test to kill it."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    """Mutate the assignment cost in a temporary import tree."""
    root = Path(__file__).resolve().parents[1]
    original = "cost = np.where(valid, 1.0 - iou, impossible)"
    replacement = "cost = np.where(valid, 1.0 + iou, impossible)"
    with tempfile.TemporaryDirectory(prefix="temporalfix-mutation-") as raw:
        import_root = Path(raw)
        destination = import_root / "temporalfix"
        shutil.copytree(root / "src" / "temporalfix", destination)
        target = destination / "association.py"
        contents = target.read_text(encoding="utf-8")
        if contents.count(original) != 1:
            raise RuntimeError("association mutation anchor was not found exactly once")
        target.write_text(contents.replace(original, replacement, 1), encoding="utf-8")
        environment = os.environ.copy()
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(import_root)
            if not previous
            else str(import_root) + os.pathsep + previous
        )
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_association.py", "-q"],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode == 0:
            raise RuntimeError("critical association mutation survived")
    print("association mutation killed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
