"""Clean-install built artifacts and exercise base import and CLI behavior."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, timeout=300)  # noqa: S603


def main() -> int:
    """Install the wheel and sdist independently and smoke-test each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = (
        arguments.directory
        if arguments.directory.is_absolute()
        else root / arguments.directory
    )
    wheels = sorted(directory.glob("temporalfix-*.whl"))
    sources = sorted(directory.glob("temporalfix-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        message = (
            "expected exactly one temporalfix wheel and sdist, found "
            f"{len(wheels)} wheel(s) and {len(sources)} sdist(s)"
        )
        raise RuntimeError(message)

    for label, artifact in (("wheel", wheels[0]), ("sdist", sources[0])):
        with tempfile.TemporaryDirectory(prefix=f"temporalfix-{label}-") as raw:
            environment = Path(raw) / "venv"
            _run([sys.executable, "-m", "venv", str(environment)], cwd=root)
            executable = (
                environment
                / ("Scripts" if sys.platform == "win32" else "bin")
                / "python"
            )
            _run(
                [
                    str(executable),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    str(artifact),
                ],
                cwd=root,
            )
            probe = (
                "import sys, temporalfix, temporalfix.adapters; "
                "print(temporalfix.__version__); "
                "assert 'ultralytics' not in sys.modules; "
                "assert 'supervision' not in sys.modules"
            )
            _run([str(executable), "-c", probe], cwd=root)
            _run([str(executable), "-m", "temporalfix.cli", "version"], cwd=root)
            _run([str(executable), "-m", "temporalfix.cli", "--help"], cwd=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
