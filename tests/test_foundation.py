from __future__ import annotations

import pytest

import temporalfix
from temporalfix.cli import main


def test_version_is_stable() -> None:
    assert temporalfix.__version__ == "0.1.0"


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "temporalfix 0.1.0\n"


def test_help_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    captured = capsys.readouterr()
    assert "Repair and stabilize" in captured.out
