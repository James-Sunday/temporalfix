from __future__ import annotations

from pathlib import Path

import pytest

from temporalfix.dataset_manifest import build_mot17_manifest


def mot17_tree(root: Path) -> Path:
    sequence = root / "train" / "MOT17-02-DPM"
    (sequence / "det").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text("[Sequence]\nname=MOT17-02\n")
    (sequence / "det" / "det.txt").write_text("1,-1,0,0,10,10,0.9,-1,-1,-1\n")
    return root


def test_mot17_manifest_is_relative_and_content_addressed(tmp_path: Path) -> None:
    manifest = build_mot17_manifest(
        mot17_tree(tmp_path / "MOT17"),
        accepted_license="CC-BY-NC-SA-3.0",
    )

    assert manifest["dataset"] == "MOT17"
    assert manifest["file_count"] == 2
    assert len(manifest["manifest_sha256"]) == 64
    assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])
    assert str(tmp_path) not in str(manifest)


def test_mot17_manifest_requires_exact_license_acceptance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"CC-BY-NC-SA-3\.0"):
        build_mot17_manifest(
            mot17_tree(tmp_path / "MOT17"),
            accepted_license="not-accepted",
        )


def test_mot17_manifest_rejects_unrecognized_layout(tmp_path: Path) -> None:
    tmp_path.joinpath("file.txt").write_text("not MOT17")
    with pytest.raises(ValueError, match=r"seqinfo\.ini"):
        build_mot17_manifest(
            tmp_path,
            accepted_license="CC-BY-NC-SA-3.0",
        )
