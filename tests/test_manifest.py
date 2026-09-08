"""The run manifest is the project's only experiment-tracking mechanism, so it is tested
as a contract: required fields present, deterministic id, and no drift between the object
returned and the bytes on disk."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from margins.config import (
    NOT_INSTALLED,
    ExperimentConfig,
    config_from_mapping,
    make_run_id,
    manifest_path,
    write_run_manifest,
)

# PROJECT_CONTEXT.md section 4. A field silently dropped here breaks reproducibility quietly,
# which is the worst way for it to break.
REQUIRED_FIELDS = {
    "run_id",
    "created_at",
    "config_hash",
    "config",
    "git_sha",
    "git_dirty",
    "package_versions",
    "python_version",
    "platform",
    "seed",
    "environment",
    "determinism_note",
    "input_checksums",
    "quant",
}

FIXED_TIME = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)


def _cfg(tmp_path: Path, **overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {
        "seed": 20260908,
        "run_name": "manifest-test",
        "results_dir": str(tmp_path / "results"),
        "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "outputs"),
    }
    data.update(overrides)
    return config_from_mapping(data)


def test_returned_manifest_equals_the_file_on_disk(tmp_path: Path) -> None:
    """Kills tuple->list / Path->str JSON drift: the caller sees exactly what is written."""
    cfg = _cfg(tmp_path)
    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME)

    path = manifest_path(cfg, manifest.run_id)
    assert path.is_file()

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert manifest.model_dump(mode="json") == on_disk


def test_manifest_carries_every_required_field(tmp_path: Path) -> None:
    manifest = write_run_manifest(_cfg(tmp_path), timestamp=FIXED_TIME)
    assert set(manifest.model_dump(mode="json")) == REQUIRED_FIELDS


def test_run_id_is_deterministic_given_config_and_timestamp(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    first = write_run_manifest(cfg, timestamp=FIXED_TIME)
    second = write_run_manifest(cfg, timestamp=FIXED_TIME)
    assert first.run_id == second.run_id
    assert first.config_hash == second.config_hash


def test_run_id_differs_for_a_different_config(tmp_path: Path) -> None:
    a = write_run_manifest(_cfg(tmp_path), timestamp=FIXED_TIME)
    b = write_run_manifest(_cfg(tmp_path, seed=1), timestamp=FIXED_TIME)
    assert a.run_id != b.run_id
    assert a.config_hash != b.config_hash


def test_run_id_differs_for_a_different_timestamp(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    a = write_run_manifest(cfg, timestamp=FIXED_TIME)
    b = write_run_manifest(cfg, timestamp=datetime(2026, 9, 9, 12, 0, 0, tzinfo=UTC))
    assert a.run_id != b.run_id


def test_run_id_is_filesystem_safe(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, run_name="a run/with:bad chars")
    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME)
    assert manifest_path(cfg, manifest.run_id).is_file()
    assert not set(manifest.run_id) & set('/\\:*?"<>| ')


def test_make_run_id_consults_no_clock() -> None:
    cfg = config_from_mapping({"seed": 1, "run_name": "x"})
    assert make_run_id(cfg, "deadbeefcafe0000", FIXED_TIME) == make_run_id(
        cfg, "deadbeefcafe0000", FIXED_TIME
    )


def test_torch_recorded_as_not_installed_in_core_env(tmp_path: Path) -> None:
    manifest = write_run_manifest(_cfg(tmp_path), timestamp=FIXED_TIME)
    assert "torch" in manifest.package_versions
    if importlib.util.find_spec("torch") is None:
        assert manifest.package_versions["torch"] == NOT_INSTALLED
        assert manifest.environment.torch_version is None
    assert manifest.package_versions["pydantic"] != NOT_INSTALLED


def test_git_fields_present_and_do_not_crash(tmp_path: Path) -> None:
    """Inside a repo the dirty flag is known; outside one both fields are None, no exception.

    ``git_sha`` may legitimately be None *inside* a repo whose HEAD is unborn (initialised but
    with no commit yet), which is why the two fields are not asserted to agree.
    """
    inside = write_run_manifest(_cfg(tmp_path), timestamp=FIXED_TIME)
    assert inside.git_dirty is not None
    assert inside.git_sha is None or len(inside.git_sha) == 40

    outside = write_run_manifest(_cfg(tmp_path), timestamp=FIXED_TIME, repo_root=tmp_path)
    assert outside.git_sha is None
    assert outside.git_dirty is None


def test_determinism_note_and_quant_constant_travel_with_the_run(tmp_path: Path) -> None:
    manifest = write_run_manifest(_cfg(tmp_path, quant="none"), timestamp=FIXED_TIME)
    assert manifest.quant == "none"
    assert manifest.quant == manifest.config.quant
    assert "No unconditional bit-identity is claimed." in manifest.determinism_note


def test_input_checksums_recorded(tmp_path: Path) -> None:
    checksums = {"crows_pairs.csv": "a" * 64}
    manifest = write_run_manifest(_cfg(tmp_path), inputs=checksums, timestamp=FIXED_TIME)
    assert manifest.input_checksums == checksums

    empty = write_run_manifest(_cfg(tmp_path), timestamp=FIXED_TIME)
    assert empty.input_checksums == {}


def test_refuses_to_return_a_manifest_it_did_not_write(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, manifest={"write_manifest": False})
    with pytest.raises(ValueError, match="write_manifest=False"):
        write_run_manifest(cfg, timestamp=FIXED_TIME)


def test_manifest_is_frozen(tmp_path: Path) -> None:
    manifest = write_run_manifest(_cfg(tmp_path), timestamp=FIXED_TIME)
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        manifest.seed = 0  # type: ignore[misc]
