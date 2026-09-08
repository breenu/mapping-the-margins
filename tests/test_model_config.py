"""The pinned checkpoints (P5 -- open decision #2, closed 9 Sep 2026).

These tests are deliberately brittle about revisions. A pin that can drift silently is not a pin,
and the whole reproducibility claim rests on a ``Response`` naming the exact commit its weights came
from. Changing a value here must be a conscious edit accompanied by a logged re-pin, exactly as
``tests/test_schema_field_freeze.py`` is for the record contract.

No network: ``scripts/pin_models.py`` is what talks to Hugging Face, and it is deliberately not part
of the suite (tests must not depend on the network, and the core CI job has none).
"""

from __future__ import annotations

from typing import Any

import pytest

from margins.inference.models import (
    CI_MODEL,
    CONFIRMATORY_MODELS,
    DEV_MODEL,
    ModelSpec,
    default_model_config_dir,
    load_model_config,
    model_config_from_mapping,
)

DECLARED_MODELS = (*CONFIRMATORY_MODELS, DEV_MODEL, CI_MODEL)

#: The pins as resolved on 9 Sep 2026. Every sha came from huggingface_hub, never from a keyboard.
EXPECTED_PINS: dict[str, tuple[str, str, str]] = {
    "llama31_8b_instruct": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
        "llama3.1",
    ),
    "mistral7b_instruct_v03": (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "apache-2.0",
    ),
    "llama32_1b_instruct": (
        "meta-llama/Llama-3.2-1B-Instruct",
        "9213176726f574b556790deb65791e0c5aa438b6",
        "llama3.2",
    ),
    "tiny_random_llama": (
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
        "unknown",
    ),
}


def _spec(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test-model",
        "model_id": "owner/repo",
        "revision": "0" * 40,
        "license": "MIT",
        "role": "dev",
        "family": "llama",
        "gated": False,
        "note": "a test spec",
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize("name", DECLARED_MODELS)
def test_pins_are_exactly_what_was_resolved(name: str) -> None:
    spec = load_model_config(name)
    model_id, revision, license_ = EXPECTED_PINS[name]
    assert (spec.model_id, spec.revision, spec.license) == (model_id, revision, license_), (
        f"the pin for {name!r} changed. That is a re-pin, not a refactor: any result produced "
        "under the old revision was produced under a DIFFERENT model. Log the rationale."
    )


def test_the_confirmatory_pair_is_two_distinct_families() -> None:
    """RQ-relevant: 'one model family' was the locked scope; the pair spans two on purpose."""
    families = {load_model_config(name).family for name in CONFIRMATORY_MODELS}
    assert len(families) == len(CONFIRMATORY_MODELS), f"pair collapses to one family: {families}"


def test_roles_are_what_the_build_expects() -> None:
    assert {load_model_config(n).role for n in CONFIRMATORY_MODELS} == {"confirmatory"}
    assert load_model_config(DEV_MODEL).role == "dev"
    assert load_model_config(CI_MODEL).role == "ci"


def test_every_config_file_on_disk_is_declared() -> None:
    """A model config nobody declares is a checkpoint that could enter a run undocumented."""
    on_disk = {path.stem for path in default_model_config_dir().glob("*.yaml")}
    assert on_disk == set(DECLARED_MODELS)


def test_gating_is_recorded_for_the_repos_that_have_it() -> None:
    """A reproducer needs to know up front that they must accept a licence themselves."""
    assert load_model_config("llama31_8b_instruct").gated is True
    assert load_model_config("llama32_1b_instruct").gated is True
    assert load_model_config("mistral7b_instruct_v03").gated is False


# --------------------------------------------------------------------------------------
# The validators
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("revision", ["main", "v1.0", "0" * 39, "0" * 41, "G" * 40, "A" * 40])
def test_a_moving_or_malformed_revision_is_rejected(revision: str) -> None:
    """'main' is the failure this validator exists for: it moves, and nothing we record changes."""
    with pytest.raises(ValueError, match="40-character lowercase hex"):
        model_config_from_mapping(_spec(revision=revision))


def test_a_malformed_repo_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="owner/repo"):
        model_config_from_mapping(_spec(model_id="just-a-name"))


def test_license_is_required() -> None:
    """An unrecorded licence blocks the Zenodo release; 'unknown' must be explicit."""
    data = _spec()
    del data["license"]
    with pytest.raises(ValueError, match="license"):
        model_config_from_mapping(data)


def test_spec_is_frozen_and_forbids_extras() -> None:
    assert ModelSpec.model_config.get("frozen") is True
    assert ModelSpec.model_config.get("extra") == "forbid"


def test_a_config_cannot_disagree_with_its_filename() -> None:
    with pytest.raises(ValueError, match="the file name is the authority"):
        load_model_config("llama32_1b_instruct", config_dir=_conflicting_dir())


def _conflicting_dir() -> Any:
    """A config dir whose file declares a name other than its filename."""
    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp())
    (directory / "llama32_1b_instruct.yaml").write_text(
        "name: something_else\nmodel_id: owner/repo\nrevision: "
        + "0" * 40
        + "\nlicense: MIT\nrole: dev\nfamily: llama\ngated: false\nnote: n\n",
        encoding="utf-8",
    )
    return directory
