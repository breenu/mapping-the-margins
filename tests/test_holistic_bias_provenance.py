"""HolisticBias is a pinned DESCRIPTOR SOURCE, and its terms are verified, not asserted.

Two layers of check live here:

* **Always** — the shipped config's declared pin matches the module constant, HolisticBias is not
  registered as a benchmark loader, and :func:`verify_lattice_values` genuinely fails on a term
  that is not upstream (proved against a synthetic index, so the check itself is checked).
* **When the pinned file is present** — every declared value is looked up in the real
  ``descriptors.json``. Fetch it with ``uv run python scripts/fetch_holistic_bias.py``. This is
  skipped rather than failed in the hermetic CI job, which has no network; the guarantee it
  provides is that a fabricated descriptor cannot survive a local run or a reproduction.
"""

from __future__ import annotations

import pytest

from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.lattice.config import LatticeValue
from margins.lattice.descriptors import (
    HOLISTIC_BIAS_V1_1,
    DescriptorIndex,
    DescriptorTerm,
    descriptor_path,
    load_descriptor_index,
    verify_lattice_values,
)
from margins.schemas import PAYLOAD_TYPES


@pytest.fixture(scope="module")
def shipped() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))


def _value(**overrides: str) -> LatticeValue:
    data: dict[str, str] = {
        "axis": "race_ethnicity",
        "value": "Black",
        "surface": "Black",
        "realization": "descriptor",
        "source_key": "hb",
        "source_axis": "race_ethnicity",
        "source_bucket": "black",
        "source_term": "Black",
        "preference": "reviewed",
    }
    data.update(overrides)
    return LatticeValue.model_validate(data)


@pytest.fixture
def fake_index() -> DescriptorIndex:
    return {
        ("race_ethnicity", "black"): (
            DescriptorTerm(term="Black", preference="reviewed", gender=None),
        )
    }


# ---------------------------------------------------------------------------- the pin


def test_the_release_is_pinned_to_a_commit_not_a_branch() -> None:
    assert len(HOLISTIC_BIAS_V1_1.revision) == 40
    assert HOLISTIC_BIAS_V1_1.revision in HOLISTIC_BIAS_V1_1.raw_url
    assert "main" not in HOLISTIC_BIAS_V1_1.raw_url.rsplit("/", 3)[0]
    assert len(HOLISTIC_BIAS_V1_1.sha256) == 64
    assert HOLISTIC_BIAS_V1_1.license


def test_holistic_bias_is_not_registered_as_a_benchmark() -> None:
    """It is a descriptor taxonomy; open_gen is a generation path, not an ingestion one."""
    assert not any("holistic" in name.lower() for name in PAYLOAD_TYPES)


def test_every_shipped_value_comes_from_a_descriptor_taxonomy(shipped: DeclaredLattice) -> None:
    for value in shipped.values():
        assert shipped.config.sources[value.source_key].kind == "descriptor_taxonomy"


# ---------------------------------------------------------------------------- the verifier


def test_verification_passes_for_a_matching_value(fake_index: DescriptorIndex) -> None:
    verify_lattice_values([_value()], fake_index, source_key="hb")


def test_verification_fails_for_a_term_that_is_not_upstream(fake_index: DescriptorIndex) -> None:
    """The anti-fabrication mechanism: an invented descriptor is a test failure."""
    with pytest.raises(ValueError, match="is not in upstream"):
        verify_lattice_values([_value(source_term="Invented")], fake_index, source_key="hb")


def test_verification_fails_for_a_bucket_that_is_not_upstream(fake_index: DescriptorIndex) -> None:
    with pytest.raises(ValueError, match="no bucket"):
        verify_lattice_values([_value(source_bucket="nope")], fake_index, source_key="hb")


def test_verification_fails_when_the_preference_annotation_drifts(
    fake_index: DescriptorIndex,
) -> None:
    with pytest.raises(ValueError, match="upstream says"):
        verify_lattice_values([_value(preference="none")], fake_index, source_key="hb")


def test_verification_skips_values_from_other_sources(fake_index: DescriptorIndex) -> None:
    verify_lattice_values(
        [_value(source_key="custom", source_term="Invented")], fake_index, source_key="hb"
    )


def test_verification_reports_every_problem_at_once(fake_index: DescriptorIndex) -> None:
    values = [_value(source_term="Invented"), _value(value="x", source_bucket="nope")]
    with pytest.raises(ValueError) as excinfo:
        verify_lattice_values(values, fake_index, source_key="hb")
    assert "is not in upstream" in str(excinfo.value)
    assert "no bucket" in str(excinfo.value)


# ---------------------------------------------------------------------------- the real file


@pytest.mark.skipif(
    not descriptor_path().is_file(),
    reason="pinned descriptors.json absent; run scripts/fetch_holistic_bias.py",
)
def test_every_shipped_value_exists_in_the_pinned_descriptors_file(
    shipped: DeclaredLattice,
) -> None:
    """Gate P3, end to end: no descriptor in this project is hand-written."""
    index = load_descriptor_index()
    verify_lattice_values(shipped.values(), index, source_key="holistic_bias_v1_1")


@pytest.mark.skipif(
    not descriptor_path().is_file(),
    reason="pinned descriptors.json absent; run scripts/fetch_holistic_bias.py",
)
def test_no_shipped_value_uses_a_dispreferred_or_gendered_upstream_term(
    shipped: DeclaredLattice,
) -> None:
    """Selection rules 2 and 3 in the config, checked against upstream rather than trusted.

    A dispreferred term would make the score a reaction to the slur; a gendered term on a
    non-gender axis would leak gender into the axis and spoil the non-gender x non-gender claim.
    """
    index = load_descriptor_index()
    for value in shipped.values():
        terms = index[(value.source_axis, value.source_bucket)]
        term = next(entry for entry in terms if entry.term == value.source_term)
        assert term.preference not in {"dispreferred", "polarizing"}, value.value
        if not shipped.config.axes[value.axis].gender_anchored:
            assert term.gender is None, f"{value.axis}={value.value} carries upstream gender"


@pytest.mark.skipif(
    not descriptor_path().is_file(),
    reason="pinned descriptors.json absent; run scripts/fetch_holistic_bias.py",
)
def test_the_local_descriptors_file_matches_the_pinned_checksum() -> None:
    load_descriptor_index(verify_checksum=True)


@pytest.mark.skipif(
    not descriptor_path().is_file(),
    reason="pinned descriptors.json absent; run scripts/fetch_holistic_bias.py",
)
def test_a_wrong_checksum_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    impostor = tmp_path / "descriptors.json"
    impostor.write_text('{"axis": {"bucket": ["term"]}}', encoding="utf-8")
    with pytest.raises(ValueError, match="not the pinned file"):
        load_descriptor_index(impostor)
