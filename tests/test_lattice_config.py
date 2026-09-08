"""Lattice config validation, and gate P3's named-pairs clause as an executable check.

The gate is not a review item here: a lattice config with fewer than two non-gender x non-gender
confirmatory pairs does not load. These tests pin that, plus the surrounding rules that keep the
FDR family bounded and every value traceable.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from margins.lattice import DEFAULT_LATTICE, LatticeConfig, load_lattice_config
from margins.lattice.config import (
    MAX_CONFIRMATORY_PAIRS,
    MIN_NON_GENDER_CONFIRMATORY_PAIRS,
    lattice_config_from_mapping,
)
from margins.lattice.descriptors import HOLISTIC_BIAS_V1_1

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "lattice"

#: Values per axis in the confirmatory config. The hard schema bounds are wider (2-6) to leave
#: room for fixtures; the shipped config is held to the tighter aim.
SHIPPED_MIN_VALUES = 3
SHIPPED_MAX_VALUES = 5


@pytest.fixture(scope="module")
def shipped() -> LatticeConfig:
    return load_lattice_config(DEFAULT_LATTICE)


@pytest.fixture(scope="module")
def tiny() -> LatticeConfig:
    return load_lattice_config("tiny", config_dir=FIXTURE_DIR)


def _base_config() -> dict[str, Any]:
    """A minimal config that validates. Tests mutate a deepcopy of it to break one rule."""
    return {
        "lattice_version": "unit@test",
        "description": "unit test lattice",
        "sources": {
            "src": {
                "kind": "descriptor_taxonomy",
                "source": "unit test source",
                "revision": "0" * 40,
                "version": "v0",
                "license": "MIT",
                "sha256": "0" * 64,
                "retrieved_at": "2026-09-08T00:00:00Z",
            }
        },
        "axes": {
            "aa": {
                "label": "aa",
                "gender_anchored": False,
                "compose_rank": 10,
                "source": "src",
                "source_axis": "aa_up",
                "values": [
                    {"source_term": "a1", "source_bucket": "b"},
                    {"source_term": "a2", "source_bucket": "b"},
                ],
            },
            "bb": {
                "label": "bb",
                "gender_anchored": False,
                "compose_rank": 20,
                "source": "src",
                "source_axis": "bb_up",
                "values": [
                    {"source_term": "b1", "source_bucket": "b"},
                    {"source_term": "b2", "source_bucket": "b"},
                ],
            },
            "cc": {
                "label": "cc",
                "gender_anchored": False,
                "compose_rank": 30,
                "source": "src",
                "source_axis": "cc_up",
                "values": [
                    {"source_term": "c1", "source_bucket": "b"},
                    {"source_term": "c2", "source_bucket": "b"},
                ],
            },
        },
        "pairs": [
            {"axes": ["aa", "bb"], "status": "confirmatory", "rationale": "r1"},
            {"axes": ["bb", "cc"], "status": "confirmatory", "rationale": "r2"},
        ],
        "power": {
            "min_cell_n": 10,
            "renderings_per_cell": {"order_1": 12, "order_2": 12},
            "pair_overrides": [],
        },
    }


def _build(**mutations: Any) -> LatticeConfig:
    data = deepcopy(_base_config())
    data.update(mutations)
    return lattice_config_from_mapping(data)


# ---------------------------------------------------------------------------- the base case


def test_base_config_is_valid() -> None:
    assert _build().lattice_version == "unit@test"


# ---------------------------------------------------------------------------- gate P3


def test_shipped_config_names_at_least_two_non_gender_confirmatory_pairs(
    shipped: LatticeConfig,
) -> None:
    """GATE P3: the primary intersections must reach beyond gender-anchored cells."""
    non_gender = shipped.non_gender_confirmatory_pairs
    assert len(non_gender) >= MIN_NON_GENDER_CONFIRMATORY_PAIRS, (
        "novelty claim 5 requires non-gender x non-gender coverage; this config has "
        f"{len(non_gender)}"
    )
    for pair in non_gender:
        for axis in pair:
            assert not shipped.axes[axis].gender_anchored


def test_a_lattice_without_two_non_gender_pairs_does_not_load() -> None:
    """The gate is structural, not a convention: such a config must not validate."""
    data = deepcopy(_base_config())
    data["axes"]["cc"]["gender_anchored"] = True
    with pytest.raises(ValidationError, match="non-gender-anchored"):
        lattice_config_from_mapping(data)


def test_shipped_confirmatory_family_is_pre_registered_size(shipped: LatticeConfig) -> None:
    """Section 8 pre-registers 6-8 confirmatory pairs; that set IS the FDR family."""
    assert 6 <= len(shipped.confirmatory_pairs) <= MAX_CONFIRMATORY_PAIRS


def test_confirmatory_family_is_capped() -> None:
    data = deepcopy(_base_config())
    for index in range(MAX_CONFIRMATORY_PAIRS + 1):
        name = f"ax{index}"
        data["axes"][name] = {
            "label": name,
            "gender_anchored": False,
            "compose_rank": 100 + index,
            "source": "src",
            "source_axis": f"{name}_up",
            "values": [
                {"source_term": f"{name}1", "source_bucket": "b"},
                {"source_term": f"{name}2", "source_bucket": "b"},
            ],
        }
        data["pairs"].append(
            {"axes": ["aa", name], "status": "confirmatory", "rationale": "too many"}
        )
    with pytest.raises(ValidationError, match="exceeds the cap"):
        lattice_config_from_mapping(data)


# ---------------------------------------------------------------------------- provenance


def test_every_shipped_value_has_non_empty_provenance(shipped: LatticeConfig) -> None:
    """Gate P3's first clause: no value without a source."""
    values = shipped.lattice_values()
    assert values
    for value in values:
        assert value.source_key in shipped.sources
        assert value.source_axis.strip()
        assert value.source_bucket.strip()
        assert value.source_term.strip()
        assert value.preference.strip()

        provenance = shipped.provenance_for(value)
        assert provenance.source.strip()
        assert provenance.license.strip()
        assert provenance.sha256.strip()
        assert provenance.revision or provenance.version


def test_every_tiny_value_has_non_empty_provenance(tiny: LatticeConfig) -> None:
    for value in tiny.lattice_values():
        assert value.source_key in tiny.sources
        assert tiny.provenance_for(value).source.strip()


def test_shipped_source_matches_the_pinned_release(shipped: LatticeConfig) -> None:
    source = shipped.sources["holistic_bias_v1_1"]
    assert source.revision == HOLISTIC_BIAS_V1_1.revision
    assert source.version == HOLISTIC_BIAS_V1_1.version
    assert source.sha256 == HOLISTIC_BIAS_V1_1.sha256
    assert source.license == HOLISTIC_BIAS_V1_1.license
    assert source.path == HOLISTIC_BIAS_V1_1.repo_path


def test_shipped_config_uses_no_name_proxied_values(shipped: LatticeConfig) -> None:
    """Explicit descriptors only. Name-proxying is permitted by the schema but not used here."""
    assert all(value.realization == "descriptor" for value in shipped.lattice_values())


def test_a_value_referencing_an_undeclared_source_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["axes"]["aa"]["values"][0]["source"] = "nope"
    with pytest.raises(ValidationError, match="not declared"):
        lattice_config_from_mapping(data)


def test_a_custom_value_without_a_justification_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["sources"]["src"]["kind"] = "custom"
    with pytest.raises(ValidationError, match="must carry a `note`"):
        lattice_config_from_mapping(data)


def test_a_custom_value_with_a_justification_is_accepted() -> None:
    data = deepcopy(_base_config())
    data["sources"]["src"]["kind"] = "custom"
    for axis in data["axes"].values():
        for value in axis["values"]:
            value["note"] = "justified for the unit test"
    assert lattice_config_from_mapping(data).sources["src"].kind == "custom"


def test_a_name_realized_value_without_a_confound_note_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["axes"]["aa"]["values"][0]["realization"] = "name"
    with pytest.raises(ValidationError, match="name-realized"):
        lattice_config_from_mapping(data)


def test_a_name_realized_value_with_a_confound_note_is_accepted() -> None:
    data = deepcopy(_base_config())
    data["axes"]["aa"]["values"][0].update(
        {
            "realization": "name",
            "surface": "Jamal",
            "note": "name-proxied; confounded with name frequency, orthography and nationality",
        }
    )
    config = lattice_config_from_mapping(data)
    assert config.axes["aa"].lattice_values("aa")[0].surface == "Jamal"


# ---------------------------------------------------------------------------- power for cells


def test_shipped_values_per_axis_stay_small(shipped: LatticeConfig) -> None:
    """Breadth here is power spent: cell count grows multiplicatively."""
    for axis_name, axis in shipped.axes.items():
        assert SHIPPED_MIN_VALUES <= len(axis.values) <= SHIPPED_MAX_VALUES, axis_name


def test_too_many_values_on_an_axis_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["axes"]["aa"]["values"] = [
        {"source_term": f"a{index}", "source_bucket": "b"} for index in range(7)
    ]
    with pytest.raises(ValidationError, match="values"):
        lattice_config_from_mapping(data)


def test_duplicate_values_on_an_axis_are_rejected() -> None:
    data = deepcopy(_base_config())
    data["axes"]["aa"]["values"] = [
        {"source_term": "a1", "source_bucket": "b"},
        {"source_term": "a1", "source_bucket": "other"},
    ]
    with pytest.raises(ValidationError, match="unique"):
        lattice_config_from_mapping(data)


def test_min_cell_n_is_documented_as_a_placeholder(shipped: LatticeConfig) -> None:
    """TODO(P9) derives the real threshold; until then it is a declared convention."""
    description = type(shipped.power).model_fields["min_cell_n"].description or ""
    assert "PLACEHOLDER" in description
    assert "TODO(P9)" in description
    assert shipped.power.min_cell_n > 0


# ---------------------------------------------------------------------------- pairs


def test_a_pair_naming_an_undeclared_axis_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["pairs"].append({"axes": ["aa", "zz"], "status": "exploratory", "rationale": "r"})
    with pytest.raises(ValidationError, match="undeclared axis"):
        lattice_config_from_mapping(data)


def test_a_pair_of_one_axis_with_itself_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["pairs"].append({"axes": ["aa", "aa"], "status": "exploratory", "rationale": "r"})
    with pytest.raises(ValidationError, match="two distinct axes"):
        lattice_config_from_mapping(data)


def test_a_pair_declared_twice_in_either_order_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["pairs"].append({"axes": ["bb", "aa"], "status": "exploratory", "rationale": "dup"})
    with pytest.raises(ValidationError, match="declared more than once"):
        lattice_config_from_mapping(data)


def test_pair_order_in_yaml_does_not_reach_the_output() -> None:
    """Canonicalisation by compose_rank, so a human's typing order never changes an artifact."""
    data = deepcopy(_base_config())
    data["pairs"] = [
        {"axes": ["cc", "bb"], "status": "confirmatory", "rationale": "r2"},
        {"axes": ["bb", "aa"], "status": "confirmatory", "rationale": "r1"},
    ]
    assert lattice_config_from_mapping(data).confirmatory_pairs == (("aa", "bb"), ("bb", "cc"))


def test_duplicate_compose_rank_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["axes"]["bb"]["compose_rank"] = 10
    with pytest.raises(ValidationError, match="compose_rank must be unique"):
        lattice_config_from_mapping(data)


def test_an_override_for_an_undeclared_pair_is_rejected() -> None:
    data = deepcopy(_base_config())
    data["power"]["pair_overrides"] = [
        {"axes": ["aa", "cc"], "renderings_per_cell": 4, "reason": "not a declared pair"}
    ]
    with pytest.raises(ValidationError, match="not a declared axis pair"):
        lattice_config_from_mapping(data)


def test_every_shipped_pair_carries_a_rationale(shipped: LatticeConfig) -> None:
    """Pre-registration text, not a comment."""
    for pair in shipped.pairs:
        assert len(pair.rationale.strip()) > 20


def test_exploratory_pairs_are_disjoint_from_the_family(shipped: LatticeConfig) -> None:
    assert not set(shipped.exploratory_pairs) & set(shipped.confirmatory_pairs)


def test_config_is_frozen(shipped: LatticeConfig) -> None:
    with pytest.raises(ValidationError):
        shipped.lattice_version = "mutated"  # type: ignore[misc]
