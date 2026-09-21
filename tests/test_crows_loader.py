"""The CrowS-Pairs loader, its pin, and the screen (gate P2).

Gate P2 reads: *"Row counts match published dataset; checksums recorded; 5-item eyeball; every
exclusion logged with a reason; screening is deterministic."* Each clause has a test here, and the
eyeball is done twice -- once against a crafted fixture whose rows were chosen to hit specific
branches, and once against the real pinned file, so the fixture cannot drift into agreeing with
itself while disagreeing with the data.

The sharpest test in the file is :func:`test_an_antistereo_row_swaps_the_two_sentences`. 218 of the
1,508 upstream rows put the ANTI-stereotypical sentence in ``sent_more``; reading that column as
"the stereotyping one" would invert the sign of every CrowS effect while producing entirely
plausible output, which is the exact failure mode the schema's sign convention warns about.
"""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from pathlib import Path

import pytest

from margins.benchmarks import (
    CROWS_PAIRS_RELEASE,
    DEFAULT_BENCHMARK,
    CrowsPairsLoader,
    ExclusionReason,
    KeptWithoutCell,
    crows_pairs_path,
    load_benchmark_config,
    read_rows,
    screen,
)
from margins.benchmarks.config import BenchmarkConfig, default_benchmark_config_dir
from margins.benchmarks.crows_pairs import CrowsPairsPayload, source_rows
from margins.config import DEFAULT_BENCHMARK_NAME
from margins.lattice import DeclaredLattice, load_lattice_config
from margins.schemas import PAYLOAD_TYPES, Item, TaskType

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "benchmarks" / "crows_eyeball.csv"
FIXTURE_CONFIG_DIR = Path(__file__).parent / "fixtures" / "benchmark"


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config())


@pytest.fixture(scope="module")
def tiny_config() -> BenchmarkConfig:
    return load_benchmark_config("tiny", config_dir=FIXTURE_CONFIG_DIR)


@pytest.fixture
def fixture_data_dir(tmp_path: Path) -> Path:
    """The eyeball fixture, placed where the loader looks for the real file."""
    destination = crows_pairs_path(tmp_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_CSV, destination)
    return tmp_path


# --------------------------------------------------------------------------------------
# The pin
# --------------------------------------------------------------------------------------


def test_the_release_is_pinned_to_a_commit_not_a_branch() -> None:
    release = CROWS_PAIRS_RELEASE
    assert len(release.revision) == 40, "a branch name is not a pin"
    assert release.revision in release.raw_url
    assert "master" not in release.raw_url.rsplit("/", 3)[0]
    assert len(release.sha256) == 64
    assert release.license, "an unrecorded licence blocks the Zenodo release"
    assert release.rows > 0


def test_the_config_and_the_release_constant_agree() -> None:
    """Two copies of one pin. Duplicated deliberately -- the YAML is what the manifest digests,
    the constant is what the fetch script uses -- so a test has to hold them together."""
    config = load_benchmark_config()
    assert config.source.revision == CROWS_PAIRS_RELEASE.revision
    assert config.source.sha256 == CROWS_PAIRS_RELEASE.sha256
    assert config.source.license == CROWS_PAIRS_RELEASE.license
    assert config.expected_rows == CROWS_PAIRS_RELEASE.rows


def test_the_default_benchmark_name_matches_the_config_module() -> None:
    """`margins.config` duplicates the name as a literal to avoid an import cycle."""
    assert DEFAULT_BENCHMARK_NAME == DEFAULT_BENCHMARK


def test_a_branch_name_is_rejected_as_a_revision() -> None:
    payload = load_benchmark_config().model_dump(mode="json")
    payload["source"]["revision"] = "master"
    with pytest.raises(ValueError, match="not a 40-character commit sha"):
        BenchmarkConfig.model_validate(payload)


def test_the_payload_type_is_registered_at_the_loader_boundary() -> None:
    """Invariant 2: source quirks live only in Item.payload, typed where the loader is."""
    assert PAYLOAD_TYPES["crows_pairs"] is CrowsPairsPayload


def test_the_config_filename_and_key_must_agree() -> None:
    """`wrong_key.yaml` declares `key: tiny`, so the manifest would name a file it did not read."""
    with pytest.raises(ValueError, match="filename and the key must agree"):
        load_benchmark_config("wrong_key", config_dir=FIXTURE_CONFIG_DIR)


def test_a_missing_config_dir_is_named() -> None:
    with pytest.raises(FileNotFoundError, match="benchmark config dir"):
        load_benchmark_config(config_dir=default_benchmark_config_dir() / "nope")


# --------------------------------------------------------------------------------------
# Reading: checksums and row counts (gate P2)
# --------------------------------------------------------------------------------------


def test_row_count_and_checksum_are_verified_on_load(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig
) -> None:
    rows = source_rows(tiny_config, data_dir=fixture_data_dir)
    assert len(rows) == tiny_config.expected_rows == 7
    digest = hashlib.sha256(crows_pairs_path(fixture_data_dir).read_bytes()).hexdigest()
    assert digest == tiny_config.source.sha256


def test_a_file_that_is_not_the_pinned_one_is_refused(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig
) -> None:
    path = crows_pairs_path(fixture_data_dir)
    path.write_bytes(path.read_bytes() + b"7,a,b,stereo,age,x,a0,x\n")
    with pytest.raises(ValueError, match="not the pinned file"):
        source_rows(tiny_config, data_dir=fixture_data_dir)


def test_a_wrong_row_count_is_refused_even_at_a_matching_digest(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig
) -> None:
    """Redundant with the digest on purpose: a digest says 'not these bytes', a row count says
    how they differ, which is what gate P2 asks to be reported."""
    path = crows_pairs_path(fixture_data_dir)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Gate P2 requires the row count"):
        read_rows(path, expected_sha256=digest, expected_rows=999)


def test_a_missing_file_names_the_fetch_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"fetch_crows_pairs\.py"):
        read_rows(crows_pairs_path(tmp_path), verify_checksum=False)


def test_an_unrecognised_stereo_direction_raises(tmp_path: Path) -> None:
    """That column decides WHICH sentence is the stereotyping one; guessing would flip a sign."""
    path = crows_pairs_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",sent_more,sent_less,stereo_antistereo,bias_type\n0,a,b,sideways,age\n",
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(ValueError, match=r"stereo_antistereo"):
        read_rows(path, verify_checksum=False, expected_rows=1)


def test_a_missing_upstream_column_raises(tmp_path: Path) -> None:
    path = crows_pairs_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",sent_more,bias_type\n0,a,age\n", encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="missing required column"):
        read_rows(path, verify_checksum=False, expected_rows=1)


# --------------------------------------------------------------------------------------
# THE SIGN HAZARD
# --------------------------------------------------------------------------------------


def test_an_antistereo_row_swaps_the_two_sentences(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig
) -> None:
    """On an `antistereo` row, upstream's `sent_more` is the ANTI-stereotypical member.

    Getting this backwards inverts the sign of every CrowS effect while the pipeline stays green,
    so the swap is resolved once, at the loader boundary, and asserted here.
    """
    rows = {row.row_index: row for row in source_rows(tiny_config, data_dir=fixture_data_dir)}

    stereo_row = rows[0]
    assert stereo_row.stereo_antistereo == "stereo"
    assert stereo_row.sent_stereo == stereo_row.sent_more
    assert stereo_row.sent_antistereo == stereo_row.sent_less

    anti_row = rows[1]
    assert anti_row.stereo_antistereo == "antistereo"
    assert anti_row.sent_stereo == anti_row.sent_less, "sent_more is the ANTI-stereotypical member"
    assert anti_row.sent_antistereo == anti_row.sent_more
    assert "Black" in anti_row.sent_stereo


def test_the_payload_carries_the_resolved_direction(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig, lattice: DeclaredLattice
) -> None:
    result = _screen(tiny_config, fixture_data_dir, lattice)
    items = {item.payload["row_index"]: item for item in result.items}
    anti = items[1]
    assert anti.payload["sent_stereo"] == anti.payload["sent_less"]
    assert anti.payload["sent_antistereo"] == anti.payload["sent_more"]


# --------------------------------------------------------------------------------------
# The 5-item eyeball, on the fixture and on the real file
# --------------------------------------------------------------------------------------


def _screen(config: BenchmarkConfig, data_dir: Path, lattice: DeclaredLattice):  # type: ignore[no-untyped-def]
    return CrowsPairsLoader(config, lattice, data_dir=data_dir).screen()


def test_the_eyeball_fixture_screens_exactly_as_documented(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig, lattice: DeclaredLattice
) -> None:
    """Every crafted row exercises one branch, and each outcome is asserted individually."""
    result = _screen(tiny_config, fixture_data_dir, lattice)
    coverage = {row.row_index: row for row in result.coverage}
    excluded = {e.row_index: e for e in result.exclusions}

    # 0: minimal race contrast on two lattice values -> order-1 on the stereo side.
    assert coverage[0].cell_id == "race_ethnicity=Black"
    assert coverage[0].order == 1

    # 1: antistereo -> the cell comes from sent_LESS, which is the stereotyping member.
    assert coverage[1].cell_id == "race_ethnicity=Black"

    # 2: gender contrast with "Black" SHARED by both sentences -> order-2.
    assert coverage[2].order == 2
    assert coverage[2].cell_id == "race_ethnicity=Black|gender_and_sex=female"

    # 3: "poor"/"rich" are not lattice values -> kept, no cell, and the reason says which.
    assert coverage[3].order == 0
    assert coverage[3].cell_id is None
    assert coverage[3].no_cell_reason == KeptWithoutCell.NO_ATTESTED_LATTICE_VALUE.value

    # 4: religion has no lattice axis -> KEPT and MARKED, never dropped, never celled.
    assert coverage[4].out_of_lattice is True
    assert coverage[4].cell_id is None
    assert coverage[4].no_cell_reason == KeptWithoutCell.OUT_OF_LATTICE_CATEGORY.value

    # 5: two genuine content spans -> excluded, with a reason.
    assert excluded[5].reason is ExclusionReason.NOT_MINIMAL_PAIR

    # 6: swap plus obligatory pronoun concord -> admitted by the declared allowance.
    assert 6 not in excluded
    assert 6 in result.admitted_by_agreement


def test_out_of_lattice_items_are_kept_not_dropped(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig, lattice: DeclaredLattice
) -> None:
    result = _screen(tiny_config, fixture_data_dir, lattice)
    religion = [i for i in result.items if i.payload["bias_type"] == "religion"]
    assert len(religion) == 1, "an unmapped category is reportable, not discarded"
    assert religion[0].payload["out_of_lattice"] is True
    assert religion[0].payload["mapped_axis"] is None
    assert religion[0].payload["attested"] == ()


def test_items_are_well_formed_pair_pll_records(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig, lattice: DeclaredLattice
) -> None:
    result = _screen(tiny_config, fixture_data_dir, lattice)
    assert result.items, "the fixture should survive screening"
    for item in result.items:
        assert isinstance(item, Item)
        assert item.task_type is TaskType.PAIR_PLL
        assert item.benchmark == "crows_pairs"
        assert item.gold is None, "pair_pll carries both sentences in payload, not a gold string"
        assert item.provenance.sha256 == tiny_config.source.sha256


# --------------------------------------------------------------------------------------
# Every exclusion carries a reason; screening is deterministic (gate P2)
# --------------------------------------------------------------------------------------


def test_every_exclusion_carries_a_machine_readable_reason(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig, lattice: DeclaredLattice
) -> None:
    result = _screen(tiny_config, fixture_data_dir, lattice)
    assert result.exclusions
    for exclusion in result.exclusions:
        assert isinstance(exclusion.reason, ExclusionReason)
        assert exclusion.detail.strip(), "a reason without a detail cannot be audited"


def test_screening_is_deterministic_and_order_independent(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig, lattice: DeclaredLattice
) -> None:
    """Deterministic is stronger than 'no random seed': the per-row decision must not depend on
    which other rows preceded it."""
    rows = list(source_rows(tiny_config, data_dir=fixture_data_dir))

    first = screen(rows, config=tiny_config, lattice=lattice)
    second = screen(rows, config=tiny_config, lattice=lattice)
    assert first == second

    shuffled = list(rows)
    random.Random(20260920).shuffle(shuffled)
    third = screen(shuffled, config=tiny_config, lattice=lattice)

    def decisions(result: object) -> dict[int, tuple[object, object]]:
        by_row = {c.row_index: ("kept", c.cell_id) for c in result.coverage}  # type: ignore[attr-defined]
        by_row.update({e.row_index: ("excluded", e.reason) for e in result.exclusions})  # type: ignore[attr-defined]
        return by_row

    assert decisions(third) == decisions(first)


def test_an_undeclared_category_raises_rather_than_being_skipped(
    fixture_data_dir: Path, tiny_config: BenchmarkConfig, lattice: DeclaredLattice
) -> None:
    """Upstream gaining a category is a human decision, not something a filter may swallow."""
    rows = list(source_rows(tiny_config, data_dir=fixture_data_dir))
    mutated = [rows[0]._replace(bias_type="hair-colour"), *rows[1:]]
    with pytest.raises(ValueError, match="does not declare"):
        screen(mutated, config=tiny_config, lattice=lattice)


def test_a_category_mapped_to_an_unknown_axis_raises(lattice: DeclaredLattice) -> None:
    payload = load_benchmark_config("tiny", config_dir=FIXTURE_CONFIG_DIR).model_dump(mode="json")
    payload["attributes"][0]["axis"] = "astrology"
    config = BenchmarkConfig.model_validate(payload)
    with pytest.raises(ValueError, match="axes the lattice does not declare"):
        config.validate_against_lattice(lattice.config.ordered_axes)


def test_an_unmapped_category_must_record_a_reason() -> None:
    """`axis: null` is a declared outcome, and is indistinguishable from an unfinished mapping
    unless it says why."""
    payload = load_benchmark_config("tiny", config_dir=FIXTURE_CONFIG_DIR).model_dump(mode="json")
    religion = next(a for a in payload["attributes"] if a["category"] == "religion")
    religion["note"] = None
    with pytest.raises(ValueError, match="records no note"):
        BenchmarkConfig.model_validate(payload)


# --------------------------------------------------------------------------------------
# The real pinned file
# --------------------------------------------------------------------------------------

_REAL = crows_pairs_path()
_needs_real_file = pytest.mark.skipif(
    not _REAL.is_file(),
    reason="pinned crows_pairs_anonymized.csv absent; run scripts/fetch_crows_pairs.py",
)


@_needs_real_file
def test_the_real_file_matches_the_published_row_count() -> None:
    """Gate P2: 'row counts match published dataset'. 1,508 pairs (Nangia et al., EMNLP 2020)."""
    rows = read_rows()
    assert len(rows) == 1508 == CROWS_PAIRS_RELEASE.rows


@_needs_real_file
def test_the_real_file_declares_exactly_the_mapped_categories() -> None:
    """The nine upstream categories, all declared. A tenth would fail the screen, not be skipped."""
    with _REAL.open(encoding="utf-8") as handle:
        categories = {row["bias_type"] for row in csv.DictReader(handle)}
    declared = set(load_benchmark_config().axis_for)
    assert categories <= declared, (
        f"undeclared upstream categories: {sorted(categories - declared)}"
    )
    assert categories == {
        "race-color",
        "gender",
        "socioeconomic",
        "nationality",
        "religion",
        "age",
        "sexual-orientation",
        "physical-appearance",
        "disability",
    }


@_needs_real_file
def test_the_real_file_has_antistereo_rows_so_the_swap_is_not_hypothetical() -> None:
    rows = read_rows()
    anti = [r for r in rows if r.stereo_antistereo == "antistereo"]
    assert len(anti) == 218
    for row in anti[:5]:
        assert row.sent_stereo == row.sent_less


@_needs_real_file
def test_a_five_item_eyeball_of_the_real_file(lattice: DeclaredLattice) -> None:
    """Gate P2's '5-item eyeball', on the data rather than on a fixture.

    Five real rows, each screened, each outcome asserted against what the sentences plainly say.
    Kept small and concrete on purpose: the point of an eyeball is that a human can check it.
    """
    config = load_benchmark_config()
    result = CrowsPairsLoader(config, lattice).screen()
    coverage = {row.row_index: row for row in result.coverage}
    excluded = {e.row_index: e for e in result.exclusions}
    rows = {row.row_index: row for row in read_rows()}

    # Row 0: "most black kids" vs "most white kids" -- race contrast, no second axis shared.
    assert "black kids" in rows[0].sent_stereo
    assert coverage[0].cell_id == "race_ethnicity=Black"
    assert coverage[0].order == 1

    # Row 1: "The poor" vs "The rich" -- neither is a lattice value, so no cell is invented.
    assert rows[1].bias_type == "socioeconomic"
    assert coverage[1].cell_id is None
    assert coverage[1].no_cell_reason == KeptWithoutCell.NO_ATTESTED_LATTICE_VALUE.value

    # Every one of the five is accounted for exactly once: kept or excluded, never both, never
    # neither. A row that fell through the screen silently is the failure this asserts against.
    for index in range(5):
        assert (index in coverage) != (index in excluded), f"row {index} was not accounted for"


@_needs_real_file
def test_the_real_screen_reports_its_order_two_yield(lattice: DeclaredLattice) -> None:
    """The MEASURED number that decides whether this paradigm can support an interaction.

    It is tiny, and that is the finding rather than a bug: each CrowS item is annotated with ONE
    attribute, so an order-2 cell needs a second identity term present unchanged in both
    sentences, which is rare. The assertion is deliberately loose -- it pins the ORDER of
    magnitude, so a future change that accidentally started inventing cells would fail here.
    """
    result = CrowsPairsLoader(load_benchmark_config(), lattice).screen()
    order_2 = result.order_2_cells()
    assert sum(order_2.values()) < 20, (
        "the extractive paradigm suddenly attests many order-2 cells; either the lattice grew or "
        "attestation stopped being literal. Both are human decisions."
    )
    assert result.counts_by_order()["order_1"] > sum(order_2.values())
