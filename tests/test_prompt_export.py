"""Prompt-set exports and the phrase review (P4a).

Gate P4a asks for byte-identical regeneration, a 20-prompt slot audit, and — folded in from P3 —
a phrase review covering all 92 order-2 composed descriptors **inside a rendered sentence**.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.prompts import (
    DEFAULT_PROMPT_SET,
    FILL_AUDIT_FILENAME,
    FILL_AUDIT_SAMPLE_SIZE,
    PHRASE_AUDIT_FILENAME,
    PHRASE_REVIEW_NOTES,
    PROMPTS_FILENAME,
    PromptSetConfig,
    TemplateRenderer,
    cell_surfaces,
    export_prompt_set,
    load_prompt_config,
    load_template_files,
    mechanical_flags,
    sample_for_fill_audit,
    template_items,
)
from margins.prompts.config import TemplateFile
from margins.schemas import Prompt


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))


@pytest.fixture(scope="module")
def config() -> PromptSetConfig:
    return load_prompt_config(DEFAULT_PROMPT_SET)


@pytest.fixture(scope="module")
def files(config: PromptSetConfig) -> tuple[TemplateFile, ...]:
    return load_template_files(config)


@pytest.fixture(scope="module")
def prompts(
    lattice: DeclaredLattice, config: PromptSetConfig, files: tuple[TemplateFile, ...]
) -> list[Prompt]:
    renderer = TemplateRenderer(config, lattice)
    return renderer.render(template_items(config, files), lattice.cells())


# ---------------------------------------------------------------------------- the prompt file


def test_exports_regenerate_byte_identically(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: list[Prompt],
    files: tuple[TemplateFile, ...],
    tmp_path: Path,
) -> None:
    """Gate P4a. A diff must show a real change, not a line-ending difference."""
    first = export_prompt_set(lattice, config, prompts, files, tmp_path / "a")
    second = export_prompt_set(lattice, config, prompts, files, tmp_path / "b")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
        assert b"\r\n" not in left.read_bytes()


def test_export_writes_the_three_named_artifacts(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: list[Prompt],
    files: tuple[TemplateFile, ...],
    tmp_path: Path,
) -> None:
    written = export_prompt_set(lattice, config, prompts, files, tmp_path)
    assert [path.name for path in written] == [
        PROMPTS_FILENAME,
        FILL_AUDIT_FILENAME,
        PHRASE_AUDIT_FILENAME,
    ]


def test_prompt_file_has_one_json_record_per_prompt(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: list[Prompt],
    files: tuple[TemplateFile, ...],
    tmp_path: Path,
) -> None:
    import json

    path, _, _ = export_prompt_set(lattice, config, prompts, files, tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(prompts)
    records = [json.loads(line) for line in lines]
    assert {record["prompt_id"] for record in records} == {p.prompt_id for p in prompts}
    assert all(record["task_type"] == "open_gen" for record in records)
    assert all(record["template_version"] == config.template_version for record in records)


# ---------------------------------------------------------------------------- the slot audit


def test_slot_audit_sample_is_stratified_and_deterministic(
    prompts: list[Prompt], lattice: DeclaredLattice
) -> None:
    """A head-of-file sample would audit one template and one axis pair."""
    first = sample_for_fill_audit(prompts, lattice)
    second = sample_for_fill_audit(prompts, lattice)
    assert [p.prompt_id for p in first] == [p.prompt_id for p in second]
    assert len(first) == FILL_AUDIT_SAMPLE_SIZE

    groups = {record.cell_id: record for record in lattice.records()}
    orders = {groups[p.cell_id].order for p in first if p.cell_id}
    assert orders == {1, 2}
    pairs = {groups[p.cell_id].axis_pair for p in first if p.cell_id}
    assert len(pairs) >= 7, "every declared axis pair plus order-1 should be represented"

    # A *slot* audit whose rows all come from one frame has not audited the frames.
    templates = {p.item_id for p in first}
    assert len(templates) >= 6, f"sample spans only {len(templates)} frames"


def test_fill_audit_documents_the_carrier_construct_choice(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: list[Prompt],
    files: tuple[TemplateFile, ...],
    tmp_path: Path,
) -> None:
    _, audit, _ = export_prompt_set(lattice, config, prompts, files, tmp_path)
    text = audit.read_text(encoding="utf-8")
    assert "Carrier noun" in text
    assert "undeclared second axis of variation" in text
    assert "Leakage guard" in text
    assert f"Slot audit — {FILL_AUDIT_SAMPLE_SIZE} sampled fills" in text


# ---------------------------------------------------------------------------- the phrase review


def test_phrase_audit_covers_every_order_2_descriptor_in_a_sentence(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: list[Prompt],
    files: tuple[TemplateFile, ...],
    tmp_path: Path,
) -> None:
    """All 92, shown in frame -- a bare phrase list cannot show an article defect."""
    _, _, phrase = export_prompt_set(lattice, config, prompts, files, tmp_path)
    text = phrase.read_text(encoding="utf-8")

    order_2 = [record for record in lattice.records() if record.order == 2]
    assert len(order_2) == 92
    for record in order_2:
        assert f"| {record.descriptor} |" in text, f"{record.descriptor} missing from phrase audit"
        # Either declared carrier is acceptable. The audit shows ONE rendered sentence per
        # descriptor and which carrier it lands on is incidental; what it exists to expose is
        # the article and the composed phrase, both carrier-independent.
        shown = any(
            f"about a {record.descriptor} {carrier.noun}." in text
            or f"about an {record.descriptor} {carrier.noun}." in text
            for carrier in config.carriers
        )
        assert shown, f"{record.descriptor} not shown inside a rendered sentence"


def test_phrase_audit_states_that_descriptors_are_never_the_lever(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: list[Prompt],
    files: tuple[TemplateFile, ...],
    tmp_path: Path,
) -> None:
    _, _, phrase = export_prompt_set(lattice, config, prompts, files, tmp_path)
    text = phrase.read_text(encoding="utf-8")
    assert "never edited to fix phrasing" in text
    assert "compose_rank" in text


def test_every_phrase_review_note_records_a_decision(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: list[Prompt],
    files: tuple[TemplateFile, ...],
    tmp_path: Path,
) -> None:
    """Notes are kept after resolution: the review is provenance, not a scratch list."""
    assert PHRASE_REVIEW_NOTES
    for note in PHRASE_REVIEW_NOTES:
        assert note.status.split()[0] in {"OPEN", "RESOLVED", "MITIGATED", "ACCEPTED"}
        assert note.proposed_lever.strip()
        assert note.why_it_matters.strip()
        assert note.resolution.strip()

    _, _, phrase = export_prompt_set(lattice, config, prompts, files, tmp_path)
    text = phrase.read_text(encoding="utf-8")
    for note in PHRASE_REVIEW_NOTES:
        assert f"`{note.id}`" in text
        assert note.status in text


def _affected(lattice: DeclaredLattice) -> dict[str, int]:
    order_2 = [record for record in lattice.records() if record.order == 2]
    return {
        note.id: sum(
            1
            for record in order_2
            if note.applies(lattice, record, mechanical_flags(cell_surfaces(lattice, record)))
        )
        for note in PHRASE_REVIEW_NOTES
    }


def test_resolved_notes_match_zero_cells(lattice: DeclaredLattice) -> None:
    """The regression guard: reverting a compose_rank must fail the suite, not pass quietly."""
    affected = _affected(lattice)
    for note in PHRASE_REVIEW_NOTES:
        if note.status.startswith("RESOLVED"):
            assert affected[note.id] == 0, (
                f"{note.id} is recorded RESOLVED but still matches {affected[note.id]} cells; "
                "either the fix was reverted or the note's status is wrong"
            )


def test_unresolved_notes_report_accurate_counts(lattice: DeclaredLattice) -> None:
    """MITIGATED and ACCEPTED notes keep a non-zero count on purpose -- it is the residual cost."""
    affected = _affected(lattice)
    # Hand-checked against the shipped confirmatory lattice v1.1.
    assert affected["gender-composed-before-non-gender"] == 0
    assert affected["age-before-status-adjective"] == 0
    assert affected["marked-register-with-neutral-carrier"] == 24
    assert affected["repeated-morpheme"] == 1


def test_compose_order_is_native_english_for_every_declared_pair(
    lattice: DeclaredLattice,
) -> None:
    """Lattice v1.1: class < age < ability < race < gender, the unique order fitting all 7 pairs."""
    rank = lattice.config.axis_rank
    assert (
        rank("socioeconomic_class")
        < rank("age")
        < rank("ability")
        < rank("race_ethnicity")
        < rank("gender_and_sex")
    )
    descriptors = {record.descriptor for record in lattice.records() if record.order == 2}
    assert "wealthy young" in descriptors and "young wealthy" not in descriptors
    assert "Black female" in descriptors and "female Black" not in descriptors
    assert "working-class Black" in descriptors
    assert "older blind" in descriptors
    assert "blind female" in descriptors


def test_mechanical_flags_catch_only_string_decidable_defects() -> None:
    assert mechanical_flags(("middle-aged", "middle-class")) == (
        "repeated-token:middle",
        "stacked-hyphenated-compounds",
    )
    assert mechanical_flags(("older", "blind")) == ()
    assert mechanical_flags(("working-class", "Black")) == ()
    assert mechanical_flags(("blind",)) == ()


def test_mechanical_flags_catch_a_nested_surface() -> None:
    flags = mechanical_flags(("class", "working-class"))
    assert any(flag.startswith("substring-overlap") for flag in flags)
