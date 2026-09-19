"""``TemplateRenderer`` — determinism, prompt ids, the leakage guard, open_gen-only (P4a).

Gate P4a is byte-identical regeneration plus a slot audit; the leakage guard is the construct-
validity half. Identity signal outside the injected slot is not a cosmetic defect: it applies to
every cell, so it contaminates precisely the interaction the instrument exists to measure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from margins.interfaces import PromptGenerator
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.prompts import (
    DEFAULT_PROMPT_SET,
    TEMPLATE_ITEM_BENCHMARK,
    PromptSetConfig,
    TemplateRenderer,
    axis_vocabulary,
    identity_vocabulary,
    load_prompt_config,
    load_template_files,
    matched_terms,
    template_item_id,
    template_items,
)
from margins.prompts.config import TEMPLATE_DIRNAME, TemplateFile, TemplateSpec, file_sha256
from margins.schemas import IdentityCell, Item, Prompt, Provenance, TaskType

LATTICE_FIXTURES = Path(__file__).parent / "fixtures" / "lattice"
PROMPT_FIXTURES = Path(__file__).parent / "fixtures" / "prompts"
FIXTURE_TEMPLATES = PROMPT_FIXTURES / TEMPLATE_DIRNAME


# ---------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tiny_lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURES))


@pytest.fixture(scope="module")
def tiny_config() -> PromptSetConfig:
    return load_prompt_config("tiny", config_dir=PROMPT_FIXTURES)


@pytest.fixture(scope="module")
def tiny_items(tiny_config: PromptSetConfig) -> list[Item]:
    return template_items(tiny_config, load_template_files(tiny_config, FIXTURE_TEMPLATES))


@pytest.fixture(scope="module")
def shipped_lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))


@pytest.fixture(scope="module")
def shipped_config() -> PromptSetConfig:
    return load_prompt_config(DEFAULT_PROMPT_SET)


@pytest.fixture(scope="module")
def shipped_prompts(
    shipped_lattice: DeclaredLattice, shipped_config: PromptSetConfig
) -> list[Prompt]:
    files = load_template_files(shipped_config)
    renderer = TemplateRenderer(shipped_config, shipped_lattice)
    return renderer.render(template_items(shipped_config, files), shipped_lattice.cells())


def _loose_template(config: PromptSetConfig, name: str) -> list[Item]:
    """Build items from a fixture template file that the tiny config deliberately omits."""
    path = FIXTURE_TEMPLATES / f"{name}.jinja"
    spec = TemplateSpec(id=name, file=f"{name}.jinja", note="deliberately broken fixture frame")
    text = " ".join(path.read_text(encoding="utf-8").split())
    return template_items(
        config, (TemplateFile(spec=spec, text=text, sha256=file_sha256(path), path=path),)
    )


# ---------------------------------------------------------------------------- the Protocol


def test_renderer_satisfies_the_prompt_generator_protocol(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice
) -> None:
    assert isinstance(TemplateRenderer(tiny_config, tiny_lattice), PromptGenerator)


def test_templates_are_items_carrying_provenance(tiny_items: list[Item]) -> None:
    """Every prompt must trace to something version-anchored (PROJECT_CONTEXT.md section 3)."""
    for item in tiny_items:
        assert item.benchmark == TEMPLATE_ITEM_BENCHMARK
        assert item.task_type is TaskType.OPEN_GEN
        assert isinstance(item.provenance, Provenance)
        assert item.provenance.version == "tiny@test"
        assert item.payload["template_text"]
        assert item.payload["slots"]


def test_template_slots_are_detected_from_the_frame_not_declared(tiny_items: list[Item]) -> None:
    by_id = {item.id: item for item in tiny_items}
    assert by_id[template_item_id("neutral")].payload["slots"] == ("identity",)
    assert by_id[template_item_id("parts")].payload["slots"] == ("descriptor", "carrier", "article")


# ---------------------------------------------------------------------------- open_gen only


def test_renderer_refuses_an_extractive_item(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice
) -> None:
    """The cell is INJECTED here. A derived cell belongs to ExtractiveRenderer (P4b)."""
    item = Item(
        id="crows-1",
        benchmark="crows_pairs",
        task_type=TaskType.PAIR_PLL,
        payload={"sent_more": "x", "sent_less": "y"},
        provenance=Provenance(
            source="fixture",
            version="v1",
            license="MIT",
            sha256="0" * 64,
            retrieved_at=datetime(2026, 9, 8, tzinfo=UTC),
        ),
    )
    renderer = TemplateRenderer(tiny_config, tiny_lattice)
    with pytest.raises(ValueError, match="ExtractiveRenderer"):
        renderer.render([item], tiny_lattice.cells())


def test_every_rendered_prompt_is_open_gen(shipped_prompts: list[Prompt]) -> None:
    """No paired-task path exists at P4a; open decision 7 stays open."""
    assert all(prompt.task_type is TaskType.OPEN_GEN for prompt in shipped_prompts)


def test_no_prompt_carries_paired_task_content(shipped_prompts: list[Prompt]) -> None:
    """A Prompt has one rendered_text. Nothing here encodes a second sequence."""
    assert all(isinstance(prompt.rendered_text, str) for prompt in shipped_prompts)
    assert all(prompt.cell_id is not None for prompt in shipped_prompts)


# ---------------------------------------------------------------------------- determinism


def test_regeneration_is_byte_identical(
    shipped_lattice: DeclaredLattice, shipped_config: PromptSetConfig
) -> None:
    """Gate P4a: the same config and lattice regenerate the same prompt set, byte for byte."""
    from margins.prompts import render_prompts_jsonl

    def build() -> list[Prompt]:
        lattice = DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))
        config = load_prompt_config(DEFAULT_PROMPT_SET)
        files = load_template_files(config)
        renderer = TemplateRenderer(config, lattice)
        return renderer.render(template_items(config, files), lattice.cells())

    first = render_prompts_jsonl(build()).encode("utf-8")
    second = render_prompts_jsonl(build()).encode("utf-8")
    assert first == second
    assert b"\r\n" not in first


def test_prompt_ids_are_stable_and_order_independent(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice, tiny_items: list[Item]
) -> None:
    """Ids are content-derived, so shuffling the inputs cannot change them."""
    renderer = TemplateRenderer(tiny_config, tiny_lattice)
    cells = tiny_lattice.cells()

    forward = renderer.render(tiny_items, cells)
    reversed_inputs = renderer.render(list(reversed(tiny_items)), list(reversed(cells)))

    assert {p.prompt_id for p in forward} == {p.prompt_id for p in reversed_inputs}
    by_id = {p.prompt_id: p.rendered_text for p in reversed_inputs}
    assert all(by_id[p.prompt_id] == p.rendered_text for p in forward)


def test_prompt_ids_are_unique_and_readable(
    shipped_prompts: list[Prompt], shipped_config: PromptSetConfig
) -> None:
    carriers = {carrier.noun for carrier in shipped_config.carriers}
    ids = [prompt.prompt_id for prompt in shipped_prompts]
    assert len(set(ids)) == len(ids)
    for prompt in shipped_prompts:
        template, cell_id, carrier, digest = prompt.prompt_id.split("::")
        assert template and carrier in carriers
        assert cell_id == prompt.cell_id
        assert len(digest) == 12


def test_no_two_prompts_share_rendered_text(shipped_prompts: list[Prompt]) -> None:
    """With two carriers, the carrier-free frame must not produce duplicates.

    A duplicate text under two ids would be counted as two independent ITEM CLUSTERS by the
    emergence estimator, shrinking every cluster-robust standard error. The renderer emits a
    carrier-free frame once for exactly this reason.
    """
    texts = [prompt.rendered_text for prompt in shipped_prompts]
    assert len(set(texts)) == len(texts)


def test_prompt_id_changes_when_the_rendered_text_changes(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice
) -> None:
    """An unversioned template edit shows up as changed ids, not as silent drift."""
    renderer = TemplateRenderer(tiny_config, tiny_lattice)
    cells = tiny_lattice.cells()[:1]

    original = template_items(tiny_config, load_template_files(tiny_config, FIXTURE_TEMPLATES))[:1]
    edited = [
        original[0].model_copy(
            update={
                "payload": {
                    "template_text": "Write a long story about {{ identity }}.",
                    "slots": ("identity",),
                }
            }
        )
    ]
    assert (
        renderer.render(original, cells)[0].prompt_id != renderer.render(edited, cells)[0].prompt_id
    )


def test_every_prompt_carries_the_template_version(
    shipped_prompts: list[Prompt], shipped_config: PromptSetConfig
) -> None:
    assert all(p.template_version == shipped_config.template_version for p in shipped_prompts)
    assert all(p.decoding_key == shipped_config.decoding_key for p in shipped_prompts)


def test_prompt_count_is_templates_times_cells_times_carriers(
    shipped_prompts: list[Prompt],
    shipped_config: PromptSetConfig,
    shipped_lattice: DeclaredLattice,
) -> None:
    # A carrier-free frame is emitted ONCE, so the count is not a flat product. This is the
    # arithmetic the P9 power analysis depends on: 13 carrier-using frames x 2 carriers + 1
    # carrier-free frame = 27 prompts per cell.
    renderer = TemplateRenderer(shipped_config, shipped_lattice)
    prepared = renderer.prepare(template_items(shipped_config, load_template_files(shipped_config)))
    using = sum(1 for template in prepared if template.uses_carrier)
    free = sum(1 for template in prepared if not template.uses_carrier)
    per_cell = using * len(shipped_config.carriers) + free
    assert per_cell == 27
    assert len(shipped_prompts) == per_cell * len(shipped_lattice.cells()) == 2970


def test_every_cell_is_reached_including_non_gender_intersections(
    shipped_prompts: list[Prompt], shipped_lattice: DeclaredLattice
) -> None:
    """Gate P3's non-gender x non-gender cells are reachable ONLY through this path."""
    rendered = {prompt.cell_id for prompt in shipped_prompts}
    assert rendered == {cell.cell_id for cell in shipped_lattice.cells()}

    config = shipped_lattice.config
    non_gender = [
        record
        for record in shipped_lattice.records()
        if record.axis_pair is not None and config.is_non_gender_pair(record.axis_pair)
    ]
    assert non_gender
    assert all(record.cell_id in rendered for record in non_gender)


# ---------------------------------------------------------------------------- leakage guard


def test_identity_terms_appear_only_in_the_injected_slot(
    shipped_prompts: list[Prompt], shipped_lattice: DeclaredLattice
) -> None:
    """The guard, restated as an assertion over the whole shipped set.

    Set equality in both directions: extra terms mean leakage, missing terms mean the injection
    silently did not happen.
    """
    vocabulary = identity_vocabulary(shipped_lattice)
    surfaces = {(v.axis, v.value): v.surface for v in shipped_lattice.values()}
    cells = {cell.cell_id: cell for cell in shipped_lattice.cells()}

    for prompt in shipped_prompts:
        assert prompt.cell_id is not None
        cell = cells[prompt.cell_id]
        expected = {surfaces[pair] for pair in cell.axes} | {value for _, value in cell.axes}
        found = set(matched_terms(prompt.rendered_text, vocabulary))
        assert found <= expected, f"leak in {prompt.prompt_id}: {sorted(found - expected)}"
        assert found, f"no identity injected into {prompt.prompt_id}"


def test_no_frame_names_an_axis_word(
    shipped_prompts: list[Prompt], shipped_lattice: DeclaredLattice
) -> None:
    """Naming the axis ('race', 'disability') primes it for every cell on that axis."""
    axis_words = axis_vocabulary(shipped_lattice)
    assert "race" in axis_words and "gender" in axis_words and "disability" in axis_words
    for prompt in shipped_prompts:
        assert not matched_terms(prompt.rendered_text, axis_words), prompt.rendered_text


def test_a_frame_naming_an_identity_value_is_rejected(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice
) -> None:
    renderer = TemplateRenderer(tiny_config, tiny_lattice)
    with pytest.raises(ValueError, match="identity term"):
        renderer.render(_loose_template(tiny_config, "leaky_value"), tiny_lattice.cells())


def test_a_frame_naming_an_axis_is_rejected(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice
) -> None:
    renderer = TemplateRenderer(tiny_config, tiny_lattice)
    with pytest.raises(ValueError, match="AXIS word"):
        renderer.render(_loose_template(tiny_config, "leaky_axis"), tiny_lattice.cells())


def test_a_sentence_initial_identity_slot_is_rejected(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice
) -> None:
    """Capitalising the slot would silently rewrite pinned descriptors ('Black', 'Deaf')."""
    renderer = TemplateRenderer(tiny_config, tiny_lattice)
    with pytest.raises(ValueError, match="sentence-initially"):
        renderer.render(_loose_template(tiny_config, "sentence_initial"), tiny_lattice.cells())


def test_a_frame_with_no_identity_slot_is_rejected(tiny_config: PromptSetConfig) -> None:
    """It would render identically for every cell: a constant, not a prompt set."""
    with pytest.raises(ValueError, match="identity-bearing"):
        _loose_template(tiny_config, "no_identity")


def test_a_carrier_that_is_itself_a_lattice_term_is_rejected(
    tiny_lattice: DeclaredLattice,
) -> None:
    from margins.prompts import prompt_config_from_mapping

    config = prompt_config_from_mapping(
        {
            "template_version": "unit@test",
            "description": "d",
            "license": "MIT",
            "authored_at": "2026-09-08T00:00:00Z",
            "decoding_key": "k",
            "templates": [{"id": "t", "file": "t.jinja", "note": "n"}],
            # "b1" is a value on the tiny lattice's beta axis.
            "carriers": [{"noun": "b1", "axis_neutral": True, "note": "n"}],
        }
    )
    with pytest.raises(ValueError, match="itself a lattice identity term"):
        TemplateRenderer(config, tiny_lattice)


def test_a_cell_from_another_lattice_is_rejected(
    tiny_config: PromptSetConfig, tiny_lattice: DeclaredLattice, tiny_items: list[Item]
) -> None:
    """Mixing lattices would break the provenance trail every prompt is supposed to carry."""
    foreign = IdentityCell(cell_id="zeta=z1", axes=(("zeta", "z1"),), order=1)
    renderer = TemplateRenderer(tiny_config, tiny_lattice)
    with pytest.raises(ValueError, match="does not declare"):
        renderer.render(tiny_items, [foreign])


# ---------------------------------------------------------------------------- rendering


def test_articles_are_correct_for_vowel_initial_descriptors(
    shipped_prompts: list[Prompt],
) -> None:
    texts = [prompt.rendered_text for prompt in shipped_prompts]
    assert any("an older person." in text for text in texts)
    assert any("an autistic" in text for text in texts)
    assert any("an Asian person." in text for text in texts)
    assert not any(
        " a older" in text or " a autistic" in text or " a Asian" in text for text in texts
    )


def test_composed_order_2_descriptors_appear_in_compose_rank_order(
    shipped_prompts: list[Prompt],
) -> None:
    texts = " ".join(prompt.rendered_text for prompt in shipped_prompts)
    assert "working-class Black person" in texts
    assert "older blind person" in texts
    assert "Black working-class" not in texts
    # Lattice v1.1 ordering.
    assert "wealthy young person" in texts and "young wealthy person" not in texts
    assert "Black female person" in texts and "female Black person" not in texts


def test_rendered_text_is_single_line_and_normalized(shipped_prompts: list[Prompt]) -> None:
    for prompt in shipped_prompts:
        assert "\n" not in prompt.rendered_text
        assert prompt.rendered_text == prompt.rendered_text.strip()
        assert "  " not in prompt.rendered_text


# --------------------------------------------------------------------------------------
# Carrier-free frames are emitted ONCE (added with the second carrier, P9)
# --------------------------------------------------------------------------------------


def test_a_carrier_free_frame_is_emitted_once_per_cell(
    shipped_lattice: DeclaredLattice, shipped_config: PromptSetConfig
) -> None:
    """Duplicating it would hand the estimator identical texts as distinct ITEM CLUSTERS.

    That is not a cosmetic duplicate. The emergence estimator clusters by prompt and the cluster
    bootstrap resamples whole prompts; N copies of one prompt look like N independent clusters and
    shrink every standard error in the anti-conservative direction. This is the arithmetic the
    P9 power analysis depends on, so it is asserted rather than assumed.
    """
    from margins.prompts.config import CarrierSpec

    template_files = load_template_files(shipped_config)
    two = shipped_config.model_copy(
        update={
            "carriers": (
                CarrierSpec(noun="person", axis_neutral=True, note="shipped carrier"),
                CarrierSpec(noun="individual", axis_neutral=True, note="second carrier"),
            )
        }
    )
    renderer = TemplateRenderer(two, shipped_lattice)
    items = template_items(two, template_files)
    prepared = renderer.prepare(items)
    carrier_free = [t for t in prepared if not t.uses_carrier]
    carrier_using = [t for t in prepared if t.uses_carrier]
    assert carrier_free, "the fixture needs at least one carrier-free frame to exercise this"

    cells = shipped_lattice.cells()[:3]
    prompts = renderer.render(items, cells)

    # The arithmetic the power analysis assumes: items per cell.
    expected_per_cell = len(carrier_using) * 2 + len(carrier_free)
    assert len(prompts) == expected_per_cell * len(cells)

    # And no two prompts share their rendered text within a cell.
    for cell in cells:
        texts = [p.rendered_text for p in prompts if p.cell_id == cell.cell_id]
        assert len(texts) == len(set(texts)), "identical texts are not independent clusters"


def test_uses_carrier_reflects_the_frame_not_a_declaration(
    shipped_lattice: DeclaredLattice, shipped_config: PromptSetConfig
) -> None:
    template_files = load_template_files(shipped_config)
    renderer = TemplateRenderer(shipped_config, shipped_lattice)
    for prepared in renderer.prepare(template_items(shipped_config, template_files)):
        source = str(prepared.item.payload["template_text"])
        references = "carrier" in source or "identity" in source
        assert prepared.uses_carrier == (
            "carrier" in prepared.slots or "identity" in prepared.slots
        )
        assert prepared.uses_carrier == references, (
            f"{prepared.key}: uses_carrier must follow the FRAME, not a declaration"
        )
