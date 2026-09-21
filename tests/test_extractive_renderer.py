"""The ``ExtractiveRenderer`` and the closed open decision 7 (gate P4b).

Gate P4b has exactly two clauses:

1. **"emits no cell unattested in its source item"** -- the guarantee that separates a renderer
   that DERIVES a cell from one that INJECTS one. Most of this file is that clause, attacked from
   several directions, because it is the one property a CrowS-Pairs prompt set has to have: you
   cannot assign "Black woman" to an item about religion, and nothing downstream would notice if
   you did.
2. **"byte-identical regeneration from the same config"** -- the same determinism gate P4a passes.

The third thing tested here is the **contract change** that made a paired prompt representable at
all: ``rendered_text_stereo`` / ``rendered_text_antistereo``, resolved 20 Sep 2026, mirroring
decision 6's one ``Response`` carrying both log-probabilities.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from margins.benchmarks import CrowsPairsLoader, crows_pairs_path, load_benchmark_config
from margins.benchmarks.config import BenchmarkConfig
from margins.interfaces import PromptGenerator
from margins.lattice import DeclaredLattice, load_lattice_config
from margins.prompts import (
    DEFAULT_EXTRACTIVE_SET,
    ExtractivePromptSetConfig,
    ExtractiveRenderer,
    load_extractive_config,
    render_prompts_jsonl,
)
from margins.schemas import Item, Prompt, TaskType

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "benchmarks" / "crows_eyeball.csv"
FIXTURE_CONFIG_DIR = Path(__file__).parent / "fixtures" / "benchmark"


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config())


@pytest.fixture(scope="module")
def prompt_config() -> ExtractivePromptSetConfig:
    return load_extractive_config(DEFAULT_EXTRACTIVE_SET)


@pytest.fixture(scope="module")
def tiny_config() -> BenchmarkConfig:
    return load_benchmark_config("tiny", config_dir=FIXTURE_CONFIG_DIR)


@pytest.fixture
def items(tiny_config: BenchmarkConfig, lattice: DeclaredLattice, tmp_path: Path) -> list[Item]:
    destination = crows_pairs_path(tmp_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_CSV, destination)
    return CrowsPairsLoader(tiny_config, lattice, data_dir=tmp_path).load()


@pytest.fixture
def renderer(
    prompt_config: ExtractivePromptSetConfig, lattice: DeclaredLattice
) -> ExtractiveRenderer:
    return ExtractiveRenderer(prompt_config, lattice)


def _retarget(item: Item, **payload: object) -> Item:
    """A copy of ``item`` with its payload tampered. Invariant 1: never mutate in place."""
    merged = {**dict(item.payload), **payload}
    return item.model_copy(update={"payload": merged})


# --------------------------------------------------------------------------------------
# The Protocol, and the shape of what it emits
# --------------------------------------------------------------------------------------


def test_renderer_satisfies_the_prompt_generator_protocol(renderer: ExtractiveRenderer) -> None:
    assert isinstance(renderer, PromptGenerator)


def test_one_prompt_per_item_carrying_both_sequences(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    prompts = renderer.render(items, lattice.cells())

    assert len(prompts) == len(items)
    assert len({p.prompt_id for p in prompts}) == len(prompts), "prompt ids must be unique"
    for prompt, item in zip(prompts, items, strict=True):
        assert prompt.task_type is TaskType.PAIR_PLL
        assert prompt.item_id == item.id
        assert prompt.rendered_text_stereo == item.payload["sent_stereo"]
        assert prompt.rendered_text_antistereo == item.payload["sent_antistereo"]
        assert prompt.rendered_text == "", "a paired prompt has no single rendered model input"
        assert prompt.is_paired


def test_the_sequences_are_copied_verbatim(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    """Nothing is composed, normalised or rewritten.

    Rewriting a benchmark's sentences -- even the punctuation -- would mean scoring a text the
    published dataset does not contain, and the comparison to prior work would be to a number
    nobody else can reproduce.
    """
    prompts = {p.item_id: p for p in renderer.render(items, lattice.cells())}
    for item in items:
        prompt = prompts[item.id]
        assert prompt.rendered_text_stereo in (item.payload["sent_more"], item.payload["sent_less"])
        assert prompt.rendered_text_antistereo in (
            item.payload["sent_more"],
            item.payload["sent_less"],
        )


def test_an_item_attesting_nothing_gets_a_null_cell_not_an_invented_one(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    prompts = {p.item_id: p for p in renderer.render(items, lattice.cells())}
    religion = next(i for i in items if i.payload["bias_type"] == "religion")
    assert prompts[religion.id].cell_id is None
    assert "nocell" in prompts[religion.id].prompt_id


# --------------------------------------------------------------------------------------
# GATE P4b: no cell unattested in the source item
# --------------------------------------------------------------------------------------


def test_every_emitted_cell_is_attested_in_its_own_sentences(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    """The gate, checked positively across everything the renderer emits."""
    by_id = {cell.cell_id: cell for cell in lattice.cells()}
    surfaces = {(v.axis, v.value): v.surface for v in lattice.values()}

    for prompt in renderer.render(items, lattice.cells()):
        if prompt.cell_id is None:
            continue
        cell = by_id[prompt.cell_id]
        text = f"{prompt.rendered_text_stereo} {prompt.rendered_text_antistereo}".lower()
        for axis, value in cell.axes:
            assert surfaces[(axis, value)].lower() in text, (
                f"{prompt.prompt_id} claims {axis}={value}, which its sentences do not contain"
            )


def test_a_tampered_payload_claiming_an_unattested_cell_raises(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    """The guarantee checks the TEXT, not the payload that claims the cell.

    A payload is an assertion. If the renderer trusted it, a loader bug -- or a hand-edited item
    file -- would put a fabricated cell into the prompt set and nothing downstream would notice,
    because `cell_id` is an opaque string everywhere after this point.
    """
    victim = next(i for i in items if i.payload["bias_type"] == "religion")
    tampered = _retarget(victim, attested=(("race_ethnicity", "Black"),))

    with pytest.raises(ValueError, match="neither of its sentences attests"):
        renderer.render([tampered], lattice.cells())


def test_a_second_axis_present_in_only_one_sentence_is_refused(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    """An order-2 cell needs one CONTRASTED axis and one SHARED axis.

    If the second value appears in only one member, it is part of the contrast rather than shared
    context -- the pair is not minimal in that axis, and attributing the likelihood difference to
    a cell only half the pair is in would be exactly the unattested assignment this gate forbids.
    """
    victim = next(i for i in items if i.payload["row_index"] == 0)
    tampered = _retarget(
        victim,
        sent_stereo="The Black female worker arrived.",
        sent_antistereo="The white worker arrived.",
        attested=(("race_ethnicity", "Black"), ("gender_and_sex", "female")),
    )
    with pytest.raises(ValueError, match="neither value appears in BOTH sentences"):
        renderer.render([tampered], lattice.cells())


def test_a_cell_the_lattice_does_not_declare_raises(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    """A renderer may not mint a cell. The metric layer resolves cell_id through the lattice and
    raises for one it cannot place, which would take down the whole run including other sources."""
    victim = items[0]
    tampered = _retarget(victim, attested=(("race_ethnicity", "Martian"),))
    with pytest.raises(ValueError, match="resolves to no declared cell"):
        renderer.render([tampered], lattice.cells())


def test_an_item_attesting_a_cell_excluded_from_the_call_is_refused(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    """A disagreement between the caller's cell list and the item is surfaced, not silently
    dropped."""
    celled = [i for i in items if i.payload["attested"]]
    assert celled
    narrowed = [c for c in lattice.cells() if c.cell_id != "race_ethnicity=Black"]
    with pytest.raises(ValueError, match="not in the cell list passed to render"):
        renderer.render(celled, narrowed)


def test_renderer_refuses_an_open_gen_item(
    renderer: ExtractiveRenderer, lattice: DeclaredLattice, provenance: object
) -> None:
    """The mirror of TemplateRenderer refusing an extractive item."""
    from margins.schemas import Provenance

    assert isinstance(provenance, Provenance)
    item = Item(
        id="tmpl-0",
        benchmark="crows_pairs",
        task_type=TaskType.OPEN_GEN,
        payload={"sent_stereo": "a", "sent_antistereo": "b", "attested": ()},
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="TemplateRenderer"):
        renderer.render([item], lattice.cells())


def test_renderer_refuses_an_item_from_another_benchmark(
    renderer: ExtractiveRenderer, lattice: DeclaredLattice, provenance: object
) -> None:
    from margins.schemas import Provenance

    assert isinstance(provenance, Provenance)
    item = Item(
        id="wino-0",
        benchmark="winobias",
        task_type=TaskType.PAIR_PLL,
        payload={"sent_stereo": "a", "sent_antistereo": "b", "attested": ()},
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="configured for"):
        renderer.render([item], lattice.cells())


# --------------------------------------------------------------------------------------
# Determinism (gate P4b, second clause)
# --------------------------------------------------------------------------------------


def test_regeneration_is_byte_identical(
    prompt_config: ExtractivePromptSetConfig,
    lattice: DeclaredLattice,
    items: list[Item],
) -> None:
    first = render_prompts_jsonl(
        ExtractiveRenderer(prompt_config, lattice).render(items, lattice.cells())
    ).encode("utf-8")
    second = render_prompts_jsonl(
        ExtractiveRenderer(prompt_config, lattice).render(items, lattice.cells())
    ).encode("utf-8")

    assert first == second
    assert b"\r\n" not in first, "CRLF would make the bytes differ between the two environments"


def test_the_prompt_id_digest_covers_both_sequences(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    """Load-bearing: `make_response_id`'s preimage does NOT cover the scored texts, so `prompt_id`
    is the only place a pair's identity can live. Two pairs sharing an item would otherwise
    collide on one response id."""
    original = items[0]
    changed = _retarget(original, sent_antistereo="A completely different counterpart sentence.")

    first = renderer.render([original], lattice.cells())[0]
    second = renderer.render([changed], lattice.cells())[0]
    assert first.prompt_id != second.prompt_id


def test_the_prompt_id_is_a_readable_natural_key_plus_a_digest(
    renderer: ExtractiveRenderer, items: list[Item], lattice: DeclaredLattice
) -> None:
    for prompt in renderer.render(items, lattice.cells()):
        parts = prompt.prompt_id.split("::")
        assert len(parts) == 4, prompt.prompt_id
        set_key, cell, item_id, digest = parts
        assert set_key == DEFAULT_EXTRACTIVE_SET
        assert cell == (prompt.cell_id or "nocell")
        assert item_id == prompt.item_id
        assert len(digest) == 12


# --------------------------------------------------------------------------------------
# Open decision 7: the contract change
# --------------------------------------------------------------------------------------


def test_a_paired_prompt_must_carry_both_sequences() -> None:
    with pytest.raises(ValueError, match="carries BOTH"):
        Prompt(
            prompt_id="p",
            item_id="i",
            rendered_text="",
            task_type=TaskType.PAIR_PLL,
            template_version="v",
            decoding_key="pll_v1",
            rendered_text_stereo="only one half",
        )


def test_an_open_gen_prompt_must_leave_both_sequences_null() -> None:
    with pytest.raises(ValueError, match="must leave rendered_text_stereo"):
        Prompt(
            prompt_id="p",
            item_id="i",
            rendered_text="Describe a person.",
            task_type=TaskType.OPEN_GEN,
            template_version="v",
            decoding_key="greedy_v1",
            rendered_text_stereo="a",
            rendered_text_antistereo="b",
        )


def test_a_paired_prompt_refuses_a_rendered_text() -> None:
    """The two sequences ARE the input, one per forward pass. Mirrors `Response.text == ""`."""
    with pytest.raises(ValueError, match="no single rendered model input"):
        Prompt(
            prompt_id="p",
            item_id="i",
            rendered_text="something",
            task_type=TaskType.PAIR_PLL,
            template_version="v",
            decoding_key="pll_v1",
            rendered_text_stereo="a",
            rendered_text_antistereo="b",
        )


def test_a_paired_prompt_refuses_two_identical_sequences() -> None:
    """Their log-probability difference would be zero by construction and measure nothing."""
    with pytest.raises(ValueError, match="identical"):
        Prompt(
            prompt_id="p",
            item_id="i",
            rendered_text="",
            task_type=TaskType.PAIR_PLL,
            template_version="v",
            decoding_key="pll_v1",
            rendered_text_stereo="same",
            rendered_text_antistereo="same",
        )


def test_the_extractive_prompt_set_cannot_be_loaded_as_an_open_gen_one() -> None:
    """`PromptSetConfig` types task_type as Literal["open_gen"] so a paired set fails to LOAD."""
    from margins.prompts import load_prompt_config

    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError or Hydra error
        load_prompt_config(DEFAULT_EXTRACTIVE_SET)
