"""Content-addressed ``score_id`` (P6).

Mirrors :func:`margins.prompts.render.make_prompt_id` and
:func:`margins.inference.backend.make_response_id`: a readable natural key a human can grep, plus a
short digest over everything that determines the value.

**What the digest covers, and why it matters here specifically.** A score is a function of the
response *and* of the instrument. Re-pin a classifier, flip a sign, move a threshold, and the same
``(scorer, response)`` pair produces a different number -- so if the id did not move, a re-scored
run would append rows that collide with the old ones and the score table would silently hold two
incompatible measurements under one key. The scorer's :attr:`~margins.scoring.config.ScorerSpec.
pin_digest` and the roster's ``scoring_hash`` are therefore both in the preimage.

``condition`` is in the preimage too: the drift control scores different text through the same
scorer, and the two conditions must never collide.
"""

from __future__ import annotations

import hashlib
from typing import Final

from margins.schemas import Condition

__all__ = ["SCORE_ID_DIGEST_LENGTH", "make_score_id"]

#: Matches the 12 used by ``prompt_id`` and ``response_id``.
SCORE_ID_DIGEST_LENGTH: Final[int] = 12

#: Separator inside a ``score_id``. Not ``|``, which already appears inside every order-2 cell id.
_SCORE_ID_SEPARATOR: Final[str] = "::"

#: Field separator for the digest preimage. A control character so no field value can forge one.
_DIGEST_SEPARATOR: Final[str] = "\x1f"


def make_score_id(
    *,
    response_id: str,
    scorer_id: str,
    harm_type: str,
    condition: Condition,
    pin_digest: str,
    scoring_hash: str,
) -> str:
    """A stable, deterministic, content-addressed ``score_id``.

    Shape: ``<scorer_id>::<condition>::<response_id>::<digest12>``.

    Nothing here depends on position in a list, so ids are identical no matter what order the
    responses arrived in or how they were batched.
    """
    preimage = _DIGEST_SEPARATOR.join(
        (scorer_id, harm_type, condition, response_id, pin_digest, scoring_hash)
    )
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:SCORE_ID_DIGEST_LENGTH]
    return _SCORE_ID_SEPARATOR.join((scorer_id, condition, response_id, digest))
