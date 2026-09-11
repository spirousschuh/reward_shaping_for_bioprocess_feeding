"""The hidden validation expert ranking — materialised at runtime from the environment.

``VALIDATION_EXPERT_RANKING`` (see :mod:`app.config`) is the only source. It never lives in
a file in this repository. In development, if it is unset, a clearly-fake ``DUMMY_RANKING``
is used and a loud warning is logged; in production an unset value makes the server refuse
to start.
"""

import json
import logging

from .config import get_settings
from .scoring import VALIDATION_IDS

logger = logging.getLogger("uvicorn.error")

# NOT the real order. Ascending numeric — used only so local dev / tests can run without the
# real secret. Production must supply VALIDATION_EXPERT_RANKING.
DUMMY_RANKING = list(VALIDATION_IDS)


def _parse_groups(raw: str) -> list[list[str]]:
    """Parse the env value into a list of tie groups (each a list of id strings)."""
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("VALIDATION_EXPERT_RANKING must be a non-empty JSON list")

    groups = []
    for entry in data:
        group = entry if isinstance(entry, list) else [entry]
        groups.append([str(x).strip() for x in group])
    return groups


def _rank_map_from_groups(groups: list[list[str]]) -> dict[str, float]:
    """Average-rank map: [[a],[b,c],[d]] -> {a:0, b:1.5, c:1.5, d:3}."""
    flat = [eid for group in groups for eid in group]
    if sorted(flat) != sorted(VALIDATION_IDS):
        raise ValueError(
            "VALIDATION_EXPERT_RANKING must be exactly the five validation ids, each once"
        )

    rank_map: dict[str, float] = {}
    position = 0
    for group in groups:
        avg = position + (len(group) - 1) / 2.0
        for eid in group:
            rank_map[eid] = avg
        position += len(group)
    return rank_map


def load_expert_rank_map() -> dict[str, float]:
    """The ``{id: rank}`` map used for scoring. Raises in production if unconfigured."""
    settings = get_settings()
    raw = (settings.validation_expert_ranking or "").strip()

    if not raw:
        if settings.app_env == "production":
            raise RuntimeError(
                "VALIDATION_EXPERT_RANKING is not set; refusing to start in production"
            )
        logger.warning(
            "VALIDATION_EXPERT_RANKING is not set -- using DUMMY ranking %s. "
            "DO NOT run a real competition like this.", DUMMY_RANKING,
        )
        return _rank_map_from_groups([[eid] for eid in DUMMY_RANKING])

    return _rank_map_from_groups(_parse_groups(raw))
