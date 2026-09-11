"""Validation-ranking checks and the Kendall tau-b metric.

The five validation experiment ids are public; their *order* is not (that lives in
:mod:`app.secret`, sourced from the environment).
"""

from scipy.stats import kendalltau

# Public: the five validation experiment ids, ascending numeric order (no ranking meaning).
VALIDATION_IDS = ["16736", "16738", "16742", "16748", "17191"]
_VALIDATION_ID_SET = set(VALIDATION_IDS)


class SubmissionError(ValueError):
    """A submitted ranking is malformed. Messages are generic and leak no expert info."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_submitted_ranking(ranking) -> list[str]:
    """Return ``ranking`` as a list of 5 unique validation id strings, or raise.

    :raises SubmissionError: not a list, wrong length, unknown id, or a duplicate.
    """
    if not isinstance(ranking, (list, tuple)):
        raise SubmissionError("invalid_ranking", "ranking must be a list of experiment ids")

    items = [str(x).strip() for x in ranking]

    if len(items) != len(VALIDATION_IDS):
        raise SubmissionError(
            "invalid_ranking",
            "ranking must contain exactly %d experiment ids" % len(VALIDATION_IDS),
        )
    if len(set(items)) != len(items):
        raise SubmissionError("invalid_ranking", "ranking contains a duplicate experiment id")
    if any(x not in _VALIDATION_ID_SET for x in items):
        raise SubmissionError("invalid_ranking", "ranking contains an unknown experiment id")

    return items


def kendall_tau_b(submitted: list[str], expert_rank_map: dict[str, float]) -> float:
    """Kendall's tau-b between a submitted order and the expert's (possibly tied) ranks.

    :param submitted: the five ids, best first (already validated).
    :param expert_rank_map: ``{id: rank}`` where tied experiments share an averaged rank.
    :returns: tau-b in ``[-1, 1]``; SciPy's default variant, which handles the ties.
    """
    ids = list(submitted)
    submitted_rank = {eid: i for i, eid in enumerate(ids)}
    x = [submitted_rank[eid] for eid in ids]
    y = [expert_rank_map[eid] for eid in ids]
    return float(kendalltau(x, y).statistic)
