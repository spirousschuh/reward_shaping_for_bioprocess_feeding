"""Leaderboard queries: each team's best validation tau, ranked."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Submission, Team


def _best_per_team(session: Session) -> list[tuple[int, str, float]]:
    """``[(team_id, display_name, best_tau), ...]`` ordered best tau first."""
    stmt = (
        select(Team.id, Team.name, func.max(Submission.kendall_tau).label("best"))
        .join(Submission, Submission.team_id == Team.id)
        .group_by(Team.id)
        .order_by(func.max(Submission.kendall_tau).desc(), Team.name.asc())
    )
    return [(tid, name, float(best)) for tid, name, best in session.execute(stmt).all()]


def rows(session: Session) -> list[dict]:
    return [
        {"rank": i + 1, "team": name, "kendall_tau": tau}
        for i, (_, name, tau) in enumerate(_best_per_team(session))
    ]


def team_position(session: Session, team_id: int) -> int:
    """1-based leaderboard position of ``team_id`` by its best tau (0 if no submissions)."""
    for i, (tid, _, _) in enumerate(_best_per_team(session)):
        if tid == team_id:
            return i + 1
    return 0
