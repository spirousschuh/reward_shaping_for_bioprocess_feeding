"""FastAPI app: submission endpoint, public leaderboard, protected admin API.

The hidden validation expert ranking is loaded once at startup (from the environment, via
:mod:`app.secret`) and kept in memory. It is never written to the database and never
returned by any endpoint — not even the admin ones.
"""

import json
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import leaderboard as lb
from .config import get_settings
from .database import get_session, init_db
from .models import CompetitionState, Submission, Team, normalize_team_name
from .schemas import (
    AdminSubmission, AdminTeam, CompetitionToggle, LeaderboardRow, ResetRequest,
    SubmitAccepted, SubmitRequest,
)
from .scoring import SubmissionError, kendall_tau_b, validate_submitted_ranking
from .secret import load_expert_rank_map

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_EXPERT_RANK_MAP: dict[str, float] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _EXPERT_RANK_MAP
    init_db()
    _EXPERT_RANK_MAP = load_expert_rank_map()  # raises in production if unconfigured
    yield


app = FastAPI(title="E. coli Reward Ranking Challenge", lifespan=lifespan)


# --------------------------------------------------------------------------- utils

def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _competition_enabled(session: Session) -> bool:
    state = session.get(CompetitionState, 1)
    return bool(state.submissions_enabled) if state else True


def _require_admin(x_admin_token: str | None) -> None:
    token = get_settings().admin_token
    if not token:
        raise HTTPException(status_code=503, detail="admin API disabled")
    if x_admin_token != token:
        raise HTTPException(status_code=401, detail="invalid admin token")


# -------------------------------------------------------------------------- public

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Nothing lives at the bare root; send a browser straight to the leaderboard."""
    return RedirectResponse(url="/leaderboard")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/api/submit")
def submit(payload: SubmitRequest, request: Request,
           session: Session = Depends(get_session)) -> JSONResponse:
    settings = get_settings()

    if not _competition_enabled(session):
        return JSONResponse(
            status_code=403,
            content={"accepted": False, "error": "submissions_disabled"},
        )

    try:
        ranking = validate_submitted_ranking(payload.ranking)
    except SubmissionError as err:
        return JSONResponse(
            status_code=422, content={"accepted": False, "error": err.code}
        )

    # get-or-create team by normalized name; keep the first-seen display name
    normalized = normalize_team_name(payload.team)
    team = session.execute(
        select(Team).where(Team.normalized_name == normalized)
    ).scalar_one_or_none()
    if team is None:
        team = Team(name=payload.team.strip(), normalized_name=normalized)
        session.add(team)
        session.flush()

    used = session.execute(
        select(func.count(Submission.id)).where(Submission.team_id == team.id)
    ).scalar_one()

    if used >= settings.max_submissions_per_team:
        session.commit()
        return JSONResponse(
            status_code=429,
            content={
                "accepted": False,
                "error": "submission_limit_reached",
                "submissions_used": used,
                "max_submissions": settings.max_submissions_per_team,
            },
        )

    last = session.execute(
        select(func.max(Submission.created_at)).where(Submission.team_id == team.id)
    ).scalar_one()
    if last is not None:
        elapsed = (datetime.now(timezone.utc) - _as_utc(last)).total_seconds()
        remaining = settings.submission_cooldown_seconds - elapsed
        if remaining > 0:
            session.commit()
            return JSONResponse(
                status_code=429,
                content={
                    "accepted": False,
                    "error": "rate_limited",
                    "retry_after_seconds": int(math.ceil(remaining)),
                },
            )

    tau = round(kendall_tau_b(ranking, _EXPERT_RANK_MAP), 6)

    session.add(Submission(
        team_id=team.id, ranking_json=json.dumps(ranking), kendall_tau=tau,
    ))
    session.commit()

    body = SubmitAccepted(
        team=team.name,
        kendall_tau=round(tau, 4),
        leaderboard_position=lb.team_position(session, team.id),
        submissions_used=used + 1,
        max_submissions=settings.max_submissions_per_team,
        cooldown_seconds=settings.submission_cooldown_seconds,
    )
    return JSONResponse(status_code=200, content=body.model_dump())


@app.get("/api/leaderboard", response_model=list[LeaderboardRow])
def api_leaderboard(session: Session = Depends(get_session)) -> list[dict]:
    return lb.rows(session)


@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page(request: Request,
                     session: Session = Depends(get_session)) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "leaderboard.html",
        {"rows": lb.rows(session), "refresh_seconds": 5},
    )


# --------------------------------------------------------------------------- admin

@app.get("/api/admin/teams", response_model=list[AdminTeam])
def admin_teams(x_admin_token: str | None = Header(default=None),
                session: Session = Depends(get_session)) -> list[dict]:
    _require_admin(x_admin_token)
    out = []
    for team in session.execute(select(Team).order_by(Team.created_at)).scalars():
        count = session.execute(
            select(func.count(Submission.id)).where(Submission.team_id == team.id)
        ).scalar_one()
        out.append({
            "id": team.id, "name": team.name, "normalized_name": team.normalized_name,
            "created_at": team.created_at, "submission_count": count,
        })
    return out


@app.get("/api/admin/submissions", response_model=list[AdminSubmission])
def admin_submissions(x_admin_token: str | None = Header(default=None),
                      session: Session = Depends(get_session)) -> list[dict]:
    _require_admin(x_admin_token)
    stmt = (
        select(Submission, Team.name)
        .join(Team, Team.id == Submission.team_id)
        .order_by(Submission.created_at)
    )
    return [
        {
            "id": sub.id, "team": name, "ranking": json.loads(sub.ranking_json),
            "kendall_tau": sub.kendall_tau, "created_at": sub.created_at,
        }
        for sub, name in session.execute(stmt).all()
    ]


@app.post("/api/admin/competition")
def admin_competition(payload: CompetitionToggle,
                      x_admin_token: str | None = Header(default=None),
                      session: Session = Depends(get_session)) -> dict:
    _require_admin(x_admin_token)
    state = session.get(CompetitionState, 1) or CompetitionState(id=1)
    state.submissions_enabled = payload.enabled
    session.add(state)
    session.commit()
    return {"submissions_enabled": state.submissions_enabled}


@app.post("/api/admin/reset")
def admin_reset(payload: ResetRequest,
                x_admin_token: str | None = Header(default=None),
                session: Session = Depends(get_session)) -> dict:
    _require_admin(x_admin_token)
    if payload.confirm != "reset":
        raise HTTPException(status_code=400, detail='send {"confirm": "reset"} to confirm')
    n_sub = session.query(Submission).delete()
    n_team = session.query(Team).delete()
    session.commit()
    return {"deleted_submissions": n_sub, "deleted_teams": n_team}
