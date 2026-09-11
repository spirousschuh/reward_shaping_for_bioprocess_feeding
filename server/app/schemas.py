"""Request / response shapes. The submit request deliberately has no score or timestamp
field, so a client cannot smuggle one in."""

from datetime import datetime

from pydantic import BaseModel, Field


class SubmitRequest(BaseModel):
    team: str = Field(min_length=1, max_length=120)
    ranking: list[str] = Field(min_length=1)


class SubmitAccepted(BaseModel):
    accepted: bool = True
    team: str
    kendall_tau: float
    leaderboard_position: int
    submissions_used: int
    max_submissions: int
    cooldown_seconds: int


class LeaderboardRow(BaseModel):
    rank: int
    team: str
    kendall_tau: float


class AdminTeam(BaseModel):
    id: int
    name: str
    normalized_name: str
    created_at: datetime
    submission_count: int


class AdminSubmission(BaseModel):
    id: int
    team: str
    ranking: list[str]
    kendall_tau: float
    created_at: datetime


class CompetitionToggle(BaseModel):
    enabled: bool


class ResetRequest(BaseModel):
    confirm: str
