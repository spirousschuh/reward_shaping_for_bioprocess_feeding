"""Database tables: teams, submissions, and the competition on/off switch."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_team_name(name: str) -> str:
    """Collapse whitespace and lowercase, so 'Team Alpha' == '  team   alpha '."""
    return " ".join(str(name).split()).lower()


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)          # display name
    normalized_name: Mapped[str] = mapped_column(
        String(120), nullable=False, unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )

    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ranking_json: Mapped[str] = mapped_column(Text, nullable=False)
    kendall_tau: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, index=True
    )

    team: Mapped[Team] = relationship(back_populates="submissions")


class CompetitionState(Base):
    __tablename__ = "competition_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # always 1
    submissions_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
