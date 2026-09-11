"""SQLAlchemy engine / session wiring.

SQLite by default; set ``DATABASE_URL`` to a PostgreSQL URL to switch backends without code
changes. On Render, point it at a file on the mounted persistent disk so data survives
redeploys.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine(url: str):
    kwargs = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        # create the parent directory for a file-backed sqlite db if it is missing
        prefix = "sqlite:///"
        if url.startswith(prefix) and not url.startswith(prefix + ":memory:"):
            db_path = Path(url[len(prefix):])
            if db_path.parent and not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, **kwargs)


engine = _make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create tables and ensure the single competition-state row exists."""
    from . import models  # noqa: F401  (register mappers)

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        state = session.get(models.CompetitionState, 1)
        if state is None:
            session.add(models.CompetitionState(id=1, submissions_enabled=True))
            session.commit()


def get_session():
    """FastAPI dependency: a session per request."""
    with SessionLocal() as session:
        yield session
