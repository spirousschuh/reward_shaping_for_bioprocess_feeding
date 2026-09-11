"""Shared test fixtures.

Environment is set *before* the app modules import, so the engine and settings pick up a
throwaway SQLite file and a DUMMY validation order that the tests know.
"""

import os
import tempfile

import pytest

# --- must run before `app.*` is imported ------------------------------------------------
_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="ecoli-challenge-test-"), "test.db")

os.environ["APP_ENV"] = "development"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["ADMIN_TOKEN"] = "test-admin-token"
os.environ["SUBMISSION_COOLDOWN_SECONDS"] = "60"
os.environ["MAX_SUBMISSIONS_PER_TEAM"] = "20"
# Flat (no-tie) dummy order the API-level tests score against. NOT the real ranking.
os.environ["VALIDATION_EXPERT_RANKING"] = '["16736","16738","16742","16748","17191"]'
# -------------------------------------------------------------------------------------- -

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app import models  # noqa: E402

ADMIN_TOKEN = "test-admin-token"
# The order the conftest env installs as the (dummy) expert ranking.
DUMMY_EXPERT_ORDER = ["16736", "16738", "16742", "16748", "17191"]


@pytest.fixture()
def client():
    with TestClient(app) as c:  # runs startup: init_db + load_expert_rank_map
        yield c


@pytest.fixture(autouse=True)
def _clean_db():
    """Empty every table before each test."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as s:
        s.query(models.Submission).delete()
        s.query(models.Team).delete()
        state = s.get(models.CompetitionState, 1)
        if state is None:
            s.add(models.CompetitionState(id=1, submissions_enabled=True))
        else:
            state.submissions_enabled = True
        s.commit()
    yield


@pytest.fixture()
def relax_limits():
    """Set cooldown=0 and a small max, for tests that submit several times quickly."""
    old = dict(os.environ)
    os.environ["SUBMISSION_COOLDOWN_SECONDS"] = "0"
    os.environ["MAX_SUBMISSIONS_PER_TEAM"] = "3"
    get_settings.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(old)
    get_settings.cache_clear()


def correct_order():
    """A valid 5-id ranking equal to the dummy expert order (tau == 1.0)."""
    return list(DUMMY_EXPERT_ORDER)
