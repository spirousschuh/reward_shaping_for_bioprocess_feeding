"""Runtime configuration, entirely from environment variables.

Nothing here is a secret except by what the deployment puts in the environment. The hidden
validation expert ranking arrives as ``VALIDATION_EXPERT_RANKING`` and is parsed in
:mod:`app.secret`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "development" | "production". In production the server refuses to start without a
    # real VALIDATION_EXPERT_RANKING (see app.secret).
    app_env: str = "development"

    # SQLAlchemy URL. Default is a file next to the app; on Render point this at a path on
    # the mounted persistent disk, e.g. sqlite:////var/data/challenge.db
    database_url: str = "sqlite:///./server_data/challenge.db"

    # Admin API auth. Empty => admin endpoints are disabled (503).
    admin_token: str = ""

    # Submission limits (server-enforced).
    submission_cooldown_seconds: int = 60
    max_submissions_per_team: int = 20

    # The hidden validation expert ranking, as JSON. Either a flat list
    #   ["<id>", "<id>", "<id>", "<id>", "<id>"]            (best first)
    # or a grouped list where an inner list is a tie group
    #   [["<id>"], ["<id>", "<id>"], ["<id>"], ["<id>"]]
    # It must be exactly the five ids 16736 16738 16742 16748 17191, each once.
    # Empty in development => a clearly-fake DUMMY order is used (with a warning).
    validation_expert_ranking: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
