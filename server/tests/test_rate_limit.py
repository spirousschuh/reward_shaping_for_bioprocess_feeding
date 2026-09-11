"""Server-side cooldown and per-team submission cap (spec §22)."""

from conftest import correct_order


def _post(client, team, ranking=None):
    return client.post("/api/submit", json={"team": team, "ranking": ranking or correct_order()})


def test_first_submission_accepted(client):
    assert _post(client, "RL A").status_code == 200


def test_immediate_second_submission_rate_limited(client):
    assert _post(client, "RL B").status_code == 200
    r = _post(client, "RL B")
    assert r.status_code == 429
    body = r.json()
    assert body["accepted"] is False
    assert body["error"] == "rate_limited"
    assert body["retry_after_seconds"] > 0


def test_submission_after_cooldown_accepted(client, relax_limits):
    assert _post(client, "RL C").status_code == 200
    assert _post(client, "RL C").status_code == 200  # cooldown == 0 now


def test_submission_limit_enforced(client, relax_limits):
    for _ in range(3):  # MAX_SUBMISSIONS_PER_TEAM == 3 under relax_limits
        assert _post(client, "RL D").status_code == 200
    r = _post(client, "RL D")
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "submission_limit_reached"
    assert body["submissions_used"] == 3
    assert body["max_submissions"] == 3


def test_teams_have_independent_limits(client, relax_limits):
    for _ in range(3):
        assert _post(client, "RL E").status_code == 200
    assert _post(client, "RL E").status_code == 429
    # a different team is unaffected
    assert _post(client, "RL F").status_code == 200
