"""Submitted-ranking validation (spec §22)."""

from conftest import correct_order


def _post(client, ranking, team="Team V"):
    return client.post("/api/submit", json={"team": team, "ranking": ranking})


def test_correct_five_accepted(client):
    r = _post(client, correct_order())
    assert r.status_code == 200
    assert r.json()["accepted"] is True


def test_missing_experiment_rejected(client):
    r = _post(client, ["16736", "16738", "16742", "16748"])  # only 4
    assert r.status_code == 422
    assert r.json() == {"accepted": False, "error": "invalid_ranking"}


def test_six_experiments_rejected(client):
    r = _post(client, ["16736", "16738", "16742", "16748", "17191", "16737"])
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_ranking"


def test_duplicate_experiment_rejected(client):
    r = _post(client, ["16736", "16736", "16742", "16748", "17191"])
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_ranking"


def test_unknown_experiment_rejected(client):
    r = _post(client, ["16736", "16738", "16742", "16748", "99999"])
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_ranking"


def test_empty_team_rejected(client):
    r = client.post("/api/submit", json={"team": "", "ranking": correct_order()})
    assert r.status_code == 422  # pydantic min_length


def test_client_supplied_score_is_ignored(client):
    r = client.post("/api/submit", json={
        "team": "Cheater", "ranking": correct_order(), "score": 1.0, "kendall_tau": 1.0,
    })
    assert r.status_code == 200
    # score comes from the server, not the payload; correct_order == dummy expert => 1.0
    assert r.json()["kendall_tau"] == 1.0
