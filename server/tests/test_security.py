"""Responses must never leak the hidden expert ranking (spec §21, §22)."""

import json

from conftest import ADMIN_TOKEN, DUMMY_EXPERT_ORDER, correct_order

# The (dummy) expert order rendered as a contiguous string — must not appear verbatim.
_EXPERT_SEQUENCE = ",".join(DUMMY_EXPERT_ORDER)


def _no_expert_leak(blob: str):
    assert _EXPERT_SEQUENCE not in blob
    lowered = blob.lower()
    assert "expert" not in lowered
    assert "validation_expert_ranking" not in lowered


def test_submit_response_has_no_expert_data(client):
    r = client.post("/api/submit", json={"team": "Sec", "ranking": correct_order()})
    _no_expert_leak(json.dumps(r.json()))


def test_error_response_has_no_expert_data(client):
    r = client.post("/api/submit", json={"team": "Sec", "ranking": ["1", "2", "3", "4", "5"]})
    _no_expert_leak(json.dumps(r.json()))


def test_leaderboard_has_no_expert_data(client, relax_limits):
    client.post("/api/submit", json={"team": "Sec", "ranking": correct_order()})
    _no_expert_leak(json.dumps(client.get("/api/leaderboard").json()))
    _no_expert_leak(client.get("/leaderboard").text)


def test_admin_submissions_exposes_own_ranking_but_not_expert_order(client):
    client.post("/api/submit", json={"team": "Sec", "ranking": correct_order()})
    r = client.get("/api/admin/submissions", headers={"X-Admin-Token": ADMIN_TOKEN})
    assert r.status_code == 200
    payload = r.json()
    # a team's own submitted ranking is visible to the admin ...
    assert payload[0]["ranking"] == correct_order()
    # ... but nothing labels it as the expert order, and there is no "expert" key
    for row in payload:
        assert "expert" not in json.dumps(row).lower()


def test_admin_requires_token(client):
    assert client.get("/api/admin/submissions").status_code == 401
    assert client.get(
        "/api/admin/submissions", headers={"X-Admin-Token": "wrong"}
    ).status_code == 401


def test_competition_can_be_disabled(client):
    off = client.post("/api/admin/competition", json={"enabled": False},
                      headers={"X-Admin-Token": ADMIN_TOKEN})
    assert off.status_code == 200

    r = client.post("/api/submit", json={"team": "Late", "ranking": correct_order()})
    assert r.status_code == 403
    assert r.json()["error"] == "submissions_disabled"

    client.post("/api/admin/competition", json={"enabled": True},
                headers={"X-Admin-Token": ADMIN_TOKEN})
    assert client.post(
        "/api/submit", json={"team": "Late", "ranking": correct_order()}
    ).status_code == 200
