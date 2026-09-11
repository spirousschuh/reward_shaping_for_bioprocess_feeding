"""Leaderboard aggregation (spec §22)."""

from conftest import DUMMY_EXPERT_ORDER, correct_order


def submit(client, team, ranking):
    return client.post("/api/submit", json={"team": team, "ranking": ranking})


def _worse_order():
    """A ranking with tau < 1 (one adjacent swap of the dummy expert order)."""
    o = list(DUMMY_EXPERT_ORDER)
    o[0], o[1] = o[1], o[0]
    return o


def test_only_best_submission_per_team(client, relax_limits):
    assert submit(client, "Alpha", _worse_order()).json()["kendall_tau"] == 0.8
    assert submit(client, "Alpha", correct_order()).json()["kendall_tau"] == 1.0
    assert submit(client, "Alpha", _worse_order()).json()["kendall_tau"] == 0.8

    board = client.get("/api/leaderboard").json()
    alpha = [r for r in board if r["team"] == "Alpha"]
    assert len(alpha) == 1
    assert alpha[0]["kendall_tau"] == 1.0


def test_sorted_by_tau_descending(client, relax_limits):
    submit(client, "HighTeam", correct_order())          # 1.0
    submit(client, "LowTeam", list(reversed(correct_order())))  # -1.0
    submit(client, "MidTeam", _worse_order())            # 0.8

    board = client.get("/api/leaderboard").json()
    assert [r["team"] for r in board] == ["HighTeam", "MidTeam", "LowTeam"]
    assert [r["rank"] for r in board] == [1, 2, 3]


def test_team_name_normalization(client, relax_limits):
    submit(client, "Team Alpha", correct_order())
    submit(client, "  team   alpha ", _worse_order())
    submit(client, "TEAM ALPHA", _worse_order())

    board = client.get("/api/leaderboard").json()
    assert len(board) == 1                      # all three are one team
    assert board[0]["team"] == "Team Alpha"     # first-seen display name
    assert board[0]["kendall_tau"] == 1.0


def test_teams_do_not_interfere(client, relax_limits):
    submit(client, "T1", correct_order())
    submit(client, "T2", _worse_order())
    board = {r["team"]: r["kendall_tau"] for r in client.get("/api/leaderboard").json()}
    assert board == {"T1": 1.0, "T2": 0.8}
