"""Kendall tau-b scoring (spec §22)."""

import pytest
from scipy.stats import kendalltau

from app.scoring import kendall_tau_b
from conftest import DUMMY_EXPERT_ORDER, correct_order


def _flat_map(order):
    return {eid: i for i, eid in enumerate(order)}


def test_identical_ranking_is_one():
    m = _flat_map(DUMMY_EXPERT_ORDER)
    assert kendall_tau_b(list(DUMMY_EXPERT_ORDER), m) == pytest.approx(1.0)


def test_reversed_ranking_is_minus_one():
    m = _flat_map(DUMMY_EXPERT_ORDER)
    assert kendall_tau_b(list(reversed(DUMMY_EXPERT_ORDER)), m) == pytest.approx(-1.0)


def test_one_adjacent_swap():
    m = _flat_map(DUMMY_EXPERT_ORDER)
    swapped = list(DUMMY_EXPERT_ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    # 5 items -> 10 pairs, one discordant => (9-1)/10
    assert kendall_tau_b(swapped, m) == pytest.approx(0.8)


# A synthetic tie map — NOT the competition ranking, just enough to exercise tau-b with a
# tied pair (16742 / 16748, the two the expert could not separate).
_TIE_MAP = {"16736": 0.0, "17191": 1.0, "16742": 2.5, "16748": 2.5, "16738": 4.0}


def test_tie_16742_16748_orientation_does_not_matter():
    a = kendall_tau_b(["16736", "17191", "16742", "16748", "16738"], _TIE_MAP)
    b = kendall_tau_b(["16736", "17191", "16748", "16742", "16738"], _TIE_MAP)
    assert a == pytest.approx(b)


def test_tie_matches_scipy_reference():
    order = ["17191", "16736", "16742", "16748", "16738"]
    x = [order.index(e) for e in order]
    y = [_TIE_MAP[e] for e in order]
    assert kendall_tau_b(order, _TIE_MAP) == pytest.approx(float(kendalltau(x, y).statistic))


def test_api_scores_against_hidden_order(client):
    r = client.post("/api/submit", json={"team": "S", "ranking": correct_order()})
    assert r.json()["kendall_tau"] == 1.0
    r2 = client.post("/api/submit", json={
        "team": "S2", "ranking": list(reversed(correct_order()))})
    assert r2.json()["kendall_tau"] == -1.0
