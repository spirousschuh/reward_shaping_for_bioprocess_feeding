"""Reward shaping for bioprocess feeding --- summer-school ranking challenge.

Design a reward function that scores fed-batch *E. coli* cultivations. The challenge has two
stages:

* **Training** --- six reactors, expert order known (:data:`TRAINING_EXPERT_RANKING`).
  Score locally with :func:`evaluate_training` (metric: Kendall's tau-b).
* **Validation** --- five reactors (:data:`VALIDATION_IDS`), expert order **secret**.
  Submit your ranking with :func:`submit_validation_ranking`; the server scores it and puts
  the result on the leaderboard.

Quick start::

    from reward_shaping import load_training, build_reactors, evaluate_training

    reactors = build_reactors(load_training())
    scores = {i: my_reward(r) for i, r in reactors.items()}
    print(evaluate_training(scores)['summary'])
"""

from .experiment_data import (
  Experiment, load_experiment, load_all, build_reactor, MissingChannel,
  OD_TO_BIOMASS, DOT_SATURATION, OFFLINE_DEVICE,
)
from .data import (
  DATA_DIR, TRAINING_DIR, VALIDATION_DIR,
  load_training, load_validation, load_all_experiments, build_reactors,
)
from .growth import spline_biomass, growth_rate
from .ranking import (
  TRAINING_EXPERT_RANKING, TRAINING_IDS, VALIDATION_IDS, ALL_IDS,
  kendall_tau_b, ranking_from_scores, evaluate_training, plot_ranking_comparison,
)
from .submission import API_URL, validate_ranking_shape, submit_validation_ranking

__all__ = [
  'Experiment', 'load_experiment', 'load_all', 'build_reactor', 'MissingChannel',
  'OD_TO_BIOMASS', 'DOT_SATURATION', 'OFFLINE_DEVICE',
  'DATA_DIR', 'TRAINING_DIR', 'VALIDATION_DIR',
  'load_training', 'load_validation', 'load_all_experiments', 'build_reactors',
  'spline_biomass', 'growth_rate',
  'TRAINING_EXPERT_RANKING', 'TRAINING_IDS', 'VALIDATION_IDS', 'ALL_IDS',
  'kendall_tau_b', 'ranking_from_scores', 'evaluate_training', 'plot_ranking_comparison',
  'API_URL', 'validate_ranking_shape', 'submit_validation_ranking',
]
