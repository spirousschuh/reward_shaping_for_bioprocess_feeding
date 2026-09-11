"""The training expert ranking and the tools to score a reward function against it.

The challenge has two stages:

* **Training** --- the six experiments in :data:`TRAINING_EXPERT_RANKING`. The expert's
  order is known and printed here; you score your reward function locally with
  :func:`evaluate_training`, whose metric is **Kendall's tau-b**.
* **Validation** --- the five experiments in :data:`VALIDATION_IDS`. The expert's order is
  **secret**: it is not in this repository in any form. You submit your ranking of these
  five to the challenge server (see :mod:`reward_shaping.submission`) and it computes the
  Kendall tau-b for you.

Nothing in this module reveals the validation order. :data:`VALIDATION_IDS` is listed in
plain ascending numeric order, which carries no ranking information.
"""

import numpy as np

__all__ = [
  'TRAINING_EXPERT_RANKING', 'TRAINING_IDS', 'VALIDATION_IDS', 'ALL_IDS',
  'kendall_tau_b', 'ranking_from_scores', 'evaluate_training', 'plot_ranking_comparison',
]

### The expert's order for the training reactors, best first. Public on purpose --- this is
### the feedback signal participants develop against.
TRAINING_EXPERT_RANKING = [20294, 20293, 17201, 16738, 16748, 16736]
TRAINING_IDS = list(TRAINING_EXPERT_RANKING)

### The validation reactors. THIS LIST IS IN ASCENDING NUMERIC ORDER AND SAYS NOTHING ABOUT
### THE EXPERT RANKING. The expert's order for these five is held only by the challenge
### server, as an environment variable; it is deliberately absent from this repository.
VALIDATION_IDS = [16737, 16742, 17191, 17193, 20285]

ALL_IDS = sorted(TRAINING_IDS + VALIDATION_IDS)


def _rank_vector(order, ids):
  """Position (0 = best) of each id in ``ids`` within the sequence ``order``."""
  pos = {eid: i for i, eid in enumerate(order)}
  return np.array([pos[eid] for eid in ids], dtype=float)


def kendall_tau_b(order_a, order_b):
  """Kendall's tau-b between two orderings of the same set of ids.

  Both arguments are sequences of experiment ids, best first. Ties are handled by SciPy's
  tau-b (the default ``variant``), so an ``order`` may also be a list of rank numbers with
  repeats if a caller needs to express a tie.

  :returns: a float in ``[-1, 1]``; ``1`` is an identical ordering, ``-1`` the reverse.
  """
  from scipy.stats import kendalltau

  ids = list(order_a)
  a = _rank_vector(order_a, ids)
  b = _rank_vector(order_b, ids)
  tau = kendalltau(a, b).statistic
  return float(tau)


def ranking_from_scores(scores, higher_is_better=True):
  """The experiment ids in ``scores`` sorted best-to-worst by their reward."""
  return sorted(scores, key=lambda eid: scores[eid], reverse=higher_is_better)


def evaluate_training(scores, higher_is_better=True):
  """Score a reward function on the six training reactors.

  :param scores: ``{experiment_id: reward}`` covering (a subset of) :data:`TRAINING_IDS`.
  :param higher_is_better: whether a larger reward should mean a better reactor.
  :returns: a dict with

    ``predicted_order``
        the training ids sorted best-to-worst by ``scores``.
    ``expert_order``
        :data:`TRAINING_EXPERT_RANKING`, narrowed to the ids that were scored.
    ``kendall_tau``
        Kendall's tau-b between the two.
    ``summary``
        a short human-readable report ending with ``Training Kendall tau: <value>``.
  """
  scored = {eid: scores[eid] for eid in TRAINING_IDS if eid in scores}
  missing = [eid for eid in TRAINING_IDS if eid not in scores]

  expert_order = [eid for eid in TRAINING_EXPERT_RANKING if eid in scored]
  predicted_order = ranking_from_scores(scored, higher_is_better=higher_is_better)
  tau = kendall_tau_b(expert_order, predicted_order)

  lines = [
    'training reactors scored: %d%s' % (
      len(scored), '' if not missing else ' (missing %s)' % (missing, )),
    'expert order   : %s' % (expert_order, ),
    'predicted order: %s' % (predicted_order, ),
    'Training Kendall tau: %.3f' % (tau, ),
  ]

  return dict(
    predicted_order=predicted_order,
    expert_order=expert_order,
    kendall_tau=tau,
    summary='\n'.join(lines),
  )


def plot_ranking_comparison(scores, higher_is_better=True, ax=None):
  """A slope chart of the training reactors: expert rank (left) vs your reward's rank (right).

  A line that stays flat is a reactor your reward placed exactly where the expert did.
  Only the six training reactors are shown --- the validation order is not known here.
  """
  import matplotlib.pyplot as plt

  result = evaluate_training(scores, higher_is_better=higher_is_better)
  expert_order = result['expert_order']
  predicted_order = result['predicted_order']

  if ax is None:
    _, ax = plt.subplots(figsize=(5.0, 4.5))

  hit, miss = '#00926c', '#c0392b'
  for eid in expert_order:
    y0 = expert_order.index(eid)
    y1 = predicted_order.index(eid)
    colour = hit if y0 == y1 else miss
    ax.plot([0, 1], [y0, y1], '-o', color=colour, ms=6, lw=1.5)
    ax.annotate(str(eid), (0, y0), textcoords='offset points', xytext=(-8, 0),
                ha='right', va='center', fontsize=9, color=colour)
    ax.annotate(str(eid), (1, y1), textcoords='offset points', xytext=(8, 0),
                ha='left', va='center', fontsize=9, color=colour)

  ax.set_xlim(-0.35, 1.35)
  ax.set_ylim(len(expert_order) - 0.5, -0.5)
  ax.set_xticks([0, 1])
  ax.set_xticklabels(['expert', 'your reward'])
  ax.set_yticks([])
  ax.set_title('Training Kendall tau = %.2f' % (result['kendall_tau'], ), fontsize=10)
  for spine in ('top', 'right', 'left'):
    ax.spines[spine].set_visible(False)
  return ax
