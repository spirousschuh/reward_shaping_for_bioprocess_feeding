"""Loading the bundled cultivation data.

The experiment ``.npz`` files (copied verbatim from the ``mode`` repository) live under
``data/`` at the repo root, split into the two challenge stages::

    data/training/     6 reactors, expert order known  (reward_shaping.ranking.TRAINING_IDS)
    data/validation/   5 reactors, expert order secret (reward_shaping.ranking.VALIDATION_IDS)

Both directories hold only measurement data --- no ranking information.
"""

from pathlib import Path

from .experiment_data import load_all, build_reactor, MissingChannel
from .ranking import TRAINING_IDS, VALIDATION_IDS, ALL_IDS

__all__ = [
  'DATA_DIR', 'TRAINING_DIR', 'VALIDATION_DIR',
  'load_training', 'load_validation', 'load_all_experiments', 'build_reactors',
]

### <repo>/src/reward_shaping/data.py  ->  <repo>/data
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
TRAINING_DIR = DATA_DIR / 'training'
VALIDATION_DIR = DATA_DIR / 'validation'


def load_training(data_dir=None):
  """The six training experiments as ``{experiment_id: Experiment}``."""
  directory = Path(data_dir) / 'training' if data_dir else TRAINING_DIR
  return load_all(str(directory), TRAINING_IDS, required=True)


def load_validation(data_dir=None):
  """The five validation experiments as ``{experiment_id: Experiment}``.

  You get the measurements and the ids; the expert's order for them is held only by the
  challenge server.
  """
  directory = Path(data_dir) / 'validation' if data_dir else VALIDATION_DIR
  return load_all(str(directory), VALIDATION_IDS, required=True)


def load_all_experiments(data_dir=None):
  """All eleven experiments (training + validation) as ``{experiment_id: Experiment}``."""
  merged = {}
  merged.update(load_training(data_dir))
  merged.update(load_validation(data_dir))
  return {eid: merged[eid] for eid in ALL_IDS if eid in merged}


def build_reactors(experiments, *, quiet=False):
  """``{experiment_id: build_reactor(...)}`` for a mapping of loaded experiments.

  A reactor that cannot be built --- a missing scored channel or feed --- is skipped with a
  message, the way ``plot_measurements.py`` does it, rather than aborting the batch.

  :param experiments: a ``{id: Experiment}`` mapping, e.g. from :func:`load_training`.
  :param quiet: suppress the per-skip message.
  """
  reactors = {}
  for experiment_id, experiment in experiments.items():
    try:
      reactors[experiment_id] = build_reactor(experiment)
    except MissingChannel as error:
      if not quiet:
        print('experiment %d: skipped -- %s' % (experiment_id, error))
  return reactors
