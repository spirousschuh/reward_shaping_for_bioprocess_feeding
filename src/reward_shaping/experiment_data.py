"""The on-disk form of a cultivation, and the reactor a reward function is scored on.

Ported, essentially verbatim, from the ``mode`` repository
(``bvt-htbd/kiwi/tf1/mode``, ``scripts/experiment_data.py``). The only change here is
that the repo-root helper and the database-facing code are gone: this module just reads
the ``.npz`` files that ``read_experiments.py`` wrote and turns one into a reactor. It
depends on nothing but :mod:`numpy`.

``read_experiments.py`` pulls a handful of channels out of the ilab database and writes one
``.npz`` per experiment; this module reads them back.

**The file format.** One compressed ``.npz`` per experiment, holding

``meta``
    a zero-dimensional unicode array with a JSON document: the experiment id, where and when
    the data came from, the run-level context under ``run`` (organism, plasmid, medium,
    vessel), and the channel table that names every other array in the file.
``c<nnn>_t`` / ``c<nnn>_v``
    timestamps in hours since the run started, and measured values, for the measurement
    channel that ``meta['channels'][nnn]`` describes.
``s<nnn>_t`` / ``s<nnn>_v``
    the same for a setpoint channel from ``meta['setpoints']``.

The indirection through ``meta`` is what keeps channel names --- which contain underscores
and are chosen by the lab, not by us --- out of the archive member names. Nothing is pickled,
so a file stays readable without this module.
"""

from typing import Optional

import json

import numpy as np

__all__ = [
  'MEASUREMENT_CHANNELS', 'SETPOINT_CHANNELS', 'FIT_OBSERVATIONS', 'MissingChannel',
  'FEED_EXECUTED_CHANNEL', 'FEED_SETPOINT_CHANNEL', 'GLUCOSE_FEEDS', 'OTHER_FEEDS',
  'OD_TO_BIOMASS', 'DOT_SATURATION', 'OFFLINE_DEVICE',
  'Experiment', 'save_experiment', 'load_experiment', 'load_all', 'refresh_metadata',
  'derive_horizon', 'build_reactor', 'spread_pulses',
]

### Conversions and rig constants, all from
### `experiments/parameter-estimation-from-database.ipynb` in the mode repo.
OD_TO_BIOMASS = 0.33      # g/L per OD600 unit; the convention in `bio_mode/utils/data.py`
FEED_GLUCOSE_S = 200.0    # g/L of glucose in the feed
DOT_SATURATION = 100.0    # the DOT probe reads percent of air saturation, so 100 -> 1. The
                          # mapping is a property of the calibration, not of the run, which
                          # is why it is a constant rather than each trace's own maximum.
OFFLINE_DEVICE = 9        # measuring_setup.device_id of the Cedex offline assay
V0 = 1.0e-2               # L, the 2mag working volume
A0_FLOOR = 1.0e-3         # g/L; a conditioned initial acetate of exactly 0 is worth avoiding.

### The glucose feed, in the order it is preferred. The platform re-versioned its feed
### channels as it changed, and `glc_v2` .. `glc_v5` are the same quantity under later names.
###
### Each entry is (delivered volume, schedule, concentration setpoint). The concentration is
### the one number in a bolus that no volume channel carries, and the database does record it
### for two of the five --- so `FEED_GLUCOSE_S` is a fallback and not a fact.
GLUCOSE_FEEDS = (
  ('Cumulated_feed_volume_glucose', 'Feed_glc_cum_setpoints',    'feed_glucose_concentration'),
  ('Cumulated_feed_volume_glc_v2',  'Feed_glc_v2_cum_setpoints', 'feed_glc_v2_concentration'),
  ('Cumulated_feed_volume_glc_v3',  'Feed_glc_v3_cum_setpoints', None),
  ('Cumulated_feed_volume_glc_v4',  'Feed_glc_v4_cum_setpoints', None),
  ('Cumulated_feed_volume_glc_v5',  'Feed_glc_v5_cum_setpoints', None),
)

FEED_EXECUTED_CHANNEL = GLUCOSE_FEEDS[0][0]   # cumulative uL the pump delivered
FEED_SETPOINT_CHANNEL = GLUCOSE_FEEDS[0][1]   # cumulative uL that was scheduled

### Feeds the model has no state for. A reactor fed glycerol, lactose, dextrine or a
### substrate mix is not a glucose fed-batch with a channel missing, it is a different
### experiment. Named here so that a skip can say which one it was.
OTHER_FEEDS = (
  ('Cumulated_feed_volume_glycerol', 'glycerol'),
  ('Cumulated_feed_volume_gly_v2',   'glycerol'),
  ('Cumulated_feed_volume_lactose',  'lactose'),
  ('Cumulated_feed_volume_lac_v2',   'lactose'),
  ('Cumulated_feed_volume_lac_v3',   'lactose'),
  ('Cumulated_feed_volume_dextrine', 'dextrine'),
  ('Cumulated_feed_volume_S_mix',    'a substrate mix'),
  ('Cumulated_feed_volume_ethanol',  'ethanol'),
  ('Cumulated_feed_volume_methanol', 'methanol'),
)

### Only what a parameter fit actually consumes. The database also carries Acid, Base,
### Fluo_CFP/RFP/YFP, Magnesium, Phosphate, Probe_Volume, Volume_evaporated and pH for these
### runs; none of them enter the likelihood or the initial state of `ecoli2023mod`, whose
### states are (X, S, A, DOTm, V), so they are not pulled.
###
### A channel absent from a given experiment is simply not stored, so listing both devices
### for a channel that moved between them costs nothing.
MEASUREMENT_CHANNELS = (
  ### (channel, device_id) -- device None is the at-line probe, 9 the Cedex offline assay
  ('OD600', None),                          # -> X, via OD_TO_BIOMASS
  ('Glucose', None),                        # -> S, the Hamilton at-line reading
  ('Glucose', OFFLINE_DEVICE),              # the offline assay, plotted against the at-line
  ('Acetate', OFFLINE_DEVICE),              # -> A, where a Cedex ran
  ('Acetate', None),                        # -> A everywhere else; see FIT_OBSERVATIONS
  ('DOT', None),                            # -> DOTm, as a fraction of DOT_SATURATION
  ('Cumulated_feed_volume_medium', None),   # not modelled; reported so it is not forgotten
  ('Volume', None),                         # net volume, likewise
) + tuple(
  ### -> the boluses; what the pump actually did, under whichever name this run used
  (delivered, None) for delivered, _, _ in GLUCOSE_FEEDS
) + tuple(
  ### not modelled, but pulled so that a reactor can say what it *was* fed instead
  (delivered, None) for delivered, _ in OTHER_FEEDS
)

SETPOINT_CHANNELS = tuple(
  ### the schedule: a cross-check where the delivered feed exists, the fallback where it does
  ### not; and the feed strength, which no measurement channel carries
  channel
  for _, schedule, concentration in GLUCOSE_FEEDS
  for channel in (schedule, concentration) if channel
)

### The four observations the likelihood scores, and which device to take each from, most
### preferred first.
FIT_OBSERVATIONS = (
  ('X',    'OD600',   (None, )),
  ('S',    'Glucose', (None, )),                  # the at-line Hamilton, not the assay
  ('A',    'Acetate', (OFFLINE_DEVICE, None)),
  ('DOTm', 'DOT',     (None, )),
)


class MissingChannel(KeyError):
  """A channel the caller needs is not usable: absent, or empty within the horizon.

  Its own class rather than a bare :exc:`KeyError` so that a batch over many reactors can
  skip one unreadable file and carry on with the rest without also swallowing a genuine bug
  in a dictionary lookup.
  """

  def __str__(self):
    ### KeyError's repr quotes its argument, which turns a sentence into '"..."'
    return self.args[0] if self.args else ''


def _sorted(t, v):
  """A channel in a canonical order: by timestamp, ties broken by value."""
  t = np.asarray(t, dtype=np.float64)
  v = np.asarray(v, dtype=np.float64)
  order = np.lexsort((v, t))
  return t[order], v[order]


class Experiment(object):
  """One cultivation as it came out of the database, on the run clock."""

  def __init__(self, experiment_id: int, meta: dict, series: dict, setpoints: dict):
    self.experiment_id = experiment_id
    self.meta = meta
    self._series = series
    self._setpoints = setpoints

  @property
  def run(self):
    """The run-level context, or ``{}`` for a file written before it was stored."""
    return self.meta.get('run') or dict()

  def channels(self):
    """The ``(channel, device)`` keys this file carries, in a stable order."""
    return tuple(sorted(self._series, key=lambda k: (k[0], str(k[1]))))

  def has(self, channel: str, device: Optional[int]=None):
    return (channel, device) in self._series

  def has_setpoints(self, channel: str):
    return channel in self._setpoints

  def series(self, channel: str, device: Optional[int]=None):
    """One channel as ``(timestamps, values)``, sorted by time.

    :raises MissingChannel: if the channel was not among the ones pulled from the database,
      which is a different situation from a channel that exists but is empty.
    """
    try:
      return self._series[channel, device]
    except KeyError:
      raise MissingChannel(
        'experiment %d has no channel %r on device %r; it carries %s' % (
          self.experiment_id, channel, device,
          ', '.join('%s/%s' % (c, d) for c, d in self.channels())
        )
      ) from None

  def pick(self, channel: str, devices=(None, )):
    """The first of ``devices`` that carries ``channel``, as ``(device, timestamps, values)``.

    :raises MissingChannel: if none of them does.
    """
    for device in devices:
      if self.has(channel, device):
        return device, self._series[channel, device][0], self._series[channel, device][1]

    raise MissingChannel(
      'experiment %d has no %r on any of the devices %s; it carries %s' % (
        self.experiment_id, channel, ', '.join(repr(d) for d in devices),
        ', '.join('%s/%s' % (c, d) for c, d in self.channels())
      )
    )

  def setpoints(self, channel: str):
    try:
      return self._setpoints[channel]
    except KeyError:
      raise MissingChannel(
        'experiment %d has no setpoint channel %r; it carries %s' % (
          self.experiment_id, channel, ', '.join(sorted(self._setpoints)) or '(none)'
        )
      ) from None

  def n_invalid(self, channel: str, device: Optional[int]=None):
    """How many rows the lab flagged invalid and ``read_measurements`` therefore dropped."""
    for entry in self.meta['channels']:
      if entry['channel'] == channel and entry['device'] == device:
        return int(entry['n_invalid'])
    return 0

  def describe(self):
    """One line naming what was cultivated, for plot titles and run reports."""
    run = self.run
    if not run:
      return 'no run metadata'

    strain = ' / '.join(
      str(run[key]) for key in ('organism_name', 'plasmid_name', 'medium_name')
      if run.get(key)
    )
    where = ', '.join(part for part in (
      run.get('bioreactor_type_name'),
      'pos %s' % (run['container_number'], ) if run.get('container_number') else None,
    ) if part)

    return ' -- '.join(part for part in (
      'run %s %s' % (run.get('run_id', '?'), run.get('run_name', '?')),
      strain or None,
      where or None,
    ) if part)

  def __repr__(self):
    return 'Experiment(%d, %d channels)' % (self.experiment_id, len(self._series))


def experiment_path(directory, experiment_id: int):
  import os
  return os.path.join(directory, 'experiment_%d.npz' % (experiment_id, ))


def save_experiment(
  directory: str, experiment_id: int,
  series: dict, setpoints: dict, invalid: dict, source: dict, metadata: Optional[dict]=None
):
  """Write one experiment to ``<directory>/experiment_<id>.npz``.

  Kept from the upstream module so a round-trip test can exercise the reader against a file
  it wrote itself; the workshop only ever reads the bundled files.
  """
  import os
  arrays = dict()
  channel_meta = list()

  for index, (channel, device) in enumerate(sorted(series, key=lambda k: (k[0], str(k[1])))):
    t, v = _sorted(*series[channel, device])
    key = 'c%03d' % (index, )
    arrays['%s_t' % (key, )], arrays['%s_v' % (key, )] = t, v
    channel_meta.append(dict(
      key=key, channel=channel, device=device, n=int(t.size),
      n_invalid=int(invalid.get((channel, device), 0)),
    ))

  setpoint_meta = list()
  for index, channel in enumerate(sorted(setpoints)):
    t, v = _sorted(*setpoints[channel])
    key = 's%03d' % (index, )
    arrays['%s_t' % (key, )], arrays['%s_v' % (key, )] = t, v
    setpoint_meta.append(dict(key=key, channel=channel, n=int(t.size)))

  meta = dict(
    experiment_id=int(experiment_id), time_unit='h', source=source,
    run=metadata or None,
    channels=channel_meta, setpoints=setpoint_meta,
  )

  os.makedirs(directory, exist_ok=True)
  path = experiment_path(directory, experiment_id)
  np.savez_compressed(path, meta=np.array(json.dumps(meta, indent=2)), **arrays)
  return path


def load_experiment(path: str) -> Experiment:
  """Read back one file written by :func:`save_experiment`."""
  with np.load(path, allow_pickle=False) as handle:
    meta = json.loads(str(handle['meta']))

    series = {
      (entry['channel'], entry['device']): (
        handle['%s_t' % (entry['key'], )], handle['%s_v' % (entry['key'], )]
      )
      for entry in meta['channels']
    }
    setpoints = {
      entry['channel']: (
        handle['%s_t' % (entry['key'], )], handle['%s_v' % (entry['key'], )]
      )
      for entry in meta['setpoints']
    }

  return Experiment(int(meta['experiment_id']), meta, series, setpoints)


def refresh_metadata(path: str, metadata: Optional[dict]):
  """Replace one file's ``meta['run']``, leaving every measurement array untouched."""
  with np.load(path, allow_pickle=False) as handle:
    meta = json.loads(str(handle['meta']))
    arrays = {name: handle[name] for name in handle.files if name != 'meta'}

  meta['run'] = metadata or None
  np.savez_compressed(path, meta=np.array(json.dumps(meta, indent=2)), **arrays)
  return path


def load_all(directory: str, experiment_ids=None, required: bool=True):
  """Every experiment in ``directory``, or just the requested ids, oldest id first.

  :param required: raise for a requested id that is not stored. ``False`` returns the ones
    that are.
  :raises FileNotFoundError: if the directory does not exist, or --- when ``required`` --- a
    requested id is missing.
  """
  import os
  if not os.path.isdir(directory):
    raise FileNotFoundError('%s does not exist' % (directory, ))

  if experiment_ids is None:
    paths = sorted(
      os.path.join(directory, name)
      for name in os.listdir(directory)
      if name.startswith('experiment_') and name.endswith('.npz')
    )
    if not paths:
      raise FileNotFoundError('no experiment_*.npz in %s' % (directory, ))
  else:
    paths = list()
    for experiment_id in experiment_ids:
      path = experiment_path(directory, experiment_id)
      if not os.path.exists(path):
        if required:
          raise FileNotFoundError('%s is missing' % (path, ))
        continue
      paths.append(path)

  experiments = [load_experiment(path) for path in paths]
  return {experiment.experiment_id: experiment for experiment in experiments}


def derive_horizon(experiment: Experiment):
  """The end of the window :func:`build_reactor` integrates, and where that number came from.

  Whichever ends first: the run, or the sensors. The horizon is the smaller of
  ``runs.end_time`` (when recorded and positive) and the last timestamp on any fitted
  channel. When ``end_time`` is missing or zero the observations are all there is.

  :returns: ``(horizon, source)``, source being ``'end_time'`` or ``'observations'``.
  :raises MissingChannel: when there is no usable ``end_time`` and a fitted channel is absent.
  """
  duration = experiment.run.get('duration_h')

  if duration is None or duration <= 0.0:
    return max(
      float(experiment.pick(channel, devices)[1].max())
      for _, channel, devices in FIT_OBSERVATIONS
    ), 'observations'

  last_seen = [
    float(experiment.pick(channel, devices)[1].max())
    for _, channel, devices in FIT_OBSERVATIONS
    if any(experiment.has(channel, device) for device in devices)
  ]
  if last_seen and max(last_seen) < float(duration):
    return max(last_seen), 'observations'

  return float(duration), 'end_time'


def biomass_start_time(experiment: Experiment):
  """The first OD600 reading, in hours since the run's own clock --- where a reactor starts.

  ``runs.start_time`` is when the operator started the run record, which is not the same
  moment as the first biomass sample. Whatever happened before the first biomass reading is
  not part of a cultivation any measurement can constrain, so :func:`reactor_window` treats
  this timestamp as the reactor's own t=0.

  :raises MissingChannel: if the experiment carries no OD600 channel at all.
  """
  t, _ = experiment.series('OD600')
  return float(t.min())


def reactor_window(experiment: Experiment, horizon: Optional[float]=None):
  """``(start, end, horizon_source)``: the window :func:`build_reactor` integrates.

  Both bounds are hours since the run's own clock. ``start`` is
  :func:`biomass_start_time`; ``end`` is :func:`derive_horizon`'s result, or ``horizon``
  when one is given explicitly.

  :raises MissingChannel: if the window is empty or inverted.
  """
  if horizon is None:
    end, horizon_source = derive_horizon(experiment)
  else:
    end, horizon_source = float(horizon), 'requested'

  start = biomass_start_time(experiment)
  if start >= end:
    raise MissingChannel(
      'experiment %d first measured biomass at %.4f h, at or after the %.4f h horizon (%s); '
      'there is no window left to integrate' % (
        experiment.experiment_id, start, end, horizon_source)
    )

  return start, end, horizon_source


def _within(t, v, start, end, what, experiment_id):
  """One channel cut to ``[start, end]`` and shifted so ``start`` becomes t=0.

  :raises MissingChannel: if nothing is left in the window.
  :returns: ``(t, v, n_before, n_after)``, ``t`` relative to ``start``.
  """
  keep = (t >= start) & (t <= end)
  n_before = int((t < start).sum())
  n_after = int((t > end).sum())

  if not keep.any():
    raise MissingChannel(
      'experiment %d has no %s reading within %.4f-%.4f h; its %d readings span %.4f-%.4f h'
      % (experiment_id, what, start, end, len(t),
         t.min() if len(t) else float('nan'), t.max() if len(t) else float('nan'))
    )

  return t[keep] - start, v[keep], n_before, n_after


def _concentration_at(times, setpoint_t, setpoint_v, default):
  """The feed strength in force at each bolus, as a zero-order hold."""
  if setpoint_t is None or not len(setpoint_t):
    return np.full(times.shape, default)

  order = np.argsort(setpoint_t, kind='stable')
  ts, vs = np.asarray(setpoint_t)[order], np.asarray(setpoint_v)[order]
  index = np.clip(np.searchsorted(ts, times, side='right') - 1, 0, len(vs) - 1)
  return vs[index]


def _boluses(t, cumulative, start, end):
  """A cumulative feed trace differenced into individual boluses within ``[start, end]``.

  Diffed over the *whole* series first, then windowed, so that microlitres delivered before
  ``start`` are dropped rather than folded into one synthetic mega-bolus at the window's
  first kept timestamp.

  :returns: ``(timestamps, volumes_in_litres, n_before, n_after)``, timestamps relative to
    ``start``.
  """
  volumes = np.diff(np.concatenate([[0.0], cumulative])) * 1e-6
  keep = (t >= start) & (t <= end)
  ts = t[keep] - start
  vs = volumes[keep]
  positive = vs > 0
  return ts[positive], vs[positive], int((t < start).sum()), int((t > end).sum())


def spread_pulses(pulse_t, pulse_v, pulse_s, pieces: int, horizon: float):
  """Divide each bolus into ``pieces`` equal parts spread over the gap to the next one.

  Off by default (``pieces == 1``). Kept from upstream for anyone who wants to ask whether
  the sub-resolution feed spikes matter; see the mode repo for the full rationale.
  """
  if pieces <= 1 or len(pulse_t) == 0:
    return pulse_t, pulse_v, pulse_s

  gaps = np.diff(np.concatenate([pulse_t, [horizon]]))
  gaps[gaps <= 0] = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 0.0

  offsets = np.arange(pieces) / pieces
  t = (pulse_t[:, None] + gaps[:, None] * offsets[None, :]).reshape(-1)
  v = np.repeat(pulse_v / pieces, pieces)
  c = np.repeat(pulse_s, pieces)

  keep = t < horizon
  return t[keep], v[keep], c[keep]


def build_reactor(experiment: Experiment, horizon: Optional[float]=None, split_pulses: int=1):
  """Everything a reward function needs for one reactor, in model units.

  **The reactor's t=0 is the first biomass measurement, not the run's ``start_time``.**
  :func:`reactor_window` gives ``(start, end)``; every channel here is cut to that window and
  shifted so ``start`` becomes t=0. A bolus delivered before ``start`` is dropped outright.

  :param experiment: a loaded experiment;
  :param split_pulses: divide each glucose bolus into this many pieces (1 leaves them alone);
  :param horizon: hours since the run's own ``start_time``; ``None`` takes the end from
    :func:`derive_horizon`.
  :returns: a dictionary with ``observations`` (``{name: (t, v)}`` in model units, ``t``
    relative to the reactor's own start), ``pulses`` (``(t, volumes_in_litres,
    concentrations)``), ``initial``, the ``horizon`` (the *duration* now integrated, i.e.
    ``end - start``), ``horizon_from``, ``biomass_start`` (hours since the run's own clock),
    and a ``report`` of what was kept and dropped.
  :raises MissingChannel: if a scored channel, or the feed, is absent or empty in the window.
  """
  report = list()
  experiment_id = experiment.experiment_id

  start, end, horizon_source = reactor_window(experiment, horizon)
  horizon = end - start

  duration = experiment.run.get('duration_h')
  report.append('biomass start: %.4f h since the run began (first OD600 reading) -- '
                'everything before it is dropped rather than shifted into it' % (start, ))
  report.append('horizon: %.4f h (%.4f-%.4f h since the run began), end from %s%s' % (
    horizon, start, end, horizon_source,
    '' if duration is None else ' (the run was open for %.4f h)' % (duration, )
  ))

  X_t, od = experiment.series('OD600')
  X_t, X, X_before, X_late = _within(X_t, OD_TO_BIOMASS * od, start, end, 'OD600', experiment_id)
  report.append('OD600: %d valid readings, %d flagged invalid, %d before the window start, '
                '%d past the horizon' % (
    len(X), experiment.n_invalid('OD600'), X_before, X_late))

  S_t, S = experiment.series('Glucose')
  S_t, S, S_before, S_late = _within(S_t, S, start, end, 'glucose', experiment_id)
  report.append('Glucose (Hamilton): %d valid, %d flagged invalid, %d before the window '
                'start, %d past the horizon' % (
    len(S), experiment.n_invalid('Glucose'), S_before, S_late))

  A_device, A_t, A = experiment.pick('Acetate', (OFFLINE_DEVICE, None))
  A_t, A, A_before, A_late = _within(A_t, A, start, end, 'acetate', experiment_id)
  report.append('Acetate (device %s, on sample_time): %d valid, %.2f-%.2f h, peak %.3f g/L '
                'at %.2f h, %d before the window start, %d past the horizon' % (
    A_device, len(A), A_t.min(), A_t.max(), A.max(), A_t[np.argmax(A)], A_before, A_late))

  dot_t, dot = experiment.series('DOT')
  dot_t, dot, dot_before, dot_late = _within(dot_t, dot, start, end, 'DOT', experiment_id)
  dot_peak = float(dot.max())
  DOTm_t, DOTm = dot_t, dot / DOT_SATURATION
  report.append('DOT: %d readings, %d before the window start, %d past the horizon; '
                'normalised by %.0f (air saturation), peak reading %.2f' % (
    len(DOTm), dot_before, dot_late, DOT_SATURATION, dot_peak))

  observations = {'X': (X_t, X), 'S': (S_t, S), 'A': (A_t, A), 'DOTm': (DOTm_t, DOTm)}

  pulses, sources, channels, strengths = list(), list(), list(), list()
  before_start = beyond = 0

  for delivered, schedule, concentration_channel in GLUCOSE_FEEDS:
    if experiment.has(delivered):
      t, cumulative = experiment.series(delivered)
      feed_source, channel = 'delivered', delivered
    elif experiment.has_setpoints(schedule):
      t, cumulative = experiment.setpoints(schedule)
      feed_source, channel = 'planned', schedule
    else:
      continue

    t, volumes, dropped_before, dropped_after = _boluses(t, cumulative, start, end)
    before_start += dropped_before
    beyond += dropped_after
    if not len(t):
      continue

    if concentration_channel and experiment.has_setpoints(concentration_channel):
      c_t, c_v = experiment.setpoints(concentration_channel)
      strength = concentration_channel
    else:
      c_t, c_v, strength = None, None, 'assumed'

    concentrations = _concentration_at(t, c_t, c_v, FEED_GLUCOSE_S)
    pulses.append((t, volumes, concentrations))
    sources.append(feed_source); channels.append(channel); strengths.append(strength)

    report.append(
      'feed (%s, %s): %d boluses, %.0f uL at %s g/L (%s), %.2f-%.2f h' % (
        feed_source, channel, len(t), 1e6 * volumes.sum(),
        '%.1f' % concentrations[0] if len(set(concentrations)) == 1
        else '%.1f-%.1f' % (concentrations.min(), concentrations.max()),
        strength, t.min(), t.max()))

    if feed_source == 'delivered' and experiment.has_setpoints(schedule):
      plan_t, plan_cumulative = experiment.setpoints(schedule)
      planned = float(np.interp(t.max() + start, plan_t, plan_cumulative))
      report.append('  %s called for %.0f uL by %.2f h, %+.0f uL against what ran' % (
        schedule, planned, t.max(), 1e6 * volumes.sum() - planned))

  if not pulses:
    instead = sorted({
      substrate for channel, substrate in OTHER_FEEDS if experiment.has(channel)
    })
    raise MissingChannel(
      'experiment %d records no glucose feed within the %.4f-%.4f h window -- neither a '
      'delivered volume nor a schedule, under any of %s%s' % (
        experiment_id, start, end, ', '.join(d for d, _, _ in GLUCOSE_FEEDS),
        ('; it was fed %s instead' % (' and '.join(instead), )) if instead else ''
      )
    )

  pulse_t = np.concatenate([p[0] for p in pulses])
  pulse_v = np.concatenate([p[1] for p in pulses])
  pulse_s = np.concatenate([p[2] for p in pulses])
  order = np.argsort(pulse_t, kind='stable')
  pulse_t, pulse_v, pulse_s = pulse_t[order], pulse_v[order], pulse_s[order]

  pulse_source = '+'.join(sorted(set(sources)))
  report.append('total glucose fed: %.0f uL over %d boluses from %s (%d before the window '
                'start, %d past the %.2f h horizon, both dropped)' % (
                  1e6 * pulse_v.sum(), len(pulse_t), ', '.join(channels),
                  before_start, beyond, horizon))

  if split_pulses > 1:
    delivered = pulse_v.sum()
    pulse_t, pulse_v, pulse_s = spread_pulses(
      pulse_t, pulse_v, pulse_s, split_pulses, horizon)
    report.append('each bolus split into %d pieces spread over the gap to the next: %d events, '
                  '%.0f uL (was %.0f)' % (
                    split_pulses, len(pulse_t), 1e6 * pulse_v.sum(), 1e6 * delivered))

  net_volume = None
  try:
    medium_t, medium = experiment.series('Cumulated_feed_volume_medium')
    net_t, net = experiment.series('Volume')
  except MissingChannel as error:
    report.append('not modelled, and not recorded either: %s' % (str(error).split(';')[0], ))
  else:
    net_volume = (net_t, net)
    report.append(
      'not modelled: %.0f uL of medium boluses over %.2f-%.2f h, and the sampling withdrawals '
      '-- the measured net volume change is %+.0f uL against %+.0f uL of glucose feed' % (
        medium[-1], medium_t.min(), medium_t.max(), net[-1], 1e6 * pulse_v.sum()
      )
    )

  first_S = S[S_t == S_t.min()]
  report.append('S0: %.3f g/L, the median of the %d valid readings at t=%.3f h' % (
    np.median(first_S), len(first_S), S_t.min()))

  initial = dict(
    X=float(np.median(X[X_t == X_t.min()])),
    S=float(np.median(first_S)),
    A=float(max(A[A_t == A_t.min()].mean(), A0_FLOOR)),
    DOTm=float(DOTm[np.argmin(DOTm_t)]),
    V=V0,
  )

  return dict(
    experiment=experiment_id,
    observations=observations,
    pulses=(pulse_t, pulse_v, pulse_s),
    pulse_source=pulse_source,
    feed_channels=tuple(channels),
    feed_concentrations=tuple(strengths),
    initial=initial,
    horizon=float(horizon),
    horizon_from=horizon_source,
    biomass_start=float(start),
    net_volume=net_volume,
    dot_saturation=DOT_SATURATION,
    dot_peak=dot_peak,
    report=report,
  )
