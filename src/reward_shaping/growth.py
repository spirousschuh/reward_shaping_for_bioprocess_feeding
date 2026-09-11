"""Specific growth rate from noisy biomass measurements.

The organism's specific growth rate is ``mu(t) = (dX/dt) / X``. The measurements are a
handful of at-line OD readings per run, so differentiating them directly is hopeless ---
the noise dominates the slope. Instead: fit a smoothing spline through the biomass points,
then read the growth rate off the fitted curve.

Fitting ``log(X)`` rather than ``X`` is the better-behaved choice: for exponential growth
``log(X)`` is a straight line, ``d/dt log(X)`` *is* ``mu`` directly (no division by a small,
noisy ``X``), and the reconstructed ``X = exp(spline)`` stays positive by construction.
Pass ``log=False`` to fit ``X`` on its own scale and form the ratio explicitly.

**Which spline.** :func:`scipy.interpolate.make_smoothing_spline` --- a penalised natural
cubic smoothing spline (degree 3 B-spline basis, penalty on the integrated squared second
derivative), with the smoothing parameter chosen automatically by generalised
cross-validation. On SciPy versions that lack it, the fallback is
:class:`scipy.interpolate.UnivariateSpline` (a FITPACK cubic smoothing spline) with a mild
smoothing factor derived from the scatter of the data.
"""

import numpy as np

__all__ = ['spline_biomass', 'growth_rate']


def _fit_spline(t, y, s=None):
  """A smoothing spline ``y(t)``. Uses ``make_smoothing_spline`` where SciPy has it.

  ``s`` stiffens the fit: with ``make_smoothing_spline`` it is passed through as the
  regularisation weight ``lam`` (``None`` lets generalised cross-validation choose it);
  with the ``UnivariateSpline`` fallback it is the FITPACK smoothing factor.
  """
  from scipy.interpolate import UnivariateSpline
  try:
    from scipy.interpolate import make_smoothing_spline
  except ImportError:
    make_smoothing_spline = None

  if make_smoothing_spline is not None:
    return make_smoothing_spline(t, y, lam=s)

  ### UnivariateSpline's default s = len(t) is often far too stiff for a dozen points;
  ### a mild default tied to the scatter of y works better as a fallback.
  if s is None:
    s = 0.05 * len(t) * float(np.var(y)) or None
  spline = UnivariateSpline(t, y, k=min(3, len(t) - 1), s=s)
  return spline


def spline_biomass(t, X, *, log=True, s=None):
  """The fitted spline through the biomass points.

  :param log: fit ``log(X)`` (default) rather than ``X``.
  :param s: stiffness of the fit. ``None`` (default) lets SciPy choose it by generalised
    cross-validation; a positive number over-rides that with a fixed smoothing weight ---
    raise it if the fitted curve chases the measurement noise.
  :returns: ``(spline, log)`` --- the callable spline object and the flag, so a caller
    knows which scale it is on.
  """
  t = np.asarray(t, dtype=float)
  X = np.asarray(X, dtype=float)
  order = np.argsort(t)
  t, X = t[order], X[order]

  ### replicate readings share a sample_time; the spline fitters need a strictly ascending
  ### grid, so collapse ties to their mean before fitting
  unique_t, inverse = np.unique(t, return_inverse=True)
  if unique_t.size != t.size:
    X = np.bincount(inverse, weights=X) / np.bincount(inverse)
    t = unique_t

  if t.size < 4:
    raise ValueError('need at least 4 distinct biomass times to fit a spline, got %d' % (t.size, ))

  y = np.log(np.clip(X, 1e-6, None)) if log else X
  return _fit_spline(t, y, s=s), log


def growth_rate(t, X, *, log=True, s=None, n_dense=200):
  """Specific growth rate ``mu = (dX/dt) / X`` from a smoothing spline through ``(t, X)``.

  :param t: measurement times, hours (need not be sorted).
  :param X: biomass, g/L, at those times.
  :param log: fit ``log(X)`` and take ``mu = d/dt log(X)`` (default), rather than fitting
    ``X`` and dividing.
  :param s: spline stiffness; ``None`` picks it by cross-validation, a positive number
    over-rides that. See :func:`spline_biomass`.
  :param n_dense: number of points on the returned dense grid.
  :returns: ``(t_dense, mu_dense, X_dense, X_smooth_at_t)``

    ``t_dense``
        an evenly spaced grid over ``[min(t), max(t)]``.
    ``mu_dense``
        the growth rate on that grid, 1/h --- for plotting a smooth curve.
    ``X_dense``
        the fitted biomass on that same grid, g/L --- for drawing the spline over the
        biomass measurements.
    ``X_smooth_at_t``
        the fitted biomass evaluated at the original measurement times --- handy for a
        reward that wants a de-noised final or peak biomass, or ``mu`` at the samples
        (``np.gradient(np.log(X_smooth_at_t), sorted(t))``).
  """
  t = np.asarray(t, dtype=float)
  X = np.asarray(X, dtype=float)
  order = np.argsort(t)
  t_sorted, X_sorted = t[order], X[order]

  spline, fitted_log = spline_biomass(t_sorted, X_sorted, log=log, s=s)
  d_spline = spline.derivative()

  t_dense = np.linspace(t_sorted[0], t_sorted[-1], n_dense)
  if fitted_log:
    mu_dense = d_spline(t_dense)
    X_dense = np.exp(spline(t_dense))
    X_smooth_at_t = np.exp(spline(t_sorted))
  else:
    mu_dense = d_spline(t_dense) / np.clip(spline(t_dense), 1e-9, None)
    X_dense = spline(t_dense)
    X_smooth_at_t = spline(t_sorted)

  ### return X_smooth back in the caller's original time order
  inverse = np.argsort(order)
  return t_dense, mu_dense, X_dense, X_smooth_at_t[inverse]
