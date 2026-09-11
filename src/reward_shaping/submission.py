"""Client for submitting a validation ranking to the challenge leaderboard server.

This module holds **no** expert ranking --- it only packages your five validation ids and
POSTs them. The server computes Kendall's tau-b against its hidden expert order and returns
your score.

Point it at a different server by editing :data:`API_URL` below, passing ``api_url=`` to
:func:`submit_validation_ranking`, or setting the ``ECOLI_CHALLENGE_API_URL`` env var.
"""

import os

from .ranking import VALIDATION_IDS

__all__ = ['API_URL', 'validate_ranking_shape', 'submit_validation_ranking']

### The one place to configure the leaderboard server.
API_URL = 'https://reward-shaping-for-bioprocess-feeding.onrender.com'

_VALIDATION_ID_STRINGS = {str(eid) for eid in VALIDATION_IDS}


def _resolve_api_url(api_url=None):
  return (api_url or os.environ.get('ECOLI_CHALLENGE_API_URL') or API_URL).rstrip('/')


def validate_ranking_shape(ranking):
  """Local sanity check before hitting the server (the server re-validates authoritatively).

  :param ranking: a list of five validation experiment ids, best first (str or int).
  :returns: the ranking as a list of strings.
  :raises ValueError: if it is not exactly the five validation ids, each once.
  """
  as_str = [str(x).strip() for x in ranking]
  if len(as_str) != len(VALIDATION_IDS):
    raise ValueError(
      'ranking must have exactly %d entries, got %d' % (len(VALIDATION_IDS), len(as_str)))
  if len(set(as_str)) != len(as_str):
    raise ValueError('ranking has duplicate ids: %s' % (as_str, ))
  unknown = [x for x in as_str if x not in _VALIDATION_ID_STRINGS]
  if unknown:
    raise ValueError('not validation experiment ids: %s (expected %s)' % (
      unknown, sorted(_VALIDATION_ID_STRINGS)))
  return as_str


def submit_validation_ranking(team, ranking, api_url=None, timeout=20):
  """Submit ``ranking`` for ``team`` to the leaderboard server and print the outcome.

  :param team: your team name (free text; the server normalises case/whitespace).
  :param ranking: your ordering of the five validation experiments, best first.
  :param api_url: override the configured server URL.
  :returns: the server's JSON response as a dict. On a network failure returns
    ``{"accepted": False, "error": "unreachable", ...}`` rather than raising.
  """
  import requests

  ranking = validate_ranking_shape(ranking)
  if not str(team).strip():
    raise ValueError('team name must not be empty')

  url = _resolve_api_url(api_url) + '/api/submit'
  try:
    response = requests.post(
      url, json={'team': str(team).strip(), 'ranking': ranking}, timeout=timeout)
  except requests.RequestException as error:
    print('Submission failed: could not reach the server at %s\n  (%s)' % (url, error))
    return {'accepted': False, 'error': 'unreachable', 'detail': str(error)}

  try:
    body = response.json()
  except ValueError:
    body = {}

  _print_outcome(response.status_code, body)
  return body or {'accepted': False, 'error': 'bad_response',
                  'status_code': response.status_code}


def _print_outcome(status_code, body):
  error = body.get('error')

  if body.get('accepted'):
    tau = body.get('kendall_tau')
    pos = body.get('leaderboard_position')
    used = body.get('submissions_used')
    cap = body.get('max_submissions')
    cooldown = body.get('cooldown_seconds')
    print('Submission accepted.\n')
    if tau is not None:
      print('Validation Kendall tau: %s' % (tau, ))
    if pos is not None:
      print('Current leaderboard position: #%s' % (pos, ))
    if used is not None and cap is not None:
      print('Submissions used: %s / %s' % (used, cap))
    if cooldown is not None:
      print('Next submission available in %s seconds.' % (cooldown, ))
    return

  if error == 'rate_limited':
    retry = body.get('retry_after_seconds', '?')
    print('Submission rejected.\n\nYou can submit again in %s seconds.' % (retry, ))
  elif error == 'submission_limit_reached':
    cap = body.get('max_submissions', '?')
    print('Submission rejected.\n\n'
          'Your team has used all %s validation submissions.' % (cap, ))
  elif error == 'submissions_disabled':
    print('Submission rejected.\n\nThe competition is currently closed.')
  elif error == 'invalid_ranking':
    print('Submission rejected.\n\nThe server did not accept the ranking '
          '(it must be exactly the five validation experiment ids, each once).')
  else:
    print('Submission rejected (HTTP %s): %s' % (status_code, error or body))
