"""Command-line front end for submitting a validation ranking.

    python submission/client.py --team "Team Alpha" --ranking 17193 16737 20285 16742 17191

(the ids after --ranking are just an example order — put your own ranking there)

The real implementation lives in :mod:`reward_shaping.submission`; the notebook imports it
from there. This file exists so the client is runnable without opening the notebook.
"""

import argparse

from reward_shaping.submission import (  # noqa: F401  (re-exported for convenience)
  API_URL, validate_ranking_shape, submit_validation_ranking,
)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--team', required=True, help='your team name')
  parser.add_argument('--ranking', required=True, nargs='+', metavar='ID',
                      help='the five validation experiment ids, best first')
  parser.add_argument('--api-url', default=None,
                      help='override the configured server URL (default: %s)' % API_URL)
  args = parser.parse_args(argv)

  submit_validation_ranking(team=args.team, ranking=args.ranking, api_url=args.api_url)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
