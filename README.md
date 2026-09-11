# Reward shaping for bioprocess feeding

A summer-school exercise **and ranking competition**. You are given fed-batch *E. coli*
cultivation runs and must **design a reward function** — a formula that rewards the good
aspects of a run and penalises the bad ones — so that ranking the runs by your reward
reproduces an expert's ranking.

## Two stages

```
TRAINING                                  VALIDATION
--------                                  ----------
6 experiments                             5 experiments
20294 20293 17201 16738 16748 16736       16737 16742 17191 17193 20285
Expert ranking is KNOWN.                  Expert ranking is HIDDEN (on the server).
You evaluate LOCALLY in the notebook.     You SUBMIT a ranking; the server scores it.
Metric: Kendall's tau-b.                  Metric: Kendall's tau-b. Leaderboard.
```

The **competition score is the validation Kendall tau-b**, computed server-side against a
ranking that is **not in this repository in any form**. The training stage is only for
developing and sanity-checking your reward function.

Training expert ranking (best → worst): `20294, 20293, 17201, 16738, 16748, 16736`.
The validation ids above are in plain numeric order and carry no ranking information.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
jupyter lab notebooks/
```

Requires Python ≥ 3.9. Dependencies (`numpy`, `scipy`, `matplotlib`, `jupyter`, `requests`)
are installed by `pip install -e .`.

## What's here

```
data/training/    6 experiment_<id>.npz  (expert order known)
data/validation/  5 experiment_<id>.npz  (expert order hidden — data only, no labels)
src/reward_shaping/
  experiment_data.py  the data reader, ported from mode/scripts/experiment_data.py
  data.py             load_training(), load_validation(), build_reactors()
  growth.py           growth_rate(): smoothing-spline fit to biomass, then mu = (dX/dt)/X
  ranking.py          TRAINING_EXPERT_RANKING, evaluate_training(), kendall_tau_b(), ...
  submission.py       submit_validation_ranking(), API_URL  (the leaderboard client)
submission/client.py  the same client as a command-line script
notebooks/
  01_explore_measurements.ipynb   one 2x3 figure per reactor: X (+ spline), S, DOTm /
                                  growth rate, A, feed  (training then validation)
  02_reward_function_starter.ipynb  baseline reward, local training score, and a single
                                    cell that submits your validation ranking
figures/                PNGs written by notebook 01
server/                 the private scoring server (FastAPI) — see server/README.md
```

## Using the data

```python
from reward_shaping import load_training, load_validation, build_reactors, growth_rate
from reward_shaping import evaluate_training, ranking_from_scores, submit_validation_ranking

train = build_reactors(load_training())             # {id: reactor dict}
reactor = train[20294]

t, X    = reactor["observations"]["X"]              # biomass, g/L, on the reactor clock
t, S    = reactor["observations"]["S"]              # glucose, g/L
t, A    = reactor["observations"]["A"]              # acetate, g/L
t, DOTm = reactor["observations"]["DOTm"]           # dissolved O2, fraction of air saturation
pt, vol_L, conc_gL = reactor["pulses"]              # glucose boluses delivered
glucose_fed_g = float((vol_L * conc_gL).sum())
t_dense, mu, X_dense, X_at_samples = growth_rate(t, X)   # smoothing-spline growth rate, 1/h

# local training feedback
scores = {i: my_reward(r) for i, r in train.items()}
print(evaluate_training(scores)["summary"])         # -> "Training Kendall tau: ..."

# competition entry
val = build_reactors(load_validation())
ranking = [str(i) for i in ranking_from_scores({i: my_reward(r) for i, r in val.items()})]
submit_validation_ranking(team="Your Team", ranking=ranking)
```

`t = 0` for every reactor is its first OD600 reading, not the run's own clock zero;
readings before that or past the reactor's horizon are dropped. See the docstring of
`build_reactor` in `src/reward_shaping/experiment_data.py` for the details.

## The leaderboard

Submissions go to the server at `API_URL` (in `src/reward_shaping/submission.py`; the
organiser sets the deployed URL). The server computes Kendall's tau-b against its hidden
ranking and shows every team's **best** score at `API_URL + "/leaderboard"`. Limits: one
submission per team every 60 s, 20 per team total. See `server/README.md` to deploy it.

**This public repo contains no validation answers** — no validation expert order, no class
labels, no pairwise data. The hidden ranking exists only as an environment variable on the
server.

## Credits

The cultivation data and the data reader come from the
[`mode`](https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/mode) repository
(`scripts/experiment_data.py`, `scripts/plot_measurements.py`).
