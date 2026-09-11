# E. coli Ranking Challenge — scoring server

FastAPI service that scores validation submissions against the **hidden** expert ranking and
serves a projector leaderboard. The hidden ranking lives **only** in the
`VALIDATION_EXPERT_RANKING` environment variable — never in this repository.

## Endpoints

| method | path | who | purpose |
|---|---|---|---|
| `POST` | `/api/submit` | participants | submit a validation ranking, get Kendall τ-b back |
| `GET`  | `/leaderboard` | projector | HTML leaderboard, auto-refresh every 5 s |
| `GET`  | `/api/leaderboard` | anyone | JSON leaderboard (best τ per team) |
| `GET`  | `/healthz` | monitoring | `{"status":"ok"}` |
| `GET`  | `/api/admin/teams` | organiser | all teams + submission counts |
| `GET`  | `/api/admin/submissions` | organiser | full submission history (no expert order) |
| `POST` | `/api/admin/competition` | organiser | `{"enabled": true|false}` — open/close submissions |
| `POST` | `/api/admin/reset` | organiser | `{"confirm":"reset"}` — wipe teams + submissions |

Admin endpoints require header `X-Admin-Token: <ADMIN_TOKEN>`. If `ADMIN_TOKEN` is unset the
admin API returns `503`.

`POST /api/submit` rejections (never reveal anything about the expert order):

| status | body `error` | when |
|---|---|---|
| 403 | `submissions_disabled` | competition closed by the organiser |
| 422 | `invalid_ranking` | not exactly the 5 validation ids, each once |
| 429 | `rate_limited` (+ `retry_after_seconds`) | within the cooldown of your last submission |
| 429 | `submission_limit_reached` (+ `submissions_used`, `max_submissions`) | used all your submissions |

## Configuration (environment variables)

| var | default | notes |
|---|---|---|
| `APP_ENV` | `development` | `production` makes the server refuse to start without `VALIDATION_EXPERT_RANKING` |
| `DATABASE_URL` | `sqlite:///./server_data/challenge.db` | any SQLAlchemy URL; use a path on a persistent disk in prod |
| `ADMIN_TOKEN` | *(empty)* | required to use the admin API |
| `SUBMISSION_COOLDOWN_SECONDS` | `60` | min seconds between a team's submissions |
| `MAX_SUBMISSIONS_PER_TEAM` | `20` | hard cap per team |
| `VALIDATION_EXPERT_RANKING` | *(empty)* | **the secret** — see below |

### `VALIDATION_EXPERT_RANKING` format

A JSON list, best first. Flat:

```
["<id>","<id>","<id>","<id>","<id>"]
```

or grouped, where an inner list is a tie the expert could not separate:

```
[["<id>"],["<id>","<id>"],["<id>"],["<id>"]]
```

It must contain exactly the five ids `16737 16742 17191 17193 20285`, each once, in the
expert's order (which is not written down anywhere in this repository — paste it straight
into the Render dashboard). In development, leaving it empty falls back to a
clearly-labelled **dummy** order (ascending numeric) with a warning — fine for local
testing, never for a real competition.

## Local development

```bash
cd server
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # dummy secret, local admin token; .env is gitignored
uvicorn app.main:app --reload
```

Then `http://localhost:8000/leaderboard` and:

```bash
curl -X POST localhost:8000/api/submit -H 'content-type: application/json' \
  -d '{"team":"Team Alpha","ranking":["16737","16742","17191","17193","20285"]}'
```

Run the tests:

```bash
pytest
```

## Deploy to Render

1. New **Web Service** from this repository. Set **Root Directory** to `server`
   (`render.yaml` already does this if you deploy via Blueprint).
2. Build command `pip install -r requirements.txt`; start command
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Add a **persistent disk**, mount path `/var/data`, ≥ 1 GB, and set
   `DATABASE_URL=sqlite:////var/data/challenge.db` (four slashes = absolute path) so teams
   and submissions survive redeploys.
4. Set environment variables in the Render dashboard (not in any file):
   - `APP_ENV=production`
   - `ADMIN_TOKEN=` a long random string
   - `VALIDATION_EXPERT_RANKING=` the real expert order in the JSON format above
   - optionally override `SUBMISSION_COOLDOWN_SECONDS` / `MAX_SUBMISSIONS_PER_TEAM`
5. Deploy. Check `GET /healthz`. If `APP_ENV=production` and the ranking var is missing the
   service will fail to boot on purpose.
6. Put the service URL into `src/reward_shaping/submission.py` (`API_URL`) in the
   participant repo.

## Running the competition

```bash
# close submissions before discussing results
curl -X POST $URL/api/admin/competition -H "X-Admin-Token: $TOK" -d '{"enabled":false}'

# inspect everything
curl $URL/api/admin/submissions -H "X-Admin-Token: $TOK"

# start over (e.g. between sessions)
curl -X POST $URL/api/admin/reset -H "X-Admin-Token: $TOK" -d '{"confirm":"reset"}'
```

The expert ranking is never returned by any endpoint, admin included — it is only in the
environment and only used to compute τ.
