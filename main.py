"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five
quantiles of its Exophase earn rate at the case's target, scored by pinball
loss on the logit scale.

Model
-----
Exophase's page is not live: the house's recorder shows it unchanged for
hours, then refreshed in one jump (launch night: 638 players tracked, every
rate flat from 2026-11-18T23:50Z to 2026-11-19T04:00Z, then 3,868 players
and every rate but the 100% one lower). A rate is earners / players-tracked,
so a refresh moves a rate through two forces: more players earning the
trophy (earners grow as the launch population plays) and more players being
tracked (dilution: the newly tracked cohort has earned far less).

Per trophy the model therefore works with the two counts, not the rate:
  * log earners E(t) and log players-tracked N(t) are each fitted by least
    squares against log(1 + hours since release), over the distinct page
    states in the history (consecutive identical reads are one state);
  * the earners slope is shrunk toward its tier mean and the global mean;
  * the forecast rate is E(target) / N(target), with the implied logit move
    from the last read shrunk by n/(n+3) and clamped to +/-2 logit, so a
    thin history cannot drag the median far from the last read.

Spread: the logit spread combines a random-walk term (0.5 * sqrt(days),
the house baseline's) with the pooled residual spread of the per-trophy
fits, which is where refresh jumps show up.

Flat history (no page change at all, as in both practice cases): the trend
fit is exactly flat, so a second component covers the refresh the stale
page has not shown yet. With probability p = 1 - exp(-horizon / 5h) the
page refreshes before the target; in that branch each rate is multiplied
by 0.25 (the launch-night refresh took the tracked base 638 -> 3,868 and
the non-100% rates to about a fifth of their old values, the new cohort
having earned almost nothing yet), with a 0.7 logit spread. The 100%
trophy is exempt: it cannot be diluted below 100% by players who all
hold it. Quantiles come from inverting the two-component mixture CDF
numerically, so the forecast is an honest mixture, not a shifted median.

Everything is deterministic: no randomness, no network, no clock reads
other than the case's own timestamps. Pure standard library.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an import
# of a module shipped beside this file would find it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (tuned only on the public practice description) ---
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
SIGMA_RW = 0.5          # logit spread per sqrt(day), the house baseline's
DELTA_SHRINK_K = 3.0    # trend delta shrunk by n / (n + K)
DELTA_CLAMP = 2.0       # |logit move from last rate| allowed from the trend
SLOPE_SHRINK = 2.0      # earners-slope shrinkage weight toward tier/global
REFRESH_GAP_H = 5.0     # prior mean gap between Exophase refreshes (hours)
REFRESH_DILUTION = 0.25  # rate multiplier in a refresh branch
REFRESH_SIGMA = 0.7     # logit spread of the refresh branch
SPREAD_MIN = 0.15
SPREAD_MAX = 2.0
TOP_SIGMA = 0.5         # logit spread for a trophy last seen at 100%


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg)) if lg < 700 else 100.0
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, residuals)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, []
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, [y - my for y in ys]
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    return a, b, [y - (a + b * x) for x, y in zip(xs, ys)]


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(mu0, s0, w1, mu1, s1, tau):
    """tau-quantile of (1-w1)*N(mu0,s0) + w1*N(mu1,s1), by bisection."""
    lo = min(mu0 - 10 * s0, mu1 - 10 * s1)
    hi = max(mu0 + 10 * s0, mu1 + 10 * s1)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = (1.0 - w1) * norm_cdf((mid - mu0) / s0) + w1 * norm_cdf((mid - mu1) / s1)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    x_target = math.log1p((parse_time(case["target"]) - RELEASE).total_seconds() / 3600.0)

    # Distinct page states per trophy: consecutive reads with the same
    # (players_tracked, rate) are one state; the page only moves at a refresh.
    series = []  # per trophy: list of (x, rate, players)
    n_states_global = 0
    for i in range(len(trophies)):
        states = []
        prev = None
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            key = (read["players_tracked"], rate)
            if key == prev:
                continue
            prev = key
            t = (parse_time(read["at"]) - RELEASE).total_seconds() / 3600.0
            states.append((math.log1p(max(t, 0.0)), float(rate), read["players_tracked"]))
        series.append(states)
        n_states_global = max(n_states_global, len(states))

    # Players-tracked growth, fitted once on the distinct N states.
    n_states = []
    prev_n = None
    for read in history:
        n = read["players_tracked"]
        if n == prev_n:
            continue
        prev_n = n
        t = (parse_time(read["at"]) - RELEASE).total_seconds() / 3600.0
        n_states.append((math.log1p(max(t, 0.0)), math.log(max(n, 1))))
    if len(n_states) >= 2:
        a_n, b_n, _ = ols([x for x, _ in n_states], [y for _, y in n_states])
    else:
        a_n, b_n = (math.log(max(history[-1]["players_tracked"], 1)), 0.0)
    log_n_target = a_n + b_n * x_target

    # Per-trophy log-earners slopes, with tier and global pooling.
    earn_slopes = []   # (tier, slope, n_states, intercept)
    for i, trophy in enumerate(trophies):
        xs, ys = [], []
        for x, rate, n in series[i]:
            e = n * rate / 100.0
            if e > 0:
                xs.append(x)
                ys.append(math.log(e))
        if len(xs) >= 2:
            a_e, b_e, _ = ols(xs, ys)
        else:
            a_e, b_e = (math.log(max(1.0, 0.01)), 0.0)
        earn_slopes.append((trophy["tier"], b_e, len(xs), a_e))

    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [b for (t, b, n, _) in earn_slopes if t == tier and n >= 2]
        if vals:
            tier_slopes[tier] = sum(vals) / len(vals)
    all_vals = [b for (_, b, n, _) in earn_slopes if n >= 2]
    global_slope = sum(all_vals) / len(all_vals) if all_vals else 0.0

    # Pooled residual spread of the logit trend fits (refresh-step noise).
    residuals = []
    for i in range(len(trophies)):
        xs = [x for x, _, _ in series[i]]
        ys = [logit(rate) for _, rate, _ in series[i]]
        if len(xs) >= 3:
            _, _, res = ols(xs, ys)
            residuals.extend(res)
    if len(residuals) >= 5:
        s_res = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    else:
        s_res = 0.0
    s_base = min(SPREAD_MAX, max(SPREAD_MIN,
                                  math.sqrt(SIGMA_RW ** 2 * horizon_days + s_res ** 2)))

    # Refresh branch, only when the page never moved in the whole history:
    # the stale page hides the next refresh.
    flat = n_states_global < 2
    horizon_h = case["horizon_minutes"] / 60.0
    p_refresh = 1.0 - math.exp(-horizon_h / REFRESH_GAP_H) if flat else 0.0

    # Fallback last rates (tier median) for trophies with no read at all.
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                rate = float(read["rates"][i])
                break
        last_rates.append(rate)
    for i, rate in enumerate(last_rates):
        if rate is None:
            same = [r for j, r in enumerate(last_rates)
                    if r is not None and trophies[j]["tier"] == trophies[i]["tier"]]
            pool = same or [r for r in last_rates if r is not None]
            last_rates[i] = sorted(pool)[len(pool) // 2] if pool else 0.01

    rows = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i]
        tier, b_e, n_e, a_e = earn_slopes[i]
        if last >= 99.99:
            # A trophy everyone holds: median stays 100, small spread.
            qs = [round4(min(100.0, inv_logit(logit(99.99) + z * TOP_SIGMA))) for z in Z]
            rows.append({"id": trophy["id"], "q": qs})
            continue
        # Trend median from the earners / players decomposition.
        if n_e >= 2:
            b_tier = tier_slopes.get(tier, global_slope)
            b_shrunk = (n_e * b_e + SLOPE_SHRINK * b_tier + SLOPE_SHRINK * global_slope) / (
                n_e + 2 * SLOPE_SHRINK)
            log_e_target = a_e + b_shrunk * x_target
            rate_trend = 100.0 * math.exp(log_e_target - log_n_target)
            rate_trend = clamp_rate(rate_trend)
            delta = logit(rate_trend) - logit(last)
            delta *= n_e / (n_e + DELTA_SHRINK_K)
            delta = max(-DELTA_CLAMP, min(DELTA_CLAMP, delta))
        else:
            delta = 0.0
        mu0 = logit(last) + delta
        if flat and p_refresh > 0.0:
            mu1 = logit(max(0.01, last * REFRESH_DILUTION))
            qs = [round4(min(100.0, max(0.01, inv_logit(
                mixture_quantile(mu0, s_base, p_refresh, mu1, REFRESH_SIGMA, tau)))))
                for tau in LEVELS]
        else:
            qs = [round4(min(100.0, max(0.01, inv_logit(mu0 + z * s_base)))) for z in Z]
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
