"""Rule grid search, signal deduplication, and rule selection."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from config import (HORIZON, MIN_DEDUP_SIGNALS, MIN_RAW_SIGNALS, N_SELECT,
                    OVERLAP_MAX, RATE_TIERS, SCREEN_TOP)
from features import CONDITION_GROUPS, Panel


def wilson_lb(hits: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - adj) / denom


def dedup(mask: np.ndarray, gap: int = HORIZON) -> np.ndarray:
    """Keep only signals spaced >= gap rows apart per ticker.

    Removes the overlapping-window double counting: each kept signal's
    HORIZON-day outcome is independent in time of the previous kept signal
    for the same stock.
    """
    out = np.zeros_like(mask)
    for j in range(mask.shape[1]):
        idx = np.flatnonzero(mask[:, j])
        last = -gap
        for i in idx:
            if i - last >= gap:
                out[i, j] = True
                last = i
    return out


@dataclass
class RuleStats:
    rule: list[str]
    n: int
    hits: int

    @property
    def rate(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def lb(self) -> float:
        return wilson_lb(self.hits, self.n)


def _screen_rules(panel: Panel, base: np.ndarray) -> list[RuleStats]:
    """Raw screen over the full condition grid with subtree pruning.

    Counts are on overlapping signal-days (fast but N is inflated); used
    only for ranking. Honest stats come from dedup evaluation afterwards.
    """
    hit = base & (panel.fwd > 0)
    groups = list(CONDITION_GROUPS.values())
    results: list[RuleStats] = []

    def recurse(level: int, mask: np.ndarray, names: list[str]):
        if level == len(groups):
            n = int(np.count_nonzero(mask))
            h = int(np.count_nonzero(mask & hit))
            results.append(RuleStats(list(names), n, h))
            return
        for cond in groups[level]:
            if cond is None:
                recurse(level + 1, mask, names)
            else:
                sub = mask & panel.conds[cond]
                if np.count_nonzero(sub) < MIN_RAW_SIGNALS:
                    continue  # adding conditions only shrinks; prune subtree
                recurse(level + 1, sub, names + [cond])

    recurse(0, base.copy(), [])
    return results


def evaluate_rule(panel: Panel, rule: list[str], region: np.ndarray) -> RuleStats:
    """Honest (non-overlapping) stats for a rule within a row region."""
    mask = panel.valid & region
    for name in rule:
        mask &= panel.conds[name]
    d = dedup(mask)
    n = int(np.count_nonzero(d))
    h = int(np.count_nonzero(d & (panel.fwd > 0)))
    return RuleStats(rule, n, h)


def _raw_signal_set(panel: Panel, rule: list[str],
                    base: np.ndarray) -> np.ndarray:
    mask = base.copy()
    for name in rule:
        mask &= panel.conds[name]
    return mask


def select_rules(panel: Panel, train_region: np.ndarray,
                 verbose: bool = False) -> list[RuleStats]:
    """Pick the N_SELECT best rules on the training region.

    Candidates (deduplicated, n >= MIN_DEDUP_SIGNALS) are taken from the
    highest hit-rate tier downward, ranked by Wilson lower bound within a
    tier, and added greedily subject to a signal-overlap cap so the ensemble
    is not three copies of the same setup.
    """
    base = panel.valid & train_region
    raw = _screen_rules(panel, base)
    raw.sort(key=lambda r: r.lb, reverse=True)
    candidates = raw[:SCREEN_TOP]

    scored = [evaluate_rule(panel, c.rule, train_region) for c in candidates]
    scored = [s for s in scored if s.n >= MIN_DEDUP_SIGNALS]

    chosen: list[RuleStats] = []
    chosen_masks: list[np.ndarray] = []
    for tier in RATE_TIERS:
        pool = [s for s in scored if s.rate >= tier
                and not any(s.rule == c.rule for c in chosen)]
        pool.sort(key=lambda s: s.lb, reverse=True)
        for s in pool:
            if len(chosen) >= N_SELECT:
                break
            m = _raw_signal_set(panel, s.rule, base)
            n_m = np.count_nonzero(m)
            too_similar = any(
                np.count_nonzero(m & cm) / max(min(n_m, np.count_nonzero(cm)), 1)
                > OVERLAP_MAX
                for cm in chosen_masks
            )
            if too_similar:
                continue
            chosen.append(s)
            chosen_masks.append(m)
        if len(chosen) >= N_SELECT:
            break

    if not chosen and scored:
        # No rule clears the lowest tier: fall back to the single most
        # statistically defensible rule rather than recommending nothing
        # forever.
        chosen = [max(scored, key=lambda s: s.lb)]

    if verbose:
        for s in chosen:
            print(f"  rule={s.rule} n={s.n} rate={s.rate:.3f} lb={s.lb:.3f}")
    return chosen


def union_mask(panel: Panel, rules: list[list[str]],
               region: np.ndarray) -> np.ndarray:
    """Deduplicated union of rule signals within a region."""
    u = np.zeros_like(region)
    for rule in rules:
        m = panel.valid_hist & region
        for name in rule:
            m &= panel.conds[name]
        u |= m
    return dedup(u)
