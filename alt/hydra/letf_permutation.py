"""Phase 3 validation — permutation null test.

Null hypothesis: observed Sharpe of DD-throttled winner is no better than
what you'd get by shuffling signal/weights timing.

Method (signal-block shuffle):
  - Keep daily asset returns intact (preserve cross-sectional structure)
  - Shuffle the daily W (weights) matrix along the TIME axis in contiguous
    blocks of length = rebal_days (21 days). This preserves the one-month
    holding structure but destroys the timing of when each weight is held.
  - Compute the portfolio return for each shuffle, summarise its Sharpe.

Then also do a simpler overlay-level test:
  - Take base TSMOM K=3m tv=15% as given.
  - Randomise the DD-throttle multiplier by shuffling it in 21-day blocks.
  - If the observed DD-overlay Sharpe beats >95% of nulls => real alpha
    from the overlay (not coincidental path-length effect).

Pre-registered alpha = 0.05 (two-sided), 500 permutations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import summarise
from letf_tsmom import tsmom_with_vol_target, prep as tsmom_prep
from letf_dd_throttle import apply_dd_throttle


OUT = Path("/home/user/bonds/data/results")
N_PERM = 500
BLOCK = 21
SEED = 20260421


def block_shuffle(arr, block, rng):
    n = len(arr)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n_blocks) * block
    order = rng.permutation(n_blocks)
    out = np.empty(n, dtype=arr.dtype)
    cursor = 0
    for bi in order:
        s = starts[bi]
        e = min(s + block, n)
        seg = arr[s:e]
        out[cursor:cursor + len(seg)] = seg
        cursor += len(seg)
    return out


def sharpe(r):
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    return float(mu / sd) if sd > 0 else 0.0


def mdd(r):
    nav = (1 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())


def main():
    rng = np.random.default_rng(SEED)

    # Base strategy
    tsmom_px = tsmom_prep()
    base, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.15)

    # Observed DD-throttled
    overlay, m = apply_dd_throttle(base)
    sr_obs = sharpe(overlay)
    sr_base = sharpe(base)
    mdd_obs = mdd(overlay)
    mdd_base = mdd(base)
    print(f"Observed base SR:        {sr_base:.3f}   MDD: {mdd_base*100:+.2f}%")
    print(f"Observed DD-overlay SR:  {sr_obs:.3f}   MDD: {mdd_obs*100:+.2f}%")
    print(f"ΔSR (overlay lifts):     {sr_obs - sr_base:+.3f}")
    print(f"ΔMDD (overlay cuts):     {(mdd_obs - mdd_base)*100:+.2f}pp")

    # Permutation: shuffle the DD multiplier in 21d blocks
    # If the overlay's lift is real (the DD signal has timing content),
    # random-timed multipliers should on average NOT deliver the lift.
    m_arr = m.values
    base_arr = base.values
    null_sr = np.zeros(N_PERM)
    null_mdd = np.zeros(N_PERM)
    for i in range(N_PERM):
        m_perm = block_shuffle(m_arr, BLOCK, rng)
        perm_ret = pd.Series(m_perm * base_arr, index=base.index)
        null_sr[i] = sharpe(perm_ret)
        null_mdd[i] = mdd(perm_ret)

    null_dsr = null_sr - sr_base
    null_dmdd = null_mdd - mdd_base
    p_sr = (null_sr >= sr_obs).mean()
    p_mdd = (null_mdd >= mdd_obs).mean()  # higher = less-bad MDD
    print(f"\nPermutation null (shuffled DD multiplier, {N_PERM} perms, block={BLOCK}):")
    print(f"  null SR     mean={null_sr.mean():.3f}  sd={null_sr.std():.3f}  "
          f"p95={np.percentile(null_sr, 95):.3f}")
    print(f"  null ΔSR    mean={null_dsr.mean():+.3f}  sd={null_dsr.std():.3f}  "
          f"p95={np.percentile(null_dsr, 95):+.3f}")
    print(f"  null MDD    mean={null_mdd.mean()*100:+.2f}%  sd={null_mdd.std()*100:.2f}pp "
          f"p95={np.percentile(null_mdd, 95)*100:+.2f}%  best={null_mdd.max()*100:+.2f}%")
    print(f"  null ΔMDD   mean={null_dmdd.mean()*100:+.2f}pp  "
          f"p95={np.percentile(null_dmdd, 95)*100:+.2f}pp  best={null_dmdd.max()*100:+.2f}pp")
    print(f"  p(SR  >= observed) = {p_sr:.3f}")
    print(f"  p(MDD >= observed, i.e. overlay's MDD no worse) = {p_mdd:.3f}")

    out = pd.DataFrame({
        "permutation": np.arange(N_PERM),
        "null_sr": null_sr,
        "null_dsr": null_dsr,
        "null_mdd": null_mdd,
        "null_dmdd": null_dmdd,
    })
    out.to_csv(OUT / "letf_permutation.csv", index=False)

    mdd_ok = p_mdd < 0.05
    sr_ok = p_sr < 0.05
    print(f"\nVerdict @ α=0.05:")
    print(f"  Sharpe lift real? {'YES' if sr_ok else 'NO (noise-indistinguishable)'}")
    print(f"  MDD cut real?     {'YES' if mdd_ok else 'NO (noise-indistinguishable)'}")
    if mdd_ok and not sr_ok:
        print("  => DD-throttle delivers GENUINE MDD protection via path-dependent "
              "\n     timing, even if Sharpe lift itself is within sampling noise.")


if __name__ == "__main__":
    main()
