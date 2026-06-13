"""Sweep the volume/accumulation signal family at k=3 (stage 1)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data, protocol
import signals_volume as sv

P = data.build_panel()

VARIANTS = {
    # 1. high-volume premium
    "vol_hv_gate_v20":   lambda: sv.hv_premium(P, 20, 120, 20, "gate"),
    "vol_hv_gate_v10":   lambda: sv.hv_premium(P, 10, 120, 20, "gate"),
    "vol_hv_interact":   lambda: sv.hv_premium(P, 20, 120, 20, "interact"),
    # 2. up/down volume ratio
    "vol_updown_21":     lambda: sv.updown_ratio(P, 21),
    "vol_updown_63":     lambda: sv.updown_ratio(P, 63),
    "vol_updown_21_dlr": lambda: sv.updown_ratio(P, 21, dollar=True),
    "vol_updown_63_dlr": lambda: sv.updown_ratio(P, 63, dollar=True),
    # 3. OBV
    "vol_obv_trend_63":  lambda: sv.obv_trend(P, 63),
    "vol_obv_div_126":   lambda: sv.obv_divergence(P, 126),
    # 4. money flow
    "vol_chaikin_21":    lambda: sv.chaikin_flow(P, 21),
    "vol_chaikin_63":    lambda: sv.chaikin_flow(P, 63),
    # 5. dry-up in uptrend
    "vol_dryup":         lambda: sv.dryup_uptrend(P),
    "vol_dryup_90":      lambda: sv.dryup_uptrend(P, near_high=0.90),
    # 6. footprints
    "vol_fp_63_m2":      lambda: sv.footprints(P, 63, 2.0),
    "vol_fp_63_m15":     lambda: sv.footprints(P, 63, 1.5),
    "vol_fp_126_m2":     lambda: sv.footprints(P, 126, 2.0),
    # 7. interactions with 6m momentum
    "vol_fp_x_mom":      lambda: sv.accum_x_momentum(P, "footprints"),
    "vol_chaikin_x_mom": lambda: sv.accum_x_momentum(P, "chaikin"),
    "vol_updown_x_mom":  lambda: sv.accum_x_momentum(P, "updown"),
    "vol_obvdiv_x_mom":  lambda: sv.accum_x_momentum(P, "obv_div"),
    "vol_hv_x_mom":      lambda: sv.accum_x_momentum(P, "hv"),
    "vol_chaikin_p_mom": lambda: sv.accum_plus_momentum(P, "chaikin"),
    "vol_fp_p_mom":      lambda: sv.accum_plus_momentum(P, "footprints"),
}

for name, build in VARIANTS.items():
    t0 = time.time()
    scores = build()
    protocol.evaluate_signal(scores, name, k=3)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
