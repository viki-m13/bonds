"""Stage 2: momentum-dominant blends (volume as tilt / veto / gate), k=3."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data, protocol
import signals_volume as sv

P = data.build_panel()

VARIANTS = {
    # tilt: mom rank + w * accum rank
    "vol_tilt_updown_w25":  lambda: sv.mom_tilt(P, "updown", 0.25),
    "vol_tilt_updown_w50":  lambda: sv.mom_tilt(P, "updown", 0.50),
    "vol_tilt_hv_w25":      lambda: sv.mom_tilt(P, "hv", 0.25),
    "vol_tilt_hv_w50":      lambda: sv.mom_tilt(P, "hv", 0.50),
    "vol_tilt_chaikin_w25": lambda: sv.mom_tilt(P, "chaikin", 0.25),
    "vol_tilt_fp_w25":      lambda: sv.mom_tilt(P, "footprints", 0.25),
    "vol_tilt_obvdiv_w25":  lambda: sv.mom_tilt(P, "obv_div", 0.25),
    # veto: momentum, distribution names pushed to bottom
    "vol_veto_updown":      lambda: sv.mom_veto(P, "updown"),
    "vol_veto_chaikin":     lambda: sv.mom_veto(P, "chaikin"),
    "vol_veto_fp":          lambda: sv.mom_veto(P, "footprints"),
    # gate: accumulation rank within top momentum names
    "vol_gate_updown_q80":  lambda: sv.mom_gate_accum(P, "updown", 0.8),
    "vol_gate_updown_q90":  lambda: sv.mom_gate_accum(P, "updown", 0.9),
    "vol_gate_hv_q80":      lambda: sv.mom_gate_accum(P, "hv", 0.8),
    "vol_gate_chaikin_q80": lambda: sv.mom_gate_accum(P, "chaikin", 0.8),
    "vol_gate_fp_q80":      lambda: sv.mom_gate_accum(P, "footprints", 0.8),
}

for name, build in VARIANTS.items():
    t0 = time.time()
    protocol.evaluate_signal(build(), name, k=3)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
print("STAGE2 DONE")
