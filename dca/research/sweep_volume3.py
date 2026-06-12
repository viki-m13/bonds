"""Stage 3: k-sweep {1,2,5} for the most promising volume-family signals."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, protocol
import signals_volume as sv

P = data.build_panel()
CAND = {
    "vol_veto_updown":  sv.mom_veto(P, "updown"),
    "vol_veto_chaikin": sv.mom_veto(P, "chaikin"),
    "vol_tilt_hv_w25":  sv.mom_tilt(P, "hv", 0.25),
    "vol_hv_x_mom":     sv.accum_x_momentum(P, "hv"),
}
for name, scores in CAND.items():
    for k in (1, 2, 5):
        protocol.evaluate_signal(scores, f"{name}_k{k}", k=k)
print("STAGE3 DONE")
