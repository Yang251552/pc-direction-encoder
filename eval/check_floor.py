"""D1 gate: showcase floor f1-f3 (gated; f4 reported only, user decision 09-28) + anti-degeneration checks on the committed weights. Prints PASS/FAIL.

  .venv/bin/python -m eval.check_floor results/pc.json --ablation results/send_pc_ablate_pc_f3.json

Thresholds come from eval/protocol.json; f5 (RGB) is interview evidence (user decision 09-28), not checked here.
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights/send_pc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--data", default="data/demos.npz")
    a = ap.parse_args()
    P = json.loads((ROOT / "eval/protocol.json").read_text())
    TH = P["thresholds"]
    res, abl = json.loads(Path(a.results).read_text()), json.loads(Path(a.ablation).read_text())
    fails, info = [], {}

    def need(ok, msg):
        if not ok:
            fails.append(msg)

    # provenance: one config, the committed weights, the frozen protocol, every case attempted once
    wsha = {f: hashlib.sha256((WEIGHTS / f).read_bytes()).hexdigest()[:16] for f in ("encoder.pt", "policy.pt")}
    for name, r in (("results", res), ("ablation", abl)):
        need(r["weights_sha256"] == wsha, f"{name}: weights sha256 != weights/send_pc")
    need(res["config_hash"] == abl["config_hash"], "results and ablation come from different configs")
    from eval.protocol_util import protocol_sha
    live = protocol_sha(P)
    for name, r in (("results", res), ("ablation", abl)):
        need(r.get("protocol_sha256") == live, f"{name}: protocol_sha256 {r.get('protocol_sha256')} != live protocol {live}")
    need(not res["ablate_pc"] and abl["ablate_pc"], "ablate_pc flags wrong")
    want = Counter(c["group"] for c in P["cases"])
    got = Counter(r["group"] for r in res["attempts"])
    need(got == want, f"attempt counts {dict(got)} != protocol {dict(want)}")
    need(len({r["case_id"] for r in res["attempts"]}) == len(res["attempts"]), "a case was attempted twice")

    ok = [r for r in res["attempts"] if r["success"]]
    f3 = {r["case_id"] for r in ok if r["group"] == "f3"}
    need(len(f3) >= TH["f3_min_success_distinct_cases"], f"f3: {len(f3)} distinct successes")
    held = [o["object_id"] for o in P["objects"]["heldout"]]
    f4 = Counter(r["object_id"] for r in ok if r["group"] == "f4")
    # f4 is reported, not gated: the user moved it to interview evidence on 09-28 (protocol amendment)
    info["f4_gate"] = "reported only (user decision 09-28)"
    info["f4_would_pass"] = all(f4[o] >= TH["f4_min_success_per_object"] for o in held)
    info["success/attempts"] = {g: f"{sum(r['success'] for r in res['attempts'] if r['group'] == g)}/{n}"
                                for g, n in want.items()}
    info["f4_per_object"] = {o: f"{f4[o]}/{sum(r['object_id'] == o for r in res['attempts'] if r['group'] == 'f4')}"
                             for o in held}

    # closed-loop ablation on f3: mismatched point cloud must at least halve the successes
    abl_f3 = sum(r["success"] for r in abl["attempts"] if r["group"] == "f3")
    need(len(f3) > 0 and abl_f3 <= TH["closed_loop_pc_ablation_max_success_ratio"] * len(f3),
         f"closed-loop ablation: {abl_f3} vs normal {len(f3)}")
    info["f3_ablation"] = f"{abl_f3} (normal {len(f3)})"

    # f1/f2 sensitivity + inference swap, on the committed weights
    from policy.checks import sensitivity, swap_loss
    from policy.model import load_policy
    from policy.train import load_data
    torch.set_num_threads(2)
    model = load_policy(WEIGHTS)
    need(model.config_hash == res["config_hash"], "weights config_hash != results config_hash")
    data = load_data(ROOT / a.data, "pc")
    own, swapped = swap_loss(model, data)
    s = sensitivity(model, data)
    th = TH["sensitivity_dpos_mm"]
    need(swapped / own >= TH["inference_swap_loss_ratio"], f"inference swap ratio {swapped / own:.2f}")
    need(s["determinism_max_abs"] == 0, "sampling not deterministic")
    need(s["swap_direction_mm"] >= th, f"f1 direction sensitivity {s['swap_direction_mm']:.3f} mm")
    need(s["zero_wrench_mm"] >= th, f"f2 wrench sensitivity (contact windows) {s['zero_wrench_mm']:.3f} mm")
    info.update(inference_swap_ratio=round(swapped / own, 1), f1_swap_direction_mm=round(s["swap_direction_mm"], 3),
                f2_zero_wrench_contact_mm=round(s["zero_wrench_mm"], 3), contact_windows=s["contact_windows"], config_hash=res["config_hash"][:16], weights=wsha)

    print(json.dumps(info, indent=1))
    print("FAIL\n  " + "\n  ".join(fails) if fails else "D1: PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
