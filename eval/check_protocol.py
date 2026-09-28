"""P1 gate: validate eval/protocol.json against PROJECT_SPEC (d, f3, f4) and PLAN P1. Prints PASS/FAIL."""
import hashlib, json, sys
from collections import Counter

p = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "eval/protocol.json"))
fails = []
def need(ok, msg):
    if not ok:
        fails.append(msg)

train = {o["object_id"]: o for o in p["objects"]["train"]}
held = {o["object_id"]: o for o in p["objects"]["heldout"]}
hp, g, cases = p["hole_pose"], p["groups"], p["cases"]

need(p["success"] == {"control_hz": 10, "max_steps": 150, "min_depth_mm": 20.0, "hole_depth_mm": 30.0},
     "success criterion differs from PROJECT_SPEC d")
need(len(train) == 8 and len(held) == 3, "expected 8 train / 3 heldout objects")
need(not set(train) & set(held), "object_id overlap train/heldout")
need(not {o["shape"] for o in train.values()} & {o["shape"] for o in held.values()},
     "heldout cross-section shape also appears in training (f4 needs substantively different geometry)")
need(hp["f3_tilt_deg"] not in hp["train_tilt_deg"] and hp["extrap_tilt_deg"] not in hp["train_tilt_deg"],
     "held-out tilt overlaps training tilts")
need(len({c["case_id"] for c in cases}) == len(cases), "duplicate case_id")
cnt = Counter(c["group"] for c in cases)
need(cnt["train_control"] == g["train_control"] and cnt["f3"] == g["f3"] and cnt["extrap"] == g["extrap"],
     f"group counts mismatch: {dict(cnt)}")
f4 = Counter(c["object_id"] for c in cases if c["group"] == "f4")
need(set(f4) == set(held) and all(v == g["f4_per_object"] for v in f4.values()), f"f4 per-object counts {dict(f4)}")
for c in cases:
    grp, obj, tilt = c["group"], c["object_id"], c["tilt_deg"]
    seen, tr_tilt = obj in train, tilt in hp["train_tilt_deg"]
    ok = {"train_control": seen and tr_tilt, "f3": seen and tilt == hp["f3_tilt_deg"],
          "f4": obj in held and tr_tilt, "extrap": seen and tilt == hp["extrap_tilt_deg"]}.get(grp, False)
    need(ok, f"case {c['case_id']} violates its group definition")
    (x0, x1), (y0, y1) = hp["hole_xy_range_m"]
    need(x0 <= c["hole_xy"][0] <= x1 and y0 <= c["hole_xy"][1] <= y1, f"case {c['case_id']} hole_xy out of range")
need(p["demos"]["total"] == p["demos"]["per_train_object"] * len(train), "demo total != per_train_object x train objects")
need(p["thresholds"]["f3_min_success_distinct_cases"] == 3 and p["thresholds"]["f4_min_success_per_object"] == 1,
     "floor thresholds differ from f3/f4")
need(p["results_must_share_config_hash"] is True, "config_hash rule missing")

from eval.protocol_util import protocol_sha
print("protocol_sha256", protocol_sha(p), "| cases", dict(cnt))
print("FAIL\n  " + "\n  ".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
