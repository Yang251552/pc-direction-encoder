"""Generate the frozen evaluation protocol (eval/protocol.json). Run once, before any training/eval."""
import json, random
from datetime import datetime

SEED = 20260928
TRAIN = [("round", 16, 1), ("round", 22, 2), ("square", 16, 2), ("square", 22, 1),
         ("hex", 16, 1), ("hex", 22, 2), ("octagon", 16, 2), ("octagon", 22, 1)]
HELDOUT = [("rect", [16, 22], 1), ("triangle", 24, 2), ("pentagon", 20, 1)]
TRAIN_TILTS, F3_TILT, EXTRAP_TILT = [0, 10, 20], 15, 30
# Workspace box chosen for FR3 reach; re-freeze (before any policy eval) if A1/A2 find it unreachable.
XY_RANGE = [[0.45, 0.60], [-0.15, 0.15]]


def oid(shape, size, ch):
    s = "x".join(map(str, size)) if isinstance(size, list) else str(size)
    return f"{shape}{s}c{ch}"


def objects(spec):
    return [{"object_id": oid(*o), "shape": o[0], "size_mm": o[1], "chamfer_mm": o[2]} for o in spec]


def main():
    rng = random.Random(SEED)
    train, held = objects(TRAIN), objects(HELDOUT)

    def case(group, obj, tilt):
        return {"group": group, "object_id": obj["object_id"], "tilt_deg": tilt,
                "yaw_deg": round(rng.uniform(0, 360), 2),
                "hole_xy": [round(rng.uniform(*XY_RANGE[0]), 4), round(rng.uniform(*XY_RANGE[1]), 4)],
                "init_seed": rng.randrange(2**31)}

    cases = []
    cases += [case("train_control", rng.choice(train), rng.choice(TRAIN_TILTS)) for _ in range(10)]
    cases += [case("f3", rng.choice(train), F3_TILT) for _ in range(20)]
    cases += [case("f4", o, rng.choice(TRAIN_TILTS)) for o in held for _ in range(10)]
    cases += [case("extrap", rng.choice(train), EXTRAP_TILT) for _ in range(10)]
    for i, c in enumerate(cases):
        c["case_id"] = f"{c['group']}-{i:03d}"

    protocol = {
        "version": 1, "master_seed": SEED, "frozen_at": datetime.now().isoformat(timespec="minutes"),
        "success": {"control_hz": 10, "max_steps": 150, "min_depth_mm": 20.0, "hole_depth_mm": 30.0},
        "geometry": {"clearance_mm_per_side": 1.0,
                     "size_def": "circumscribed diameter; rect gives both side lengths"},
        "objects": {"train": train, "heldout": held},
        "hole_pose": {"train_tilt_deg": TRAIN_TILTS, "f3_tilt_deg": F3_TILT, "extrap_tilt_deg": EXTRAP_TILT,
                      "yaw_deg_range": [0, 360], "hole_xy_range_m": XY_RANGE},
        "demos": {"per_train_object": 15, "total": 15 * len(train)},
        "groups": {"train_control": 10, "f3": 20, "f4_per_object": 10, "extrap": 10},
        "thresholds": {"sensitivity_mean_abs_da": 0.05, "overfit_steps": 500, "overfit_loss_ratio": 0.1,
                       "shuffle_control_loss_ratio": 2.0, "f3_min_success_distinct_cases": 3,
                       "f4_min_success_per_object": 1},
        "primary_metric": "success rate on f3 + f4 cases",
        "wording_rule": {"superior": "pc_success > rgb_success and one-sided Fisher exact p < 0.05 on f3+f4",
                         "preliminary_lead": "pc_success > rgb_success and p >= 0.05",
                         "not_superior": "otherwise"},
        "results_must_share_config_hash": True,
        "cases": cases,
    }
    json.dump(protocol, open("eval/protocol.json", "w"), indent=1)
    print(f"wrote eval/protocol.json: {len(cases)} cases")


if __name__ == "__main__":
    main()
