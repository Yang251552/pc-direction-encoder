"""Closed-loop evaluation on the frozen protocol. Every attempt is written to the results JSON.

  .venv/bin/python -m eval.run --policy runs/send_pc --group train_control
  .venv/bin/python -m eval.run --policy runs/send_pc --all
  .venv/bin/python -m eval.run --policy runs/send_pc --group f3 --ablate-pc   # D1 closed-loop ablation

--ablate-wrench (report only) feeds the policy zero wrench.
--ablate-pc feeds the policy a mismatched point cloud: the first observation of another case in the
same group (different hole pose), frozen for the whole episode. Everything else (pose, wrench) is real.
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch

from policy.model import load_policy
from sim.env import PegEnv, MAX_STEPS, N_POINTS

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "eval/protocol.json"


from eval.protocol_util import protocol_sha  # noqa: E402


def rollout(env, policy, case, fake_pc=None, zero_wrench=False):
    torch.manual_seed(case["init_seed"])  # diffusion noise reproducible per case
    obs = env.reset(case)
    hist, info, max_f, k, lost = [obs], {"success": False, "depth_mm": float("-inf")}, 0.0, 0, False
    while k < MAX_STEPS:
        h = hist[-policy.n_obs_steps:]
        oh = {key: np.stack([o[key] for o in h]) for key in policy.obs_keys()}
        if fake_pc is not None:
            oh["pc"] = np.stack([fake_pc] * len(h))
        if zero_wrench:
            oh["wrench"] = np.zeros_like(oh["wrench"])
        for a in policy.predict(oh):
            try:
                obs, done, info = env.step(a)
            except AssertionError:  # the tip-centred crop lost the part (< 32 points): the attempt has failed
                d, _ = env.insertion_depth()
                info, done, lost = {"success": False, "depth_mm": 1000.0 * d}, True, True
                k += 1
                break
            k += 1
            max_f = max(max_f, float(np.linalg.norm(obs["wrench"][:3])))
            hist.append(obs)
            if done:
                break
        if done:
            break
    return {"case_id": case["case_id"], "group": case["group"], "object_id": case["object_id"],
            "tilt_deg": case["tilt_deg"], "success": bool(info["success"]),
            "depth_mm": round(float(info["depth_mm"]), 2), "steps": k, "max_force_n": round(max_f, 2), "lost_view": lost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--group")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--ablate-pc", action="store_true")
    ap.add_argument("--ablate-wrench", action="store_true", help="policy sees zero wrench (report-only ablation)")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(2)

    protocol = json.loads(PROTOCOL_PATH.read_text())
    cases = [c for c in protocol["cases"] if a.all or c["group"] == a.group]
    policy = load_policy(a.policy)
    env = PegEnv()
    fakes = {}
    if a.ablate_pc:
        assert policy.visual_key == "pc", "--ablate-pc only applies to point cloud policies"
        for i, c in enumerate(cases):  # partner = another case of the same group (different hole pose), same object
            same = [o for o in cases if o["group"] == c["group"] and o["case_id"] != c["case_id"]]
            for j in range(len(same)):  # a few object/pose combinations leave < N points in the crop: try the next
                try:
                    fakes[c["case_id"]] = env.reset(dict(same[(i + j) % len(same)], object_id=c["object_id"]))["pc"]
                    break
                except AssertionError:
                    continue

        missing = [c["case_id"] for c in cases if c["case_id"] not in fakes]
        assert not missing, f"--ablate-pc: no valid partner point cloud for {missing}"
    t0, attempts = time.time(), []
    for c in cases:
        r = rollout(env, policy, c, fakes.get(c["case_id"]), a.ablate_wrench)
        attempts.append(r)
        print(f"{r['case_id']:>20} {r['object_id']:>14} tilt {r['tilt_deg']:>4} "
              f"{'OK ' if r['success'] else 'FAIL'} depth {r['depth_mm']:7.2f} mm steps {r['steps']:3d} "
              f"maxF {r['max_force_n']:6.2f} N", flush=True)

    summary = {}
    for r in attempts:
        s = summary.setdefault(r["group"], {"success": 0, "attempts": 0})
        s["success"] += r["success"]
        s["attempts"] += 1
    run = Path(a.policy).name
    wsha = {f: hashlib.sha256((Path(a.policy) / f).read_bytes()).hexdigest()[:16] for f in ("encoder.pt", "policy.pt")}
    out = Path(a.out or ROOT / f"results/{run}{'_ablate_pc' if a.ablate_pc else ''}{'_ablate_wrench' if a.ablate_wrench else ''}"
               f"_{'all' if a.all else a.group}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": run, "visual": policy.visual_key, "config_hash": policy.config_hash, "weights_sha256": wsha,
                               "protocol_sha256": protocol_sha(protocol), "n_points": N_POINTS, "ablate_pc": a.ablate_pc, "ablate_wrench": a.ablate_wrench,
                               "seconds": round(time.time() - t0, 1), "summary": summary,
                               "attempts": attempts}, indent=1))
    print("summary", json.dumps(summary), "->", out)


if __name__ == "__main__":
    main()
