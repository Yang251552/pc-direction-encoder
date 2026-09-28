"""Translation-equivariance test for relative_pos (C3 repair r1).

  .venv/bin/python -m policy.test_relative [--data data/demos.npz]

Shift a scene by delta (point cloud xyz, agent_pos positions, action labels all + delta):
  1. every normalised network input (encoder input after preprocessing, agent_pos, wrench) and the
     normalised training target are bit-identical;
  2. sample_chunk() and predict() (same initial noise, DDIM eta = 0) return absolute poses whose
     positions move by exactly delta (tolerance 1 um, float32 add-back) and whose rotations are identical;
  3. control: with relative_pos = False the same shift changes the inputs and the actions do not follow.
Positions are snapped to a 2^-12 m grid and delta is a multiple of 2^-8 m, so float32 additions and
subtractions are exact and "identical" is tested bit for bit.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from policy.model import DiffusionPolicy
from policy.train import load_data, windows_for

ROOT = Path(__file__).resolve().parents[1]
DELTA = torch.tensor([16, -24, 8]) / 256.0  # m: (0.0625, -0.09375, 0.03125)


def snap(x):
    return np.round(x * 4096) / 4096


def shift(obs, action):
    o = {k: v.clone() for k, v in obs.items()}
    o["pc"][..., :3] += DELTA
    o["agent_pos"][..., :3] += DELTA
    a = action.clone()
    a[..., :3] += DELTA
    return o, a


def inputs(model, obs, action):
    rel = model.relative_obs(obs)
    return {"encoder_in": model.encoder.preprocess(rel["pc"]),
            "agent_pos_in": model.norm["agent_pos"].normalize(rel["agent_pos"]),
            "wrench_in": model.norm["wrench"].normalize(obs["wrench"]),
            "target": model.normalized_target(action, obs["agent_pos"])}


def check(relative, data, rows):
    torch.manual_seed(0)
    mcfg = dict(json.loads((ROOT / "configs" / "send_pc.json").read_text())["model"], relative_pos=relative)
    model = DiffusionPolicy(**mcfg)
    windows, _ = windows_for(data, model.config)
    model.fit_normalizers(data, windows)
    model.eval()
    idx = windows[rows]
    obs = {k: torch.from_numpy(data[k][idx[:, :model.n_obs_steps]]) for k in model.obs_keys()}
    action = torch.from_numpy(data["action"][idx])
    obs_s, action_s = shift(obs, action)
    a, b = inputs(model, obs, action), inputs(model, obs_s, action_s)
    diff = {k: float((a[k] - b[k]).abs().max()) for k in a}
    noise = torch.randn((len(rows), model.horizon, 9), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        c0, c1 = model.sample_chunk(obs, noise), model.sample_chunk(obs_s, noise)
        hist = {k: v[0].numpy() for k, v in obs.items()}
        hist_s = {k: v[0].numpy() for k, v in obs_s.items()}
        p0, p1 = model.predict(hist, noise[:1]), model.predict(hist_s, noise[:1])
    pos_err = max(float((c1[..., :3] - c0[..., :3] - DELTA).abs().max()),
                  float(np.abs(p1[:, :3] - p0[:, :3] - DELTA.numpy()).max()))
    rot_diff = max(float((c1[..., 3:] - c0[..., 3:]).abs().max()), float(np.abs(p1[:, 3:] - p0[:, 3:]).max()))
    return dict(bitwise_equal={k: v == 0.0 for k, v in diff.items()}, max_input_diff=diff,
                abs_pos_shift_err_m=pos_err, rot_diff=rot_diff, predict_shape=list(p0.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "demos.npz"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    data = load_data(a.data, "pc", 5)
    data["pc"][..., :3] = snap(data["pc"][..., :3])
    for k in ("agent_pos", "action"):
        data[k][:, :3] = snap(data[k][:, :3])
    mcfg = json.loads((ROOT / "configs" / "send_pc.json").read_text())["model"]
    rows = np.random.default_rng(0).choice(len(windows_for(data, mcfg)[0]), 16, replace=False)
    rel, ctrl = check(True, data, rows), check(False, data, rows)
    ok_rel = (all(rel["bitwise_equal"].values()) and rel["abs_pos_shift_err_m"] <= 1e-6 and rel["rot_diff"] == 0.0
              and rel["predict_shape"] == [mcfg["n_action_steps"], 9])
    ok_ctrl = ctrl["abs_pos_shift_err_m"] > 1e-3 and not ctrl["bitwise_equal"]["encoder_in"]
    print(("PASS " if ok_rel and ok_ctrl else "FAIL ") + json.dumps(dict(
        delta_m=DELTA.tolist(), windows=len(rows), relative=rel, control_absolute=dict(
            bitwise_equal=ctrl["bitwise_equal"], abs_pos_shift_err_m=ctrl["abs_pos_shift_err_m"]))))
    return ok_rel and ok_ctrl


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
