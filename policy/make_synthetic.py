"""Synthetic stand-in for data/demos.npz (same contract, SPEC 5), for the B-phase checks only.

Built so the expert's action depends on each input the checks probe, with no shortcut:
- hole position: visible ONLY in xyz (centre of the gap in the top-face patch);
- hole axis (tilted <= 30 deg): the top face tilts with the hole, so the tilt is in the face
  normals (and, much less directly for a PointNet, in the face xyz). The axis channel here carries
  a FIXED nominal axis (0, 0, -1) for every episode, i.e. no tilt information, so the policy has to
  read the tilt from normal / pcurv channels (in the real data the axis channel is the hole axis);
- wrench: a per-episode external wrench bias (synthetic), visible ONLY in the wrench. The robot is
  compliant, so the expert pre-compensates: the commanded pose (= action label) is its desired pose
  shifted by CF * force and rotated by CT * torque, while the pose actually reached (= next
  agent_pos) is the desired one. The bias is therefore absent from the pose history and every
  action differs from the uncompensated one by the full offset.
Expert = P-controller (gain K) to a pre-insert pose, then along the axis; executed with DART-style
noise (2 mm / 2 deg), labels stay clean. Contact force along the axis once past the hole mouth.
With many episodes neither the hole, the axis nor the bias can be memorised per episode, and the
low gain K keeps the noisy pose history a poor estimate of the target (error ~ noise * sqrt2 / K).
RGB (96x96) is a point-splat render of the same scene from a fixed camera (f5 baseline).
make() also reports an oracle: mean |delta a| (normalised) between the compensated action and the
uncompensated desired pose, i.e. what a perfect policy shows in the zero-wrench test.

  .venv/bin/python -m policy.make_synthetic --out runs/synthetic/demos_synth.npz --episodes 100
"""
import argparse
import json
from pathlib import Path

import numpy as np

R_HOLE, FACE, PEG_LEN, PEG_R = 0.012, 0.05, 0.08, 0.008
CF, CT = 0.006, 3.0             # compliance compensation: m per N, rad per Nm
K, DART_POS, DART_ROT, Z0 = 0.2, 0.002, np.deg2rad(2), 0.1
CAM, LOOK, IMG, FOV = np.array([0.0, -0.35, 0.55]), np.array([0.0, 0.0, 0.08]), 96, np.deg2rad(60)
COLORS = {"table": (200, 200, 200), "face": (60, 90, 200), "hole": (10, 10, 10), "peg": (220, 40, 40)}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frame_from_z(z):
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = unit(a - a.dot(z) * z)
    return np.stack([x, np.cross(z, x), z], axis=1)  # columns x, y, z


def pose9(pos, R):
    return np.concatenate([pos, R[:, 0], R[:, 1]])  # position + first two rotation columns


def sign_fix(d, ins):
    """Contract convention for pcurv_dir: dot(d, axis) >= 0, tie -> dot(d, world x) >= 0."""
    s = d @ ins
    s = np.where(np.abs(s) < 1e-9, d[:, 0], s)
    return d * np.where(s < 0, -1.0, 1.0)[:, None]


def surfaces(rng, n_face, n_hole, n_table, n_peg, hole, out_ax, peg_pos, peg_z, ins):
    """Per-surface (xyz, normal, pcurv_dir, kappa) arrays. The top face is the plane through the
    hole centre with normal out_ax (the box tilts with the hole)."""
    F = frame_from_z(out_ax)
    uv = rng.uniform(-FACE, FACE, (4 * n_face + 8, 2))
    uv = uv[np.hypot(uv[:, 0], uv[:, 1]) > R_HOLE][:n_face]
    up = np.array([0, 0, 1.0])
    z3 = lambda n: np.zeros((n, 3))  # noqa: E731
    out = {"face": (hole + uv @ F[:, :2].T, np.tile(out_ax, (n_face, 1)), z3(n_face), np.zeros((n_face, 2)))}
    r = R_HOLE * np.sqrt(rng.uniform(0, 1, n_hole))
    th = rng.uniform(0, 2 * np.pi, n_hole)
    out["hole"] = (hole + np.c_[r * np.cos(th), r * np.sin(th)] @ F[:, :2].T, np.tile(out_ax, (n_hole, 1)),
                   z3(n_hole), np.zeros((n_hole, 2)))
    out["table"] = (np.c_[rng.uniform(-0.25, 0.25, (n_table, 2)), np.zeros(n_table)],
                    np.tile(up, (n_table, 1)), z3(n_table), np.zeros((n_table, 2)))
    P = frame_from_z(peg_z)
    ph = rng.uniform(0, 2 * np.pi, n_peg)
    radial = np.cos(ph)[:, None] * P[:, 0] + np.sin(ph)[:, None] * P[:, 1]
    peg = peg_pos + rng.uniform(0, PEG_LEN, (n_peg, 1)) * peg_z + PEG_R * radial
    normal = radial * np.where(((CAM - peg) * radial).sum(1) < 0, -1.0, 1.0)[:, None]  # face camera
    tang = -np.sin(ph)[:, None] * P[:, 0] + np.cos(ph)[:, None] * P[:, 1]
    out["peg"] = (peg, normal, sign_fix(tang, ins), np.tile([1 / PEG_R, 0.0], (n_peg, 1)))
    return out


def cloud(rng, n, hole, out_ax, peg_pos, peg_z, ins):
    n_face, n_table = n // 2, n // 4
    s = surfaces(rng, n_face, 0, n_table, n - n_face - n_table, hole, out_ax, peg_pos, peg_z, ins)
    pts = np.vstack([np.hstack(v) for k, v in s.items() if k != "hole"])
    pts[:, :3] += rng.normal(0, 0.001, (n, 3))
    pts = np.hstack([pts, np.tile(ins, (n, 1))])
    return pts[rng.permutation(n)].astype(np.float32)


def camera():
    f = unit(LOOK - CAM)
    r = unit(np.cross(f, [0, 0, 1.0]))
    return np.stack([r, np.cross(f, r), f]), IMG / 2 / np.tan(FOV / 2)


def render(rng, hole, out_ax, peg_pos, peg_z, ins):
    """Point-splat render (painter's order, 2x2 splats) of table, face, hole, peg."""
    Rc, fpx = camera()
    s = surfaces(rng, 3000, 300, 4000, 1500, hole, out_ax, peg_pos, peg_z, ins)
    xyz = np.vstack([v[0] for v in s.values()])
    col = np.vstack([np.tile(COLORS[k], (len(v[0]), 1)) for k, v in s.items()]).astype(np.uint8)
    pc = (xyz - CAM) @ Rc.T
    u = (fpx * pc[:, 0] / pc[:, 2] + IMG / 2).astype(int)
    v = (fpx * pc[:, 1] / pc[:, 2] + IMG / 2).astype(int)
    order = np.argsort(-pc[:, 2])  # far first, near overwrites
    img = np.zeros((IMG, IMG, 3), np.uint8)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u[order] + du, v[order] + dv
            ok = (uu >= 0) & (uu < IMG) & (vv >= 0) & (vv < IMG)
            img[vv[ok], uu[ok]] = col[order][ok]
    return img


def expert(pos, z, phase, pre, goal, z_goal):
    """One clean expert step -> (desired pos, desired z, phase)."""
    if phase == 0 and np.linalg.norm(pre - pos) < 0.005:
        phase = 1
    d = (pre if phase == 0 else goal) - pos
    step = K * d
    step *= min(1.0, (0.015 if phase == 0 else 0.004) / (np.linalg.norm(step) + 1e-9))
    return pos + step, unit(z + K * (z_goal - z)), phase


def episode(rng, n_points, max_steps, insert_steps=25):
    hole = rng.uniform([-0.15, -0.15, 0.05], [0.15, 0.15, 0.15])
    tilt, az = np.deg2rad(rng.uniform(0, 30)), rng.uniform(0, 2 * np.pi)
    out_ax = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    ins = np.array([0.0, 0.0, -1.0])  # nominal insertion axis channel: fixed, carries no tilt (see top)
    bias_f, bias_t = rng.normal(0, 2.0, 3), rng.normal(0, 0.05, 3)  # N, Nm (per episode)
    sp = (hole + 0.06 * out_ax, hole - 0.025 * out_ax, out_ax)       # pre, goal, z_goal
    pos = sp[0] + np.r_[rng.uniform(-0.06, 0.06, 2), rng.uniform(0.04, 0.12)]
    z = unit(out_ax + Z0 * rng.normal(size=3))
    pcs, rgbs, obs, acts, wr, oracle = [], [], [], [], [], []
    phase, n_ins = 0, 0
    for _ in range(max_steps):
        R = frame_from_z(z)
        pcs.append(cloud(rng, n_points, hole, out_ax, pos, z, ins))
        rgbs.append(render(rng, hole, out_ax, pos, z, ins))
        obs.append(pose9(pos, R))
        depth = (hole - pos) @ out_ax                       # > 0 once past the hole mouth
        f = bias_f + (out_ax * (2 + 200 * depth) if depth > 0 else 0) + rng.normal(0, 0.05, 3)
        wr.append(np.r_[R.T @ f, R.T @ (bias_t + rng.normal(0, 0.002, 3))])  # sensor frame
        pos_d, z_d, phase = expert(pos, z, phase, *sp)
        z_c = unit(z_d + np.cross(CT * bias_t, z_d))        # compensate the compliant deflection
        acts.append(pose9(pos_d + CF * bias_f, frame_from_z(z_c)))
        oracle.append(np.abs(acts[-1] - pose9(pos_d, frame_from_z(z_d))))
        pos = pos_d + rng.normal(0, DART_POS, 3)            # reached pose = desired + DART noise
        z = unit(z_d + np.cross(rng.normal(0, DART_ROT, 3), z_d))
        n_ins += phase
        if n_ins >= insert_steps:
            break
    meta = dict(hole=hole.round(4).tolist(), axis=(-out_ax).round(4).tolist(), success=True, length=len(obs))
    return pcs, rgbs, obs, acts, wr, oracle, meta


def make(n_episodes=100, max_steps=150, n_points=256, seed=0):
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("pc", "rgb", "agent_pos", "action", "wrench", "oracle")}
    ends, metas = [], []
    for _ in range(n_episodes):
        *vals, m = episode(rng, n_points, max_steps)
        for k, v in zip(cols, vals):
            cols[k] += v
        ends.append(len(cols["pc"]))
        metas.append(m)
    oracle = np.asarray(cols.pop("oracle"))
    data = {k: np.asarray(v, dtype=np.uint8 if k == "rgb" else np.float32) for k, v in cols.items()}
    span = data["action"].max(0) - data["action"].min(0)
    scale = np.where(span < 1e-4, 0.0, 2.0 / np.maximum(span, 1e-4))  # same rule as policy.model.MinMax
    oracle_zero_wrench = float((oracle * scale).mean())
    data["episode_ends"] = np.asarray(ends, dtype=np.int64)
    data["geom_id"] = np.zeros(n_episodes, dtype=np.int32)
    data["meta"] = json.dumps(dict(synthetic=True, control_hz=10, oracle_zero_wrench=oracle_zero_wrench,
                                   episodes=metas))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = make(a.episodes, n_points=a.points, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **d)
    L = np.diff(np.r_[0, d["episode_ends"]])
    print(f"{a.out}: pc {d['pc'].shape} rgb {d['rgb'].shape}, episodes {len(L)}, "
          f"length min/mean/max {L.min()}/{L.mean():.1f}/{L.max()}, "
          f"oracle zero-wrench mean |da| {json.loads(str(d['meta']))['oracle_zero_wrench']:.4f}")


if __name__ == "__main__":
    main()
