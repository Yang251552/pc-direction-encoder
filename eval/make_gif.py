"""E1: GIF of one evaluated case, traceable to a results JSON.

  .venv/bin/python -m eval.make_gif --policy <run_dir> --case <case_id> --results <results.json> \
      --out figures/<name>.gif [--n-points N]

The case is re-run with eval.run.rollout itself (same torch.manual_seed(init_seed), same
policy.predict / env.step, torch threads = 2 as in eval.run); a thin env wrapper records every
step. Before any GIF is written the re-run must match the case's entry in --results exactly
(success, depth_mm, steps, max_force_n), and the results file must belong to --policy
(weights_sha256, config_hash, no ablation flags); otherwise exit 1.
--n-points: observation point count of the re-run (default: results["n_points"] if recorded,
else sim.env.N_POINTS).
Frames every 2 control steps (0.2 s of sim time per frame, played in real time):
  left   scene: FR3 + hole box, camera follows the midpoint of tip and mouth (fixed azimuth/elevation);
  middle observed point cloud, orthographic view with the scene camera's azimuth and a fixed 45 deg
         elevation, zoomed to the cloud's extent over the episode; normals, insertion axis, tip;
  right  Fz (F/T sensor frame) and insertion depth vs time (20 mm success line), current time marked.
The group's successes/attempts in the title are counted from --results. Display-only quantities
(camera placement, depth curve, axis arrow position) use the simulator's ground truth; the policy
inputs are exactly what eval.run feeds it.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sim.env as simenv  # noqa: E402
from eval.run import PROTOCOL_PATH, rollout  # noqa: E402
from policy.model import load_policy  # noqa: E402
from sim.features import KAPPA_FLAT  # noqa: E402

EVERY = 2  # control steps per GIF frame
SCENE = dict(azimuth=120.0, elevation=-14.0, distance=1.05)  # FR3 base, arm and hole box in view
SCENE_OFFSET = np.array([-0.12, 0.0, 0.20])  # lookat = midpoint(tip, mouth) + offset
CLOUD_ELEV = -45.0  # point-cloud panel: orthographic, scene azimuth, this elevation
PANEL = 400  # px, scene render
N_NORMALS = 110  # normals drawn for the first N points (FPS order -> spread over the cloud)
NORMAL_LEN = 0.005  # m
DEPTH_FLOOR_MM = -60.0  # depth plot lower limit (the approach starts ~100-140 mm out)
MAX_BYTES = 8_000_000


class Recorder:
    """Pass-through env for eval.run.rollout that records what the policy saw."""

    def __init__(self, env):
        self.env, self.frames, self.fz, self.depth, self.k = env, [], [], [], 0

    def reset(self, case):
        obs = self.env.reset(case)
        self.k, self.frames, self.fz, self.depth = 0, [], [], []
        self._record(obs, final=False)
        return obs

    def step(self, a):
        obs, done, info = self.env.step(a)
        self.k += 1
        self._record(obs, final=done)
        return obs, done, info

    def _record(self, obs, final):
        self.fz.append(float(obs["wrench"][2]))
        self.depth.append(1e3 * self.env.insertion_depth()[0])  # display only
        if self.k % EVERY == 0 or final:
            self.frames.append({"k": self.k, "rgb": scene_rgb(self.env), "pc": obs["pc"].copy(),
                                "tip": obs["agent_pos"][:3].astype(float).copy()})


def scene_rgb(env):
    tip = env.tip_pose()[0]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = 0.5 * (tip + env.mouth) + SCENE_OFFSET
    cam.distance, cam.azimuth, cam.elevation = SCENE["distance"], SCENE["azimuth"], SCENE["elevation"]
    r = env.renderer(PANEL, PANEL, kind="free")
    r.update_scene(env.data, camera=cam)
    return r.render().copy()


def ortho_basis():
    """(right, up) of an orthographic view with the scene camera's azimuth and CLOUD_ELEV."""
    az, el = np.deg2rad(SCENE["azimuth"]), np.deg2rad(CLOUD_ELEV)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    return np.cross(fwd, up), up, fwd


def results_entry(results, case_id, policy, run_dir):
    """The case's attempt + the group tally, after checking the results belong to this policy."""
    err = []
    wsha = {f: hashlib.sha256((Path(run_dir) / f).read_bytes()).hexdigest()[:16] for f in ("encoder.pt", "policy.pt")}
    if results.get("weights_sha256") != wsha:
        err.append(f"weights_sha256 {results.get('weights_sha256')} != {run_dir} {wsha}")
    if results.get("config_hash") != policy.config_hash:
        err.append("config_hash differs from the policy's")
    if results.get("ablate_pc") or results.get("ablate_wrench"):
        err.append("results come from an ablation run")
    hits = [a for a in results.get("attempts", []) if a["case_id"] == case_id]
    if len(hits) != 1:
        err.append(f"{case_id}: {len(hits)} entries in results")
        return None, None, err
    group = [a for a in results["attempts"] if a["group"] == hits[0]["group"]]
    tally = (sum(bool(a["success"]) for a in group), len(group))
    s = results.get("summary", {}).get(hits[0]["group"])
    if s and (s["success"], s["attempts"]) != tally:
        err.append(f"summary {s} disagrees with the attempts list {tally}")
    return hits[0], tally, err


def render_frames(rec, env, case, entry, tally, run_name):
    right, up, fwd = ortho_basis()
    P2 = lambda X: np.stack([X @ right, X @ up], -1)  # noqa: E731  (metres, orthographic)
    allp = np.concatenate([f["pc"][:, :3] for f in rec.frames] + [np.array([f["tip"] for f in rec.frames])]).astype(float)
    uv_all = P2(allp)
    lo, hi = np.percentile(uv_all, 0.5, 0), np.percentile(uv_all, 99.5, 0)
    c, half = (lo + hi) / 2, 0.56 * max(hi - lo) + 0.004  # square window, ~12 % margin
    ax_uv = P2(np.array([env.mouth - 0.055 * env.axis, env.mouth + 0.012 * env.axis]))
    t = np.arange(len(rec.fz)) / 10.0
    fz, depth = np.array(rec.fz), np.array(rec.depth)
    flo, fhi = min(fz.min(), -1.0), max(fz.max(), 1.0)
    head = (f"{case['case_id']} | {case['object_id']} | tilt {case['tilt_deg']}\u00b0 | "
            f"{case['group']}: {tally[0]}/{tally[1]} successes ({run_name}) | this attempt: "
            f"{'SUCCESS' if entry['success'] else 'FAIL'}, depth {entry['depth_mm']:.1f} mm, {entry['steps']} steps")
    out = []
    for fr in rec.frames:
        now = fr["k"] / 10.0
        fig = plt.figure(figsize=(11.6, 4.3), dpi=100)
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], hspace=0.12, wspace=0.12,
                              left=0.01, right=0.99, top=0.84, bottom=0.11)
        fig.suptitle(head, fontsize=9.5, y=0.985)
        a0, a1 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
        a2, a3 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
        a0.imshow(fr["rgb"])
        a0.set_title(f"FR3 in simulation   t = {now:.1f} s", fontsize=8.5)
        pc = fr["pc"].astype(float)
        uv = P2(pc[:, :3])
        order = np.argsort(pc[:, :3] @ fwd)[::-1]  # far first
        k1 = pc[:, 9]
        col = np.where((k1 < KAPPA_FLAT)[:, None], [[0.62, 0.62, 0.62, 1.0]],
                       plt.cm.turbo(np.clip(np.log10(np.maximum(k1, 1.0)) / np.log10(300.0), 0, 1)))
        a1.set_facecolor((0.08, 0.08, 0.1))
        a1.scatter(uv[order, 0], uv[order, 1], s=9, c=col[order], linewidths=0)
        dn = P2(pc[:N_NORMALS, 3:6]) * NORMAL_LEN
        a1.quiver(uv[:N_NORMALS, 0], uv[:N_NORMALS, 1], dn[:, 0], dn[:, 1], angles="xy", scale_units="xy",
                  scale=1, color=(0.95, 0.95, 0.95, 0.9), width=0.004, headwidth=3.5, headlength=4.0,
                  headaxislength=3.5, zorder=3)
        a1.annotate("", xy=ax_uv[1], xytext=ax_uv[0],
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.9", color=(1.0, 0.82, 0.0), lw=3.0),
                    zorder=5)
        tip = P2(fr["tip"])
        a1.plot(tip[0], tip[1], marker="x", ms=10, mew=2.5, color=(0.2, 0.9, 1.0), zorder=6)
        a1.set_xlim(c[0] - half, c[0] + half)
        a1.set_ylim(c[1] - half, c[1] + half)
        a1.set_aspect("equal")
        a1.set_title(f"policy input: point cloud ({len(pc)} pts), orthographic, scene azimuth, "
                     f"{-CLOUD_ELEV:.0f}\u00b0 down\ndots |k1| (grey = flat) | white: normals | "
                     f"yellow: insertion axis | cyan x: tip", fontsize=7.2)
        a2.plot(t, fz, lw=1.1, color="tab:blue")
        a2.axhline(0, color="0.6", lw=0.6)
        a2.axvline(now, color="tab:red", lw=1.2)
        a2.set_xlim(0, max(t[-1], 0.1))
        a2.set_ylim(flo - 0.08 * (fhi - flo), fhi + 0.08 * (fhi - flo))
        a2.set_ylabel("Fz [N]", fontsize=8)
        a2.set_title("Fz, F/T sensor frame (+z = peg forward; push-back < 0)", fontsize=7.5)
        a2.tick_params(labelsize=7, labelbottom=False)
        a3.plot(t, np.maximum(depth, DEPTH_FLOOR_MM), lw=1.1, color="tab:purple")
        a3.axhline(20.0, color="tab:green", ls="--", lw=1.2, label="success: 20 mm")
        a3.axhline(30.0, color="0.45", ls=":", lw=1.0, label="hole bottom: 30 mm")
        a3.axhline(0.0, color="0.6", lw=0.6)
        a3.axvline(now, color="tab:red", lw=1.2)
        a3.set_xlim(0, max(t[-1], 0.1))
        a3.set_ylim(DEPTH_FLOOR_MM, 34)
        a3.set_ylabel("depth [mm]", fontsize=8)
        a3.set_xlabel("time [s]", fontsize=8)
        a3.text(0.01, 0.04, f"tip past the mouth plane along the hole axis (clipped at {DEPTH_FLOOR_MM:.0f})",
                transform=a3.transAxes, fontsize=6.5, color="0.35")
        a3.legend(fontsize=6.5, loc="upper left", frameon=False)
        a3.tick_params(labelsize=7)
        for a in (a0, a1):
            a.set_xticks([])
            a.set_yticks([])
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return out


def write_gif(frames, path):
    """Write, shrinking (then dropping every other frame) until the file is <= MAX_BYTES."""
    scale, stride = 1.0, 1
    while True:
        fr = frames[::stride]
        if scale < 1.0:
            import cv2
            fr = [cv2.resize(f, (int(f.shape[1] * scale), int(f.shape[0] * scale)), interpolation=cv2.INTER_AREA) for f in fr]
        imageio.mimsave(path, fr, duration=0.2 * stride, loop=0)
        size = path.stat().st_size
        if size <= MAX_BYTES:
            return size, scale, stride
        if scale > 0.6:
            scale *= 0.85
        else:
            stride *= 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-points", type=int, default=None, help="point count the results were produced with")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)  # as eval.run
    protocol = json.loads(PROTOCOL_PATH.read_text())
    case = next((c for c in protocol["cases"] if c["case_id"] == a.case), None)
    if case is None:
        sys.exit(f"ERROR: case {a.case} not in the protocol")
    policy = load_policy(a.policy)
    results = json.loads(Path(a.results).read_text())
    entry, tally, err = results_entry(results, a.case, policy, a.policy)
    if err:
        sys.exit("ERROR (results do not belong to this policy/case):\n  " + "\n  ".join(err))
    n_points = a.n_points or results.get("n_points")
    if n_points:
        simenv.N_POINTS = n_points  # read at call time by PegEnv.point_cloud
    env = simenv.PegEnv()
    rec = Recorder(env)
    r = rollout(rec, policy, case)
    diff = {k: (r[k], entry[k]) for k in ("success", "depth_mm", "steps", "max_force_n") if r[k] != entry[k]}
    if diff:
        sys.exit(f"ERROR: re-run of {a.case} does not reproduce {a.results} (re-run, results): {diff}. No GIF written.")
    print(f"traceability OK: {a.case} success={r['success']} depth_mm={r['depth_mm']} steps={r['steps']} "
          f"max_force_n={r['max_force_n']} == {Path(a.results).name}; group {case['group']} {tally[0]}/{tally[1]}")
    frames = render_frames(rec, env, case, entry, tally, Path(a.policy).name)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size, scale, stride = write_gif(frames, out)
    print(f"wrote {out} ({size / 1e6:.2f} MB, {len(frames[::stride])} frames, every {EVERY * stride} control steps, "
          f"scale {scale:.2f})")


if __name__ == "__main__":
    main()
