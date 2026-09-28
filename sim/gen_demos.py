"""A8: generate data/demos.npz (PROJECT_SPEC §5 contract) with the DART expert, in parallel.

Usage: python -m sim.gen_demos [--out data/demos.npz] [--per-object N] [--workers 3]
  --per-object defaults to protocol demos.per_train_object.
Episodes are defined per slot (train object gid, k < per_object): the slot's own stream
default_rng([DEMO_SEED, gid, k]) draws fresh cases (train tilts {0,10,20}, yaw U(0,360), hole_xy
U(protocol range), init_seed never a protocol seed) until one succeeds; the DART noise stream is
default_rng([init_seed, 11]) and the aiming error comes from init_seed. The dataset is therefore
identical for any worker count; worker w takes slots[w::W] (disjoint slot streams -> disjoint
seeds) and the shards are merged in fixed (gid, k) order. Step k stores the observation at t_k and
the clean expert label computed from that same state; the executed (noisy) action takes the env
from t_k to t_k + 0.1 s. The npz is self-contained (arrays + a JSON meta string, no paths, no
pickles).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import cv2
import imageio
import numpy as np

from sim import expert as X
from sim import features as F
from sim import scene
from sim.env import CONTROL_DT, DEPTH_RES, MAX_DPOS, MAX_DROT, N_POINTS, RGB_RES, PegEnv

DEMO_SEED = 20260929  # != protocol master_seed 20260928
SIGMA_POS, SIGMA_ROT = 0.002, np.deg2rad(2.0)
MAX_ATTEMPTS_PER_SLOT = 10
OUT = scene.ROOT / "data/demos.npz"
FIGS = scene.ROOT / "sim/figs"
ALL_OBJECTS = scene.PROTOCOL["objects"]["train"] + scene.PROTOCOL["objects"]["heldout"]  # geom_id = index
TRAIN = [o["object_id"] for o in scene.PROTOCOL["objects"]["train"]]
KEYS = ("pc", "rgb", "agent_pos", "wrench", "action")


def viz(pc, cam_pos, cam_R, f_px, mouth, axis, title, path, size=600):
    """Point cloud (colour = |k1|, grey = flat) + normals (6 mm), projected with the bench camera
    and zoomed to the points; yellow arrow = insertion axis at the mouth."""
    def proj(X_):
        c = (X_ - cam_pos) @ cam_R
        return np.stack([f_px * c[:, 0] / -c[:, 2], -f_px * c[:, 1] / -c[:, 2]], 1), -c[:, 2]

    xyz, nrm, k1 = pc[:, :3].astype(float), pc[:, 3:6].astype(float), pc[:, 9].astype(float)
    uv, z = proj(xyz)
    uv_n, _ = proj(xyz + 0.006 * nrm)
    ax_uv, _ = proj(np.array([mouth - 0.05 * axis, mouth + 0.015 * axis]))
    lo, hi = uv.min(0), uv.max(0)
    s = 0.85 * size / max(hi - lo)
    off = size / 2 - s * (lo + hi) / 2
    P, Pn, A = (s * uv + off).astype(int), (s * uv_n + off).astype(int), (s * ax_uv + off).astype(int)
    img = np.full((size, size, 3), 25, np.uint8)
    val = np.clip(np.log10(np.maximum(k1, 1.0)) / np.log10(300.0), 0, 1)  # |k1| 1..300 /m
    col = cv2.applyColorMap((255 * val).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    for i in np.argsort(-z):  # far first
        c = (150, 150, 150) if k1[i] < F.KAPPA_FLAT else tuple(int(v) for v in col[i])
        cv2.line(img, tuple(P[i]), tuple(Pn[i]), (230, 230, 230), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(P[i]), 3, c, -1, cv2.LINE_AA)
    cv2.arrowedLine(img, tuple(A[0]), tuple(A[1]), (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.15)
    for j, line in enumerate([title, "dots: |k1| (grey = flat) | white: normals 6 mm | yellow: insertion axis"]):
        cv2.putText(img, line, (8, 20 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    imageio.imwrite(path, img[..., ::-1])  # OpenCV BGR -> RGB


def run_slots(args):
    """Worker: generate the given slots, write <shard_dir>/<w>_<key>.npy + <w>_meta.json."""
    w, slots, shard_dir = args
    env = PegEnv()
    ex = X.Expert(env, SIGMA_POS, SIGMA_ROT)
    buf, metas, t0 = {k: [] for k in KEYS}, [], time.time()
    for n, (gid, k) in enumerate(slots):
        rng, failed = np.random.default_rng([DEMO_SEED, gid, k]), []
        while True:
            if len(failed) >= MAX_ATTEMPTS_PER_SLOT:
                raise RuntimeError(f"slot {(gid, k)}: {len(failed)} failed attempts")
            case = X.random_case(rng, TRAIN[gid])
            ex.rng = np.random.default_rng([case["init_seed"], 11])
            r = X.run_episode(env, case, ex, record=True)
            if r["success"]:
                break
            failed.append({**case, "depth_mm": r["depth_mm"], "steps": r["steps"]})
        for obs, a in r["frames"]:
            for key in KEYS[:4]:
                buf[key].append(obs[key])
            buf["action"].append(a)
        metas.append({"slot": [gid, k], "failed": failed, "n": len(r["frames"]), "episode": {
            **case, "geom_id": gid, "mouth": env.mouth.tolist(), "axis": env.axis.tolist(), "R_box": env.R_box.tolist(),
            "success": True, "steps": r["steps"], "attempts": len(failed) + 1,
            "t": [float(o["t"]) for o, _ in r["frames"]], "full_sigma_steps": int(sum(r["noisy"])),
            "f_max_N": r["f_max"], "aim_err_mm": r["aim_mm"], "belief_err_end_mm": r["belief_err_mm"],
            "force_corrections": r["corr_steps"], "spiral_steps": r["spiral_steps"],
            "tip_lat_first_contact_mm": r["tip_lat_first_contact_mm"], "tip_lat_end_mm": r["tip_lat_end_mm"]}})
        if (n + 1) % 10 == 0 or n + 1 == len(slots):
            print(f"[w{w} {time.time() - t0:6.0f}s] {n + 1}/{len(slots)} slots", flush=True)
    for key in KEYS:
        np.save(Path(shard_dir) / f"{w}_{key}.npy", np.stack(buf[key]))
    (Path(shard_dir) / f"{w}_meta.json").write_text(json.dumps(metas))
    return w


def main(out=OUT, per_object=None, workers=3):
    t0 = time.time()
    per_object = per_object or scene.PROTOCOL["demos"]["per_train_object"]
    slots = [(g, k) for g in range(len(TRAIN)) for k in range(per_object)]
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".gen_shards_") as tmp:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = "1"  # inherited by the spawned workers: `workers` threads in total
        with mp.get_context("spawn").Pool(workers) as pool:
            pool.map(run_slots, [(w, slots[w::workers], tmp) for w in range(workers)])
        gen_s = time.time() - t0
        recs = []  # (gid, k, worker, row offset in shard, n rows, meta)
        for w in range(workers):
            off = 0
            for m in json.loads((Path(tmp) / f"{w}_meta.json").read_text()):
                recs.append((*m["slot"], w, off, m["n"], m))
                off += m["n"]
        recs.sort(key=lambda r: (r[0], r[1]))
        S = sum(r[4] for r in recs)
        shards = {(w, key): np.load(Path(tmp) / f"{w}_{key}.npy", mmap_mode="r") for w in range(workers) for key in KEYS}
        arrays, row = {key: np.empty((S, *shards[(0, key)].shape[1:]), shards[(0, key)].dtype) for key in KEYS}, 0
        for gid, k, w, off, n, m in recs:
            for key in KEYS:
                arrays[key][row:row + n] = shards[(w, key)][off:off + n]
            row += n
        del shards
    episodes = [r[5]["episode"] for r in recs]
    failed = [f for r in recs for f in r[5]["failed"]]
    attempts = {o: sum(ep["attempts"] for ep in episodes if ep["object_id"] == o) for o in TRAIN}
    arrays["episode_ends"] = np.cumsum([r[4] for r in recs]).astype(np.int64)
    arrays["geom_id"] = np.array([r[0] for r in recs], np.int32)
    env = PegEnv()
    env.observe = lambda: None
    env.reset(X.random_case(np.random.default_rng(0), TRAIN[0]))
    cam_pos, cam_R = env.cam_pose()
    n_att = sum(attempts.values())
    meta = {
        "contract": "PROJECT_SPEC §5 v2",
        "protocol_sha256": hashlib.sha256((scene.ROOT / "eval/protocol.json").read_bytes()).hexdigest(),
        "control_hz": 1.0 / CONTROL_DT, "physics_dt": scene.DT,
        "camera": {"name": "bench", "pos": cam_pos.tolist(), "R_world_from_cam": cam_R.tolist(),
                   "lookat": scene.CAM_LOOKAT.tolist(), "fovy_deg": scene.CAM_FOVY, "depth_res": DEPTH_RES,
                   "rgb_res": RGB_RES, "n_points": N_POINTS, "extrinsics": "sim ground truth"},
        "geometry": {"clearance_m": scene.CLR, "hole_depth_m": scene.H, "plate_thickness_m": scene.TB,
                     "wall_thickness_m": scene.TW, "z_mouth_m": scene.Z_MOUTH, "peg_len_m": scene.PEG_LEN,
                     "ft_thick_m": scene.FT_THICK, "n_round_walls": scene.N_ROUND,
                     "size_def": scene.PROTOCOL["geometry"]["size_def"]},
        "objects": {str(i): o for i, o in enumerate(ALL_OBJECTS)},
        "features": {"channels": ["x", "y", "z", "nx", "ny", "nz", "pdx", "pdy", "pdz", "k1_abs", "k2_abs",
                                  "ax", "ay", "az"], "frame": "world", "knn": F.KNN, "kappa_flat_per_m": F.KAPPA_FLAT,
                     "sign": "dot(pdir, axis) >= 0, else world x, else world y; flat -> 0",
                     "axis": "nominal insertion axis (task input)",
                     "crop": "vertical cylinder around the current tip x fixed world z band (no hole-pose truth)"},
        "pose_format": "tip position (3) + rotation matrix column 0 (3) + column 1 (3), world",
        "action": "clean expert label (absolute target tip pose); executed = label + DART noise",
        "wrench": "environment -> peg at ft_site, sensor frame, tool-subtree gravity removed",
        "expert": {"version": "v2 force-reactive: lateral aiming error, F/T compliant centring, spiral search",
                   "aim_err_m": list(X.AIM_ERR), "k_lat_m_per_N": X.K_LAT, "dc_max_m": X.DC_MAX,
                   "f_contact_N": X.F_CONTACT, "f_lat_min_N": X.F_LAT_MIN, "f_ax_hold_N": X.F_AX_HOLD,
                   "adv_m": {"free": X.ADV_FREE, "near": X.ADV_NEAR, "in": X.ADV_IN, "contact": X.ADV_CONTACT},
                   "slow_zone_m": [X.SLOW_LO, X.IN_DEPTH],
                   "pre_dist_m": X.PRE_DIST, "max_cmd_depth_m": X.MAX_CMD_DEPTH, "f_back_max_N": X.F_BACK_MAX,
                   "dart_sigma_pos_m_per_axis": SIGMA_POS, "dart_sigma_rot_rms_rad": SIGMA_ROT,
                   "dart_noise_gate_m": X.NOISE_GATE, "dart_in_gate_scale": X.IN_HOLE_SCALE,
                   "env_clamp": {"dpos_m": MAX_DPOS, "drot_rad": float(MAX_DROT)}},
        "seed": DEMO_SEED, "seeding": "slot stream [DEMO_SEED, gid, k]; DART noise [init_seed, 11]; aim [init_seed, 7]",
        "per_object": per_object, "workers": workers,
        "attempts": {"total": n_att, "per_object": attempts, "successes": len(episodes),
                     "expert_success_rate": len(episodes) / n_att, "failed": failed},
        "generation_seconds": gen_s,
        "episodes": episodes,
    }
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    # figures: three frames with different tilts, tip ~ at the mouth
    starts = np.r_[0, arrays["episode_ends"][:-1]]
    picked = []
    for tilt in (0.0, 10.0, 20.0):
        e = next(i for i, ep in enumerate(episodes) if ep["tilt_deg"] == tilt and i not in picked
                 and ep["geom_id"] not in [episodes[j]["geom_id"] for j in picked])
        picked.append(e)
        ep, s0 = episodes[e], starts[e]
        pos = arrays["agent_pos"][s0:arrays["episode_ends"][e], :3].astype(float)
        depth = (pos - np.array(ep["mouth"])) @ np.array(ep["axis"])
        k = int(np.argmin(np.abs(depth + 0.004)))  # tip ~4 mm outside the mouth
        f_px = DEPTH_RES / 2 / np.tan(np.deg2rad(scene.CAM_FOVY) / 2)
        viz(arrays["pc"][s0 + k], cam_pos, cam_R, f_px, np.array(ep["mouth"]), np.array(ep["axis"]),
            f"{ep['object_id']} tilt {ep['tilt_deg']:.0f} yaw {ep['yaw_deg']:.0f} step {k} depth {1e3 * depth[k]:.1f} mm",
            FIGS / f"a8_pc_{len(picked)}.png")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB): S={S} E={len(episodes)} (per object {per_object}, "
          f"{workers} workers); attempts {n_att}, expert success rate {len(episodes) / n_att:.3f}; "
          f"generation {gen_s:.0f} s, total incl. merge/compress {time.time() - t0:.0f} s; figs a8_pc_1..3.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-object", type=int, default=None, help="default: protocol demos.per_train_object")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    main(a.out.resolve(), a.per_object, a.workers)
