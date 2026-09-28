"""A8 acceptance: python -m sim.check_contract data/demos.npz [--per-object N]
(--per-object defaults to protocol demos.per_train_object)

Checks the file against the PROJECT_SPEC §5 data contract, then runs negative self-tests (a
held-out object, 15 / 30 deg tilts, NaN / inf values, shifted actions, shifted timestamps must all
be rejected). PASS = file has no violations AND every mutation is rejected.
"""
import argparse
import copy
import json
import re
import sys

import numpy as np

from sim import scene
from sim.env import N_POINTS
from sim.expert import PROTOCOL_SEEDS

KEYS = {"pc": (np.float32, (N_POINTS, 14)), "rgb": (np.uint8, (96, 96, 3)), "agent_pos": (np.float32, (9,)),
        "wrench": (np.float32, (6,)), "action": (np.float32, (9,))}
TRAIN = [o["object_id"] for o in scene.PROTOCOL["objects"]["train"]]
TRAIN_TILTS = set(float(t) for t in scene.PROTOCOL["hole_pose"]["train_tilt_deg"])


def load(path):
    with np.load(path, allow_pickle=False) as z:  # self-contained: plain arrays + a JSON string
        D = {k: z[k] for k in z.files}
    D["meta"] = json.loads(str(D["meta"]))
    return D


def _rot_ok(v):
    c0, c1 = v[:, 3:6].astype(float), v[:, 6:9].astype(float)
    return (np.abs(np.linalg.norm(c0, axis=1) - 1).max() < 1e-4 and np.abs(np.linalg.norm(c1, axis=1) - 1).max() < 1e-4
            and np.abs(np.einsum("ij,ij->i", c0, c1)).max() < 1e-4)


def check(D, per_object=None):
    per_object = per_object or scene.PROTOCOL["demos"]["per_train_object"]
    err = []
    meta = D.get("meta", {})
    if meta.get("camera", {}).get("n_points") != N_POINTS:
        err.append(f"meta.camera.n_points {meta.get('camera', {}).get('n_points')} != env N_POINTS {N_POINTS}")
    if re.search(r"(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\\\)", json.dumps(meta)):
        err.append("meta contains a local absolute path (file must be portable)")
    if set(D) != set(KEYS) | {"episode_ends", "geom_id", "meta"}:
        return [f"keys {sorted(D)}"], float("nan")
    S = len(D["action"])
    for k, (dt, shp) in KEYS.items():
        if D[k].dtype != dt or D[k].shape != (S, *shp):
            err.append(f"{k}: {D[k].dtype} {D[k].shape}, want {np.dtype(dt)} (S,{shp})")
    ends, gid = D["episode_ends"], D["geom_id"]
    if ends.dtype != np.int64 or gid.dtype != np.int32 or ends.shape != gid.shape:
        err.append(f"episode_ends {ends.dtype}{ends.shape} / geom_id {gid.dtype}{gid.shape}")
    if len(ends) == 0 or ends[-1] != S or np.any(np.diff(np.r_[0, ends]) <= 0):
        err.append("episode_ends not strictly increasing / last != S")
    for k in KEYS:
        if k != "rgb" and not np.all(np.isfinite(D[k])):
            err.append(f"{k}: non-finite values ({int(np.sum(~np.isfinite(D[k])))})")
    eps = meta.get("episodes", [])
    if len(eps) != len(ends):
        return err + [f"meta.episodes {len(eps)} != E {len(ends)}"], float("nan")
    objs = meta.get("objects", {})
    # objects: training geometry only
    for e, (g, ep) in enumerate(zip(gid, eps)):
        oid = objs.get(str(int(g)), {}).get("object_id")
        if oid not in TRAIN or ep["object_id"] != oid or ep["geom_id"] != int(g):
            err.append(f"episode {e}: geom_id {int(g)} -> {oid}, meta object {ep['object_id']} (not a train object / mismatch)")
    # hole poses: train tilts only, protocol xy range, fresh seeds
    (x0, x1), (y0, y1) = scene.PROTOCOL["hole_pose"]["hole_xy_range_m"]
    for e, ep in enumerate(eps):
        if float(ep["tilt_deg"]) not in TRAIN_TILTS:
            err.append(f"episode {e}: tilt {ep['tilt_deg']} not in train tilts {sorted(TRAIN_TILTS)}")
        if not (x0 <= ep["hole_xy"][0] <= x1 and y0 <= ep["hole_xy"][1] <= y1 and 0 <= ep["yaw_deg"] < 360):
            err.append(f"episode {e}: hole pose out of range")
        if ep["init_seed"] in PROTOCOL_SEEDS:
            err.append(f"episode {e}: init_seed reuses a protocol seed")
        if not ep.get("success"):
            err.append(f"episode {e}: not a success")
    if len({ep["init_seed"] for ep in eps}) != len(eps):
        err.append("duplicate init_seed")
    # counts
    counts = {o: sum(ep["object_id"] == o for ep in eps) for o in TRAIN}
    if len(eps) != per_object * len(TRAIN) or set(counts.values()) != {per_object}:
        err.append(f"episode counts {counts}")
    att = meta.get("attempts", {})
    if not (att.get("total", 0) >= len(eps) and 0 < att.get("expert_success_rate", 0) <= 1):
        err.append("meta.attempts missing / inconsistent")
    # time alignment: obs t_k = k / control_hz within every episode, same clock for obs and wrench
    hz = scene.PROTOCOL["success"]["control_hz"]
    if abs(meta.get("control_hz", 0) - hz) > 1e-9:
        err.append(f"control_hz {meta.get('control_hz')} != {hz}")
    starts = np.r_[0, ends[:-1]]
    for e, (s0, s1, ep) in enumerate(zip(starts, ends, eps)):
        t = np.asarray(ep.get("t", []), float)
        if len(t) != s1 - s0 or np.abs(t - np.arange(s1 - s0) / hz).max(initial=0) > 1e-6:
            err.append(f"episode {e}: timestamps not k/{hz} for k=0..{s1 - s0 - 1}")
    # action k is the label computed at state k and executed to reach state k+1: the lag L that
    # minimises median |a_k - s_{k+L}| (same episode) must be 1, clearly below lags 0 and 2
    a, s = D["action"][:, :3].astype(float), D["agent_pos"][:, :3].astype(float)
    epi = np.repeat(np.arange(len(ends)), np.diff(np.r_[0, ends]))
    med = []
    for lag in (0, 1, 2):
        i = np.flatnonzero((np.arange(S) + lag < S) & (epi == epi[np.minimum(np.arange(S) + lag, S - 1)]))
        med.append(np.median(np.linalg.norm(a[i] - s[i + lag], axis=1)))
    r = med[1] / min(med[0], med[2])
    if not r < 0.8:
        err.append(f"state/action alignment: median |a_k - s_(k+L)| for L=0,1,2 = "
                   f"{np.round(1e3 * np.array(med), 2)} mm (L=1 must be < 0.8 x the others)")
    # feature / pose sanity
    pc = D["pc"].astype(float)
    if np.isfinite(pc).all():
        n = np.linalg.norm(pc[..., 3:6], axis=-1)
        dn = np.linalg.norm(pc[..., 6:9], axis=-1)
        ax_ok = all(np.abs(pc[s0:s1, :, 11:14] - np.asarray(ep["axis"])).max() < 1e-5 for s0, s1, ep in zip(starts, ends, eps))
        if (np.abs(n - 1).max() > 1e-3 or not np.all((dn < 1e-6) | (np.abs(dn - 1) < 1e-3))
                or np.any(pc[..., 9] < pc[..., 10]) or np.any(pc[..., 10] < 0) or not ax_ok):
            err.append("pc channel sanity (unit normals / pcurv_dir 0-or-unit / k1>=k2>=0 / axis) failed")
    if np.isfinite(D["agent_pos"]).all() and np.isfinite(D["action"]).all() and not (_rot_ok(D["agent_pos"]) and _rot_ok(D["action"])):
        err.append("rotation columns not orthonormal in agent_pos/action")
    return err, r


def mutations(D):
    def mut(f):
        M = dict(D)
        M["meta"] = copy.deepcopy(D["meta"])
        f(M)
        return M

    def heldout(M):
        g = next(int(k) for k, o in M["meta"]["objects"].items() if o["object_id"] not in TRAIN)
        M["geom_id"] = M["geom_id"].copy()
        M["geom_id"][0] = g
        M["meta"]["episodes"][0].update(object_id=M["meta"]["objects"][str(g)]["object_id"], geom_id=g)

    def tilt(v):
        return lambda M: M["meta"]["episodes"][3].update(tilt_deg=v)

    def put(key, idx, val):
        def f(M):
            M[key] = M[key].copy()
            M[key][idx] = val
        return f

    def shift_actions(n):
        def f(M):
            starts = np.r_[0, M["episode_ends"][:-1]]
            a = M["action"].copy()
            for s0, s1 in zip(starts, M["episode_ends"]):
                a[s0:s1] = np.roll(a[s0:s1], -n, axis=0)  # a_k := a_{k+n}
            M["action"] = a
        return f

    def shift_t(M):
        ep = M["meta"]["episodes"][5]
        ep["t"] = [x + 0.1 for x in ep["t"]]

    def local_path(M):
        M["meta"]["episodes"][0]["note"] = "/Users/someone/data/demos.npz"

    return [("local path in meta", mut(local_path)), ("held-out object", mut(heldout)), ("tilt 15", mut(tilt(15.0))), ("tilt 30", mut(tilt(30.0))),
            ("NaN in pc", mut(put("pc", (7, 3, 2), np.nan))), ("inf in wrench", mut(put("wrench", (11, 2), np.inf))),
            ("NaN in action", mut(put("action", (13, 0), np.nan))), ("actions shifted +1", mut(shift_actions(1))),
            ("actions shifted -1", mut(shift_actions(-1))),
            ("timestamps shifted", mut(shift_t))]


def main(path, per_object=None):
    D = load(path)
    errs, ratio = check(D, per_object)
    m = D["meta"]
    S, E = len(D["action"]), len(D["episode_ends"])
    print(f"{path}: S={S} E={E} steps/episode mean {S / E:.1f}; attempts {m['attempts']['total']}, "
          f"expert success rate {m['attempts']['expert_success_rate']:.3f}; generation {m['generation_seconds']:.0f} s; "
          f"alignment: median|a_k-s_k+1| / min(lag 0, lag 2) = {ratio:.3f} (< 0.8)")
    print(f"per object: { {o: sum(ep['object_id'] == o for ep in m['episodes']) for o in TRAIN} }")
    print(f"tilts: { {t: sum(ep['tilt_deg'] == t for ep in m['episodes']) for t in sorted(TRAIN_TILTS)} }; "
          f"|wrench| max {np.abs(D['wrench'][:, :3]).max():.1f} N")
    F = np.linalg.norm(D["wrench"][:, :3].astype(float), axis=1)
    emax = np.array([F[s0:s1].max() for s0, s1 in zip(np.r_[0, D["episode_ends"][:-1]], D["episode_ends"])])
    print(f"[info] steps with |F|>1 N: {100 * np.mean(F > 1):.1f}%, >5 N: {100 * np.mean(F > 5):.1f}%; per-episode max |F| "
          f"median {np.median(emax):.1f} p90 {np.percentile(emax, 90):.1f} max {emax.max():.1f} N; "
          f"episodes >50 N: {int(np.sum(emax > 50))}; max label depth "
          f"{1e3 * max(float(np.max((D['action'][s0:s1, :3] - np.array(ep['mouth'])) @ np.array(ep['axis']))) for s0, s1, ep in zip(np.r_[0, D['episode_ends'][:-1]], D['episode_ends'], m['episodes'])):.1f} mm (hole depth 30)")
    for e in errs[:20]:
        print("  VIOLATION", e)
    rejected = []
    for name, M in mutations(D):
        e = [x for x in check(M, per_object)[0] if x not in errs]  # only violations the mutation introduced
        rejected.append(bool(e))
        print(f"  self-test [{name}]: {'rejected' if e else 'NOT REJECTED'}" + (f" ({e[0][:90]})" if e else ""))
    ok = not errs and all(rejected)
    print(f"A8: {'PASS' if ok else 'FAIL'}  violations={len(errs)}  mutations_rejected={sum(rejected)}/{len(rejected)}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/demos.npz")
    ap.add_argument("--per-object", type=int, default=None, help="default: protocol demos.per_train_object")
    a = ap.parse_args()
    sys.exit(0 if main(a.path, a.per_object) else 1)
