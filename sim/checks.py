"""Acceptance checks for PLAN v2.1 phase A.  Usage: python -m sim.checks a1|a2|a3|a4|a5"""
import sys
import time
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np

from sim import scene
from sim.env import MAX_DROT, N_POINTS, PegEnv, fps as env_fps, pose_to_vec, rot_exp as scene_rot, rot_log

FIGS = Path(__file__).resolve().parent / "figs"
CASES = scene.PROTOCOL["cases"]


def verdict(name, ok, **nums):
    print(f"{name}: {'PASS' if ok else 'FAIL'}  " + "  ".join(f"{k}={v}" for k, v in nums.items()))
    return ok


def demo_case(object_id, tilt=0.0, yaw=0.0, xy=(0.525, 0.0), seed=0):
    return {"object_id": object_id, "tilt_deg": tilt, "yaw_deg": yaw, "hole_xy": list(xy), "init_seed": seed}


def set_q(env, q):
    env.data.qpos[:7] = q
    env.data.qvel[:] = 0
    env.data.ctrl[:7] = q
    env.q_cmd, env.v_cmd = q.copy(), np.zeros(7)
    mujoco.mj_forward(env.model, env.data)


def robot_geoms(env):
    m = env.model
    names = {m.body(i).name for i in range(m.nbody)}
    ids = {m.body(n).id for n in names if n.startswith("fr3_link") or n == "ft"}
    return set(np.flatnonzero(np.isin(m.geom_bodyid, list(ids))))


def reach(env, case):
    """IK to the pre-insert (50 mm above mouth) and inserted (28 mm deep) aligned poses from the
    reset configuration. Returns dict of residuals / limit / collision flags."""
    env.reset(case)
    _, R0 = env.tip_pose()
    out, q = {}, env.q_cmd
    for tag, depth in (("pre", -0.05), ("ins", 0.028)):
        p, R = env.aligned_pose(depth, R0)
        q, (ep, er) = env.ik(p, R, q)
        at_lim = bool(np.any(np.isclose(q, env.jlo, atol=1e-6) | np.isclose(q, env.jhi, atol=1e-6)))
        set_q(env, q)
        out[tag] = dict(ep_mm=1e3 * ep, er_deg=np.rad2deg(er), at_limit=at_lim,
                        sigma_min=float(np.linalg.svd(env._jac(q), compute_uv=False)[-1]),
                        arm_box=env.contacts(robot_geoms(env), env.box_geoms),
                        peg_box=env.contacts(env.peg_geoms, env.box_geoms))
    ok = all(v["ep_mm"] < 0.1 and v["er_deg"] < 0.1 and not v["at_limit"] and v["arm_box"] == 0
             and v["peg_box"] == 0 for v in out.values())
    return ok, out


def label(img, text):
    img = np.ascontiguousarray(img)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (w + 10, h + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (5, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------------------- A1
def check_a1():
    env = PegEnv()
    FIGS.mkdir(exist_ok=True)
    # (1) all 11 objects load; (2) zero contacts at reset for every protocol case
    loaded = []
    for oid in scene.OBJECTS:
        env.reset(demo_case(oid))
        loaded.append(oid)
    ncon_bad = []
    for c in CASES:
        env.reset(c)
        if env.data.ncon or env.contacts(env.peg_geoms, env.box_geoms):
            ncon_bad.append((c["case_id"], env.data.ncon))
    # (3) clearance geometry: aligned peg at 15 mm depth touches nothing; shifted toward a wall by
    # CLR-0.4 mm still free, by CLR+0.4 mm in contact.
    clr_bad = []
    for oid in scene.OBJECTS:
        env.reset(demo_case(oid, tilt=10, yaw=37))
        _, R0 = env.tip_pose()
        p, R = env.aligned_pose(0.015, R0)
        q, _ = env.ik(p, R, env.q_cmd)
        nrm0 = scene._edges(scene.peg_polygon(env.obj))[0][0]
        wdir = env.R_box @ np.array([nrm0[0], nrm0[1], 0.0])
        res = []
        for shift in (0.0, scene.CLR - 4e-4, scene.CLR + 4e-4):
            qs, _ = env.ik(p + shift * wdir, R, q)
            set_q(env, qs)
            res.append(env.contacts(env.peg_geoms, env.box_geoms))
        if not (res[0] == 0 and res[1] == 0 and res[2] > 0):
            clr_bad.append((oid, res))
    # (4) figures
    env.reset(CASES[0])
    imageio.imwrite(FIGS / "a1_overview.png", label(env.render_rgb(640, 480), f"{CASES[0]['case_id']} "
                                                    f"{CASES[0]['object_id']} tilt{CASES[0]['tilt_deg']}"))
    tiles = []
    for oid in scene.OBJECTS:
        env.reset(demo_case(oid, tilt=0, yaw=0))
        _, R0 = env.tip_pose()
        q, _ = env.ik(*env.aligned_pose(-0.06, R0), env.q_cmd)
        set_q(env, q)
        tiles.append(label(env.render_rgb(240, 240, azimuth=200, elevation=-50, distance=0.32), oid))
    tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, 12, 4)], 0)
    imageio.imwrite(FIGS / "a1_objects.png", grid)
    ex = next(c for c in CASES if c["group"] == "extrap")
    env.reset(ex)
    r = env.renderer(480, 480)
    env._bench_scene(r, shadow=True)
    big = r.render().copy()
    small = cv2.resize(env.rgb(), (480, 480), interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(FIGS / "a1_bench_cam.png", np.concatenate([label(big, f"bench 480 {ex['case_id']}"),
                                                                label(small, "bench 96 (obs rgb)")], 1))
    figs = [FIGS / f for f in ("a1_overview.png", "a1_objects.png", "a1_bench_cam.png")]
    # (5) reachability of every protocol case (reported; does not gate A1)
    unreach = []
    worst = dict(ep_mm=0.0, er_deg=0.0, sigma_min=9.0)
    for c in CASES:
        ok, out = reach(env, c)
        for v in out.values():
            worst["ep_mm"] = max(worst["ep_mm"], v["ep_mm"])
            worst["er_deg"] = max(worst["er_deg"], v["er_deg"])
            worst["sigma_min"] = min(worst["sigma_min"], v["sigma_min"])
        if not ok:
            unreach.append((c["case_id"], {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                               for kk, vv in v.items()} for k, v in out.items()}))
    print(f"reachability: {len(CASES) - len(unreach)}/{len(CASES)} protocol cases reachable "
          f"(pre-insert 50 mm + inserted 28 mm, aligned, no joint limit, no arm-box contact); "
          f"worst IK residual {worst['ep_mm']:.2e} mm / {worst['er_deg']:.2e} deg; "
          f"min sigma_min(J) {worst['sigma_min']:.3f}")
    for u in unreach:
        print("  UNREACHABLE", u)
    print(f"clearance geometry failures: {clr_bad}")
    return verdict("A1", len(loaded) == 11 and not ncon_bad and not clr_bad and all(f.exists() for f in figs),
                   objects_loaded=len(loaded), cases_reset=len(CASES), reset_contacts_nonzero=len(ncon_bad),
                   clearance_fail=len(clr_bad), figs=",".join(f.name for f in figs if f.exists()),
                   unreachable=len(unreach))


# ---------------------------------------------------------------------------------------- A2
class SubstepProbe:
    """Wraps mujoco.mj_step to record joint-limit margin and the peak explicit servo force / limit."""

    def __init__(self):
        self.min_margin, self.max_tau_ratio, self.orig = np.inf, 0.0, mujoco.mj_step

    def __enter__(self):
        def step(m, d):
            self.orig(m, d)
            q = d.qpos[:7]
            self.min_margin = min(self.min_margin, float(np.min(np.minimum(q - m.jnt_range[:7, 0],
                                                                            m.jnt_range[:7, 1] - q))))
            tau = m.actuator_gainprm[:7, 0] * (d.ctrl[:7] - q) + m.actuator_biasprm[:7, 2] * d.qvel[:7]
            self.max_tau_ratio = max(self.max_tau_ratio, float(np.max(np.abs(tau) / m.jnt_actfrcrange[:7, 1])))
        mujoco.mj_step = step
        return self

    def __exit__(self, *a):
        mujoco.mj_step = self.orig


def track_err(env):
    p, R = env.tip_pose()
    pt, Rt = env.last_target
    return 1e3 * np.linalg.norm(p - pt), np.rad2deg(np.linalg.norm(rot_log(Rt @ R.T)))


def a2_targets(env, rng, pre_p, pre_R):
    """Free-space command sequence after the approach: max-size steps from rest (20 mm or 4 deg = the safety clamp,
    hold, return) and a smooth wander (sum of sinusoids, +-25 mm / +-8 deg, 0.3-0.8 Hz)."""
    seq = []
    for j in range(4):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        tgt = (pre_p + 0.02 * u, pre_R) if j < 2 else (pre_p, scene_rot(MAX_DROT * u) @ pre_R)
        seq += [tgt] * 4 + [(pre_p, pre_R)] * 3
    f = rng.uniform(0.3, 0.8, 6)
    ph = rng.uniform(0, 2 * np.pi, 6)
    dirs = rng.normal(size=(6, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for k in range(1, 31):
        t = 0.1 * k
        w = np.sin(2 * np.pi * f * t + ph) - np.sin(ph)  # starts at 0
        dp = 0.025 / 3 * (w[:3, None] * dirs[:3]).sum(0)
        rv = np.deg2rad(8) / 3 * (w[3:, None] * dirs[3:]).sum(0)
        seq.append((pre_p + dp, scene_rot(rv) @ pre_R))
    return seq


def check_a2():
    """Free-space tracking. Per case: approach from the reset pose to the aligned pre-insert pose
    (50 mm above the mouth; the safety clamp makes these full-speed 20 mm / 4 deg steps), then
    a2_targets(). Error = measured tip pose vs the (clamped) commanded target at the end of every
    control step."""
    env = PegEnv()
    rng = np.random.default_rng(0)
    cases = ([c for c in CASES if c["group"] in ("train_control", "extrap")]
             + [c for c in CASES if c["group"] == "f4"][::6])
    errs, contacts, t_step, svmin = {}, 0, [], {}
    with SubstepProbe() as probe:
        for c in cases:
            env.reset(c)
            _, R0 = env.tip_pose()
            pre_p, pre_R = env.aligned_pose(-0.05, R0)
            e = []
            for tgt in [(pre_p, pre_R)] * 15 + a2_targets(env, rng, pre_p, pre_R):
                t0 = time.time()
                env._apply_target_pose(*tgt)
                t_step.append(time.time() - t0)
                e.append(track_err(env))
                contacts += env.data.ncon
            errs[c["case_id"]] = np.array(e)
            svmin[c["case_id"]] = np.linalg.svd(env._jac(env.q_cmd), compute_uv=False)[-1]
    e = np.concatenate(list(errs.values()))
    for grp in ("train_control", "extrap", "f4"):
        g = np.concatenate([v for k, v in errs.items() if k.startswith(grp)])
        print(f"  {grp:13s} pos_err_mm max={g[:, 0].max():.3f} p95={np.percentile(g[:, 0], 95):.3f} | "
              f"rot_err_deg max={g[:, 1].max():.3f} p95={np.percentile(g[:, 1], 95):.3f}")
    bad = {k: (round(v[:, 0].max(), 2), round(v[:, 1].max(), 2), round(svmin[k], 3))
           for k, v in errs.items() if v[:, 0].max() >= 2 or v[:, 1].max() >= 2}
    print(f"cases={len(errs)} steps={len(e)} pos_err_mm max={e[:, 0].max():.3f} mean={e[:, 0].mean():.3f} | "
          f"rot_err_deg max={e[:, 1].max():.3f} mean={e[:, 1].mean():.3f}")
    print(f"cases over 2 mm/2 deg (max mm, max deg, sigma_min at end): {bad}")
    print(f"sigma_min(J) at pre-insert < 0.1: {[(k, round(v, 3)) for k, v in svmin.items() if v < 0.1]}")
    print(f"min joint-limit margin over all substeps = {probe.min_margin:.4f} rad; "
          f"peak explicit servo force/limit = {probe.max_tau_ratio:.2f} (pre-implicit estimate, informational); contacts during free-space run = {contacts}; "
          f"physics+IK per control step {1e3 * np.mean(t_step):.1f} ms")
    ok = e[:, 0].max() < 2.0 and e[:, 1].max() < 2.0 and probe.min_margin > 0 and contacts == 0
    return verdict("A2", ok, max_pos_err_mm=round(e[:, 0].max(), 3), max_rot_err_deg=round(e[:, 1].max(), 3),
                   min_joint_margin_rad=round(probe.min_margin, 4), contacts=contacts)

# ---------------------------------------------------------------------------------------- A3
def check_a3():
    """(1) static: hold the reset pose and the aligned pre-insert pose (peg tilted up to 30 deg);
    gravity-compensated |F| after settling. (2) bottom: scripted aligned insertion through env.step
    to 2 mm past the hole bottom; |F| > 1 N and cos(F, insertion axis in sensor frame) < -0.5.
    (3) timestamps: obs['t'] == k * 0.1 s at every step k, and obs['wrench'] equals the wrench
    recomputed from a fresh mj_forward on a copy of the same state."""
    env = PegEnv()
    static_cases = [c for c in CASES if c["group"] == "train_control"][:4] + [c for c in CASES if c["group"] == "extrap"][:2]
    static_F, static_T = [], []
    for c in static_cases:
        env.reset(c)
        _, R0 = env.tip_pose()
        for tgt in (env.last_target, env.aligned_pose(-0.05, R0)):
            for _ in range(20):
                env._apply_target_pose(*tgt)
            w = env.wrench()
            static_F.append(np.linalg.norm(w[:3]))
            static_T.append(np.linalg.norm(w[3:]))
    bottom, t_bad, w_bad, n_steps = [], 0, 0.0, 0
    ins_cases = [c for c in CASES if c["group"] == "train_control" and c["object_id"] in ("round16c1", "square22c1", "hex22c2")][:3]
    ins_cases += [next(c for c in CASES if c["group"] == "f4" and c["object_id"] == o) for o in ("triangle24c2", "rect16x22c1")]
    for c in ins_cases:
        obs = env.reset(c)
        k = 0
        _, R0 = env.tip_pose()
        plan = [env.aligned_pose(-0.05, R0)] * 12 + [env.aligned_pose(d, R0) for d in np.arange(-0.045, 0.0321, 0.005)]
        plan += [env.aligned_pose(0.032, R0)] * 8  # 2 mm past the bottom (hole depth 30 mm)
        fz_trace = []
        for tgt in plan:
            obs, done, info = env.step(pose_to_vec(*tgt))
            k += 1
            n_steps += 1
            t_bad += abs(obs["t"] - k * 0.1) > 1e-9
            d2 = mujoco.MjData(env.model)
            d2.qpos[:], d2.qvel[:], d2.act[:], d2.ctrl[:] = env.data.qpos, env.data.qvel, env.data.act, env.data.ctrl
            d2.mocap_pos[:], d2.mocap_quat[:], d2.time = env.data.mocap_pos, env.data.mocap_quat, env.data.time
            d2.qacc_warmstart[:] = env.data.qacc_warmstart
            mujoco.mj_forward(env.model, d2)
            live = env.data
            env.data = d2
            w_ref = env.wrench()
            env.data = live
            w_bad = max(w_bad, float(np.abs(w_ref - obs["wrench"]).max()))
            fz_trace.append(obs["wrench"][2])
        R_ft = env.data.site_xmat[env.ft_site].reshape(3, 3)
        F = obs["wrench"][:3].astype(float)
        cos = float(F @ (R_ft.T @ env.axis) / (np.linalg.norm(F) + 1e-12))
        bottom.append((c["case_id"], c["object_id"], c["tilt_deg"], round(float(np.linalg.norm(F)), 2), round(cos, 3),
                       round(info["depth_mm"], 2), info["success"], round(float(min(fz_trace)), 1)))
    # success guard: tip 25 mm below the mouth plane but on the plate beside the boss must not count
    env.reset(ins_cases[0])
    _, R0 = env.tip_pose()
    p_beside = env.data.mocap_pos[env.box_mocap] + env.R_box @ np.array([0.05, 0.05, scene.TB + 0.005])
    q, _ = env.ik(p_beside, env.aligned_rotation(R0), env.q_cmd)
    set_q(env, q)
    depth_beside, inside_beside = env.insertion_depth()
    guard_ok = depth_beside > 0.02 and not inside_beside
    print(f"success guard: tip beside boss at depth {1e3 * depth_beside:.1f} mm -> inside={inside_beside} (must be False)")
    print(f"static: {len(static_F)} holds, max |F| = {max(static_F):.4f} N, max |T| = {max(static_T):.5f} Nm")
    print("bottom contact (case, object, tilt, |F| N, cos(F,axis), depth_mm, success, min Fz over run):")
    for b in bottom:
        print("   ", b)
    print(f"timestamps: {n_steps} steps, t != k*0.1 in {t_bad}; max |obs wrench - fresh mj_forward wrench| = {w_bad:.2e}")
    ok = (max(static_F) < 0.1 and all(b[3] > 1 and b[4] < -0.5 for b in bottom) and t_bad == 0 and w_bad < 1e-4
          and guard_ok and all(b[6] for b in bottom))  # 1e-4 N: contact-solver tolerance
    return verdict("A3", ok, static_max_F=round(max(static_F), 4), bottom_min_F=min(b[3] for b in bottom),
                   bottom_max_cos=max(b[4] for b in bottom), t_mismatch=t_bad, wrench_mismatch=f"{w_bad:.1e}")


# ---------------------------------------------------------------------------------------- A4
def a4_frames(env):
    """Frames at reset, pre-insert and 10 mm inserted for a mix of groups/objects."""
    picks = [c for c in CASES if c["group"] == "train_control"][:3] + [c for c in CASES if c["group"] == "f3"][:2]
    picks += [c for c in CASES if c["group"] == "f4"][::10] + [c for c in CASES if c["group"] == "extrap"][:1]
    for c in picks:
        env.reset(c)
        yield c, "reset"
        _, R0 = env.tip_pose()
        for tag, depth, n in (("pre", -0.05, 12), ("in10mm", 0.010, 10)):
            p, R = env.aligned_pose(depth, R0)
            if tag == "in10mm":
                for dd in np.arange(-0.045, depth, 0.005):
                    env._apply_target_pose(*env.aligned_pose(dd, R0))
            for _ in range(n):
                env._apply_target_pose(p, R)
            yield c, tag


def box_vertices(env):
    """World coordinates of every vertex of the hole box (plate corners + boss outline, bottom and
    top). Ground truth -- used by this check only."""
    nrm, apo = scene._edges(scene.peg_polygon(env.obj))
    a_out = apo + scene.CLR + scene.TW
    V = [scene._line_isect(nrm[i - 1], a_out[i - 1], nrm[i], a_out[i]) for i in range(len(apo))]
    pts = [[x, y, z] for x, y in V for z in (scene.TB, scene.TB + scene.H)]
    h = scene.PLATE_HALF
    pts += [[sx * h, sy * h, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, scene.TB)]
    return env.data.mocap_pos[env.box_mocap] + np.array(pts) @ env.R_box.T


def old_box_crop_pixels(env, depth):
    """The pre-C3 crop (box-frame window from the TRUE box pose) -- for the comparison only."""
    m = env.model
    z = depth.reshape(-1)
    cp, _ = env.cam_pose()
    valid = (z > 0) & (z < 0.99 * m.vis.map.zfar * m.stat.extent) & (cp[2] + z * env._ray_w[:, 2] > 0.003)
    pix = np.flatnonzero(valid)
    P = cp + env._ray_w[pix] * z[pix, None]
    Pb = (P - env.data.mocap_pos[env.box_mocap]) @ env.R_box
    half = scene.PLATE_HALF + 0.010
    keep = (np.abs(Pb[:, 0]) <= half) & (np.abs(Pb[:, 1]) <= half) & (Pb[:, 2] >= 0.0005) & (Pb[:, 2] <= scene.TB + scene.H + 0.15)
    return P[keep], pix[keep]


OBS_METHODS = ("observe", "point_cloud", "depth_cloud", "rgb", "_bench_scene", "wrench", "tip_pose", "cam_pose")
TRUTH_TOKENS = ("box_mocap", "R_box", "mouth", "mocap_pos", "stand_mocap", "self.case", "aligned_")


def privileged_info_test(env):
    """(a) source scan of the observation path; (b) poison every stored ground-truth hole quantity
    except the nominal axis and require a bit-identical observation."""
    import inspect

    from sim import features
    hits = [(f, t) for f in OBS_METHODS for t in TRUTH_TOKENS if t in inspect.getsource(getattr(PegEnv, f))]
    hits += [("features", t) for t in TRUTH_TOKENS if t in inspect.getsource(features)]
    axis_users = [f for f in OBS_METHODS if "self.axis" in inspect.getsource(getattr(PegEnv, f))]
    o1 = env.observe()
    saved = (env.R_box, env.mouth, env.box_mocap, env.stand_mocap, env.case)
    env.R_box, env.mouth = np.full((3, 3), np.nan), np.full(3, np.nan)
    env.box_mocap = env.stand_mocap = 10**6  # any use raises IndexError
    env.case = None
    try:
        o2 = env.observe()
        same = all(np.array_equal(o1[k], o2[k]) for k in ("pc", "rgb", "agent_pos", "wrench")) and o1["t"] == o2["t"]
    except Exception as e:  # noqa: BLE001
        same = f"raised {type(e).__name__}: {e}"
    finally:
        env.R_box, env.mouth, env.box_mocap, env.stand_mocap, env.case = saved
    return hits, axis_users, same


def check_a4():
    from sim.env import CROP_R, CROP_Z
    from sim.expert import random_case
    env = PegEnv()
    gg = np.array([1, 1, 1, 0, 0, 0], np.uint8)  # geom groups the renderer shows
    geomid = np.zeros(1, np.int32)
    med_errs, max_errs, n_bg, n_pts_bad, times, t_pc, rgb_bad, frames = [], [], 0, 0, [], [], 0, 0
    frac = {"new": {"box": [], "stand": [], "peg": [], "robot": []}, "old": {"box": [], "stand": [], "peg": [], "robot": []}}
    seg_r = None
    for c, tag in a4_frames(env):
        m, d = env.model, env.data
        if seg_r is None or seg_r.model is not m:
            seg_r = mujoco.Renderer(m, 480, 480)
            seg_r.enable_segmentation_rendering()
        t0 = time.time()
        P, pix = env.depth_cloud()
        sel = env_fps(P, N_POINTS)
        rgb = env.rgb()
        times.append(time.time() - t0)
        t0 = time.time()
        pc, pix_sel = env.point_cloud()
        t_pc.append(time.time() - t0)
        frames += 1
        assert np.array_equal(pix[sel], pix_sel)
        xyz = pc[:, :3].astype(float)
        n_pts_bad += len(pc) != N_POINTS
        rgb_bad += not (rgb.shape == (96, 96, 3) and rgb.dtype == np.uint8)
        cp, _ = env.cam_pose()
        errs = []
        for q, pxl in zip(xyz, pix_sel):
            v = env._ray_w[pxl] / np.linalg.norm(env._ray_w[pxl])
            dist = mujoco.mj_ray(m, d, cp, v, gg, 1, -1, geomid)
            errs.append(np.inf if dist < 0 else np.linalg.norm(cp + dist * v - q))
        errs = np.array(errs)
        med_errs.append(np.median(errs))
        max_errs.append(np.max(errs))
        env._bench_scene(seg_r, shadow=False)
        seg = seg_r.render().reshape(-1, 2)
        robot = {m.body(i).id for i in range(m.nbody) if m.body(i).name.startswith("fr3_link") or m.body(i).name == "ft"}
        dr = env.renderer(480, 480, kind="depth")
        env._bench_scene(dr, shadow=False)
        Pold, pix_old = old_box_crop_pixels(env, dr.render())
        for key, px in (("new", pix_sel), ("old", pix_old[env_fps(Pold, N_POINTS)])):
            g = np.where(seg[px, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[px, 0], -1)
            body = np.where(g >= 0, m.geom_bodyid[np.maximum(g, 0)], -1)
            if key == "new":
                n_bg += int(np.sum((g < 0) | (body == 0)))  # sky / floor (world body)
            frac[key]["box"].append(np.mean(body == m.body("box").id))
            frac[key]["stand"].append(np.mean(body == m.body("stand").id))
            frac[key]["peg"].append(np.mean(body == m.body("peg").id))
            frac[key]["robot"].append(np.mean(np.isin(body, list(robot))))
    env_obs = env.observe()
    same_step_rgb = np.array_equal(env_obs["rgb"], env.rgb()) and env_obs["t"] == float(env.data.time)
    # box fully inside the crop: all 70 protocol cases + 300 random tilt<=30 deg cases, tip at reset /
    # pre-insert (60 mm) / 20 mm inserted
    env.observe = lambda: None
    rng = np.random.default_rng(3)
    cases = list(CASES) + [random_case(rng, list(scene.OBJECTS)[i % 11], tilts=(float(rng.uniform(0, 30)),))
                           for i in range(300)]
    worst_r, zlo, zhi, outside = 0.0, 9.0, -9.0, 0
    for c in cases:
        env.reset(c)
        B = box_vertices(env)
        zlo, zhi = min(zlo, B[:, 2].min()), max(zhi, B[:, 2].max())
        for tip in (env.tip_pose()[0], env.mouth - 0.06 * env.axis, env.mouth + 0.02 * env.axis):
            r = np.linalg.norm(B[:, :2] - tip[:2], axis=1).max()
            worst_r = max(worst_r, r)
            outside += int(r > CROP_R or B[:, 2].min() < CROP_Z[0] or B[:, 2].max() > CROP_Z[1])
    del env.observe
    hits, axis_users, same = privileged_info_test(env)
    med_errs, max_errs = np.array(med_errs), np.array(max_errs)
    print(f"crop: cylinder r={CROP_R} m around the current tip (proprioception) x world z in {CROP_Z} m")
    print(f"box inside crop: {len(cases)} cases x 3 tip poses, outside={outside}; worst tip->box-vertex horizontal "
          f"{1e3 * worst_r:.1f} mm (R {1e3 * CROP_R:.0f}); box z range {zlo:.4f}..{zhi:.4f} m")
    print(f"frames={frames}: per-frame median |p - mj_ray| mm: max={1e3 * med_errs.max():.4f} "
          f"mean={1e3 * med_errs.mean():.4f}; worst single point {1e3 * max_errs.max():.3f} mm")
    for key in ("new", "old"):
        f = {k: 100 * np.mean(v) for k, v in frac[key].items()}
        print(f"  {key} crop, share of the {N_POINTS} points: box {f['box']:.1f}% (min frame {100 * min(frac[key]['box']):.1f}%), "
              f"peg {f['peg']:.1f}%, robot {f['robot']:.1f}%, stand {f['stand']:.2f}% (max frame {100 * max(frac[key]['stand']):.2f}%)")
    print(f"background (sky/table) points: {n_bg}; frames with != {N_POINTS} points: {n_pts_bad}; bad rgb: {rgb_bad}; "
          f"rgb from same step as pc: {same_step_rgb}")
    print(f"time per frame (depth render + back-project + crop + FPS + 96x96 RGB): mean {1e3 * np.mean(times):.1f} "
          f"max {1e3 * np.max(times):.1f} ms; with 14-ch features (point_cloud) mean {1e3 * np.mean(t_pc):.1f} ms")
    print(f"privileged info: truth tokens in observation path {hits or 'none'}; self.axis used by {axis_users} "
          f"(nominal insertion axis = task input, allowed); poisoned-truth observation identical: {same}")
    ok = (med_errs.max() < 1e-3 and n_bg == 0 and n_pts_bad == 0 and rgb_bad == 0 and same_step_rgb
          and np.max(times) < 0.05 and outside == 0 and not hits and same is True)
    return verdict("A4", ok, frames=frames, max_frame_median_err_mm=round(1e3 * med_errs.max(), 4), bg_points=n_bg,
                   stand_pct=round(100 * np.mean(frac["new"]["stand"]), 2), box_pct_new_old=(
                       round(100 * np.mean(frac["new"]["box"]), 1), round(100 * np.mean(frac["old"]["box"]), 1)),
                   max_ms=round(1e3 * np.max(times), 1), box_outside_crop=outside, truth_free=same is True and not hits)


# ---------------------------------------------------------------------------------------- A5
def _frame(z):
    """Orthonormal (x, y, z) columns with the given z."""
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.3, 0.5, 0.8])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def synth_view(rng, kind, r=0.01, noise=1e-5, dist=0.7, f_px=None, half_px=45):
    """Synthetic depth-camera observation of an analytic plane or cylinder: pinhole camera
    (bench-camera focal length) at `dist`, pixel-centre rays intersected analytically, depth
    noise along the ray. Returns (points, truth direction: plane normal / cylinder axis, cam)."""
    f_px = f_px or 480 / 2 / np.tan(np.deg2rad(scene.CAM_FOVY) / 2)
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)  # plane normal or cylinder axis
    while True:  # view direction: plane seen within 60 deg of its normal, cylinder from the side
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if (kind == "plane" and -v @ u > 0.5) or (kind == "cylinder" and abs(v @ u) < 0.5):
            break
    c = np.array([0.5, 0.0, 0.25])
    o = c - dist * v
    B = _frame(v)
    jj, ii = np.meshgrid(np.arange(-half_px, half_px + 1), np.arange(-half_px, half_px + 1))
    d = v + (jj.reshape(-1, 1) * B[:, 0] + ii.reshape(-1, 1) * B[:, 1]) / f_px
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    if kind == "plane":
        t = ((c - o) @ u) / (d @ u)
        keep = np.ones(len(t), bool)
    else:
        oc = o - c
        dp, op = d - (d @ u)[:, None] * u, oc - (oc @ u) * u
        A, Bq, C = (dp * dp).sum(1), 2 * dp @ op, op @ op - r * r
        disc = Bq * Bq - 4 * A * C
        keep = disc > 0
        t = np.where(keep, (-Bq - np.sqrt(np.maximum(disc, 0))) / (2 * A), 0)
    t = t + rng.normal(0, noise, len(t))
    P = o + t[:, None] * d
    if kind == "cylinder":
        keep &= np.abs((P - c) @ u) < 0.02  # 40 mm long
    return P[keep], u, o


def flat_face_mask(env, xyz, margin):
    """Points on large planar faces of the box, >= margin from any edge (ground-truth geometry)."""
    pivot = env.data.mocap_pos[env.box_mocap]
    Pb = (xyz - pivot) @ env.R_box
    nrm, apo = scene._edges(scene.peg_polygon(env.obj))
    a_out = apo + scene.CLR + scene.TW
    r_out = np.max(a_out / np.cos(np.pi / len(apo)))  # circumradius bound of the boss
    radial = np.linalg.norm(Pb[:, :2], axis=1)
    d_in = np.max(Pb[:, :2] @ nrm.T - (apo + scene.CLR + env.obj["chamfer_mm"] / 1000.0), axis=1)  # >0: outside chamfered opening
    d_out = np.min(a_out - Pb[:, :2] @ nrm.T, axis=1)  # >0: inside boss outline
    boss_top = (np.abs(Pb[:, 2] - (scene.TB + scene.H)) < 5e-4) & (d_in > margin) & (d_out > margin)
    plate_top = ((np.abs(Pb[:, 2] - scene.TB) < 5e-4) & (radial > r_out + margin)
                 & (np.max(np.abs(Pb[:, :2]), axis=1) < scene.PLATE_HALF - margin))
    return boss_top | plate_top


def check_a5():
    from scipy.spatial import cKDTree

    from sim.features import KAPPA_FLAT, KNN, SIGN_EPS, compute_features
    rng = np.random.default_rng(0)
    res = {}
    # ---- synthetic plane
    errs, flat_ok = [], []
    for _ in range(5):
        P, n0, cam = synth_view(rng, "plane")
        f = compute_features(P, cam, n0, query_idx=env_fps(P, N_POINTS))
        errs.append(np.rad2deg(np.arccos(np.clip(f[:, 3:6] @ n0, -1, 1))).max())
        flat_ok.append(np.all(f[:, 6:9] == 0))
    res["plane_max_normal_err_deg"] = round(float(max(errs)), 4)
    res["plane_all_flat_zeroed"] = all(flat_ok)
    plane_ok = max(errs) < 2.0 and all(flat_ok)
    # ---- synthetic cylinders (convex, radii of the round pegs)
    cyl_ok = True
    for r in (0.008, 0.011):
        dcos, k1, k2 = [], [], []
        for _ in range(3):
            P, a0, cam = synth_view(rng, "cylinder", r=r)
            f = compute_features(P, cam, a0, query_idx=env_fps(P, N_POINTS))
            dirn = np.linalg.norm(f[:, 6:9], axis=1)
            dcos.append(np.where(dirn > 0, np.abs(f[:, 6:9] @ a0) / np.maximum(dirn, 1e-12), 1.0))
            k1.append(f[:, 9])
            k2.append(f[:, 10])
        dcos, k1, k2 = map(np.concatenate, (dcos, k1, k2))
        frac_dir = float(np.mean(dcos < 0.1))
        frac_k1 = float(np.mean(np.abs(k1 * r - 1) <= 0.2))
        frac_k2 = float(np.mean(np.abs(k2) < 0.2 / r))
        res[f"cyl_r{int(r * 1e3)}mm"] = dict(frac_dir_perp=round(frac_dir, 3), median_k1_r=round(float(np.median(k1) * r), 3),
                                              frac_k1_within20=round(frac_k1, 3), median_k2_r=round(float(np.median(k2) * r), 3),
                                              frac_k2_small=round(frac_k2, 3))
        cyl_ok &= frac_dir >= 0.9 and frac_k1 >= 0.9 and frac_k2 >= 0.9
    # ---- real observations
    env = PegEnv()
    n_pts, facing, sign_bad, flat_bad, zero_bad, flat_face_n, flat_face_zero, axis_bad, order_bad = 0, 0, 0, 0, 0, 0, 0, 0, 0
    knn_r = []
    for c, tag in a4_frames(env):
        f, _ = env.point_cloud()
        f = f.astype(float)
        dense, _ = env.depth_cloud()
        knn_r.append(np.median(cKDTree(dense).query(dense, k=KNN)[0][:, -1]))
        xyz, nrm, d, k = f[:, :3], f[:, 3:6], f[:, 6:9], f[:, 9:11]
        cp, _ = env.cam_pose()
        n_pts += len(f)
        facing += int(np.sum(np.einsum("ij,ij->i", nrm, cp - xyz) > 0))
        nz = np.linalg.norm(d, axis=1) > 0
        sa, sx, sy = d @ env.axis, d[:, 0], d[:, 1]
        good = (sa > SIGN_EPS) | ((np.abs(sa) <= SIGN_EPS) & (sx > SIGN_EPS)) | (
            (np.abs(sa) <= SIGN_EPS) & (np.abs(sx) <= SIGN_EPS) & (sy >= -SIGN_EPS))
        sign_bad += int(np.sum(nz & ~good))
        zero_bad += int(np.sum((k[:, 0] < KAPPA_FLAT) & nz))  # flat but direction not zeroed
        flat_bad += int(np.sum((k[:, 0] >= KAPPA_FLAT) & ~nz))  # curved but zeroed
        ff = flat_face_mask(env, xyz, margin=2 * knn_r[-1])  # flat = whole 2-ring KNN footprint on one face
        flat_face_n += int(ff.sum())
        flat_face_zero += int(np.sum(ff & ~nz))
        axis_bad += int(np.sum(np.abs(f[:, 11:14] - env.axis).max(1) > 1e-6))
        order_bad += int(np.sum(k[:, 0] < k[:, 1]) + np.sum(k < 0))
    res["real_points"] = n_pts
    res["real_knn20_radius_mm_median"] = round(1e3 * float(np.median(knn_r)), 2)
    res["real_normals_facing_cam"] = round(facing / n_pts, 4)
    res["real_sign_violations"] = sign_bad
    res["real_flat_not_zeroed"] = zero_bad
    res["real_curved_but_zeroed"] = flat_bad
    res["real_flat_face_points"] = flat_face_n
    res["real_flat_face_zeroed_frac"] = round(flat_face_zero / max(flat_face_n, 1), 4)
    res["real_axis_or_order_bad"] = axis_bad + order_bad
    for k_, v in res.items():
        print(f"  {k_}: {v}")
    real_ok = (facing / n_pts >= 0.95 and sign_bad == 0 and zero_bad == 0 and flat_bad == 0 and flat_face_n > 0
               and flat_face_zero / flat_face_n >= 0.95 and axis_bad + order_bad == 0)
    return verdict("A5", plane_ok and cyl_ok and real_ok, plane=plane_ok, cylinder=cyl_ok, real=real_ok)


# ---------------------------------------------------------------------------------------- A6 / A7
def lite(env):
    """The scripted expert reads privileged state only, so A6/A7 skip the (physics-free) point-cloud
    rendering: observe() returns just the wrench. Dynamics are unchanged."""
    env.observe = lambda: {"wrench": env.wrench()}
    return env


def train_objects():
    return [o["object_id"] for o in scene.PROTOCOL["objects"]["train"]]


def check_a6():
    """Force-reactive expert without DART noise (lateral aiming error U(0.5,1.8) mm, corrected from
    the F/T only). (1) 30 fresh cases: train objects x train tilts (seeded, init_seed disjoint from
    the protocol). (2) the 30 protocol f4 cases (each held-out object x 10) -> proves the f4 cases
    are insertable. f3 / extrap protocol cases: informational."""
    from sim.expert import Expert, random_case, run_episode
    env = lite(PegEnv())
    ex = Expert(env)
    rng = np.random.default_rng(6)
    objs = train_objects()
    train = [run_episode(env, random_case(rng, objs[i % len(objs)]), ex) for i in range(30)]
    held = {}
    for c in (c for c in CASES if c["group"] == "f4"):
        held.setdefault(c["object_id"], []).append(run_episode(env, c, ex))
    info = {g: [run_episode(env, c, ex) for c in CASES if c["group"] == g] for g in ("f3", "extrap")}
    allr = train + [r for v in held.values() for r in v] + [r for v in info.values() for r in v]
    n_tr = sum(r["success"] for r in train)
    print(f"train objects x train tilts: {n_tr}/30")
    for o, v in held.items():
        print(f"held-out {o}: {sum(r['success'] for r in v)}/{len(v)} (protocol f4 cases)")
    for g, v in info.items():
        print(f"[info] protocol {g}: {sum(r['success'] for r in v)}/{len(v)}")
    fm = [r["f_max"] for r in allr]
    print(f"steps mean {np.mean([r['steps'] for r in allr]):.1f} max {max(r['steps'] for r in allr)}; "
          f"per-episode max |F| median {np.median(fm):.2f} N, p90 {np.percentile(fm, 90):.2f} N, max {max(fm):.2f} N")
    req = train + [r for v in held.values() for r in v]  # the 60 required episodes
    corr = [r for r in req if r["corr_steps"] > 0]
    print(f"aiming error |e| over the 60 required episodes: median {np.median([r['aim_mm'] for r in req]):.2f} mm, "
          f"range {min(r['aim_mm'] for r in req):.2f}-{max(r['aim_mm'] for r in req):.2f} mm")
    print(f"episodes with force-based lateral correction: {len(corr)}/{len(req)} ({100 * len(corr) / len(req):.0f}%); "
          f"spiral search used in {sum(r['spiral_steps'] > 0 for r in req)}")
    if corr:
        print(f"  in those: belief error |true centre - estimate| median {np.median([r['aim_mm'] for r in corr]):.2f} -> "
              f"{np.median([r['belief_err_mm'] for r in corr]):.2f} mm (max after {max(r['belief_err_mm'] for r in corr):.2f}); "
              f"true tip lateral offset at first contact median {np.median([r['tip_lat_first_contact_mm'] for r in corr]):.2f} "
              f"-> at the end {np.median([r['tip_lat_end_mm'] for r in corr]):.2f} mm")
    ok = n_tr >= 29 and all(sum(r["success"] for r in v) >= 9 for v in held.values()) and len(held) == 3
    return verdict("A6", ok, train=f"{n_tr}/30", **{o: f"{sum(r['success'] for r in v)}/{len(v)}" for o, v in held.items()})


def check_a7():
    """DART expert on 30 fresh train cases. Noise: position N(0, 2 mm) per axis, rotation vector
    isotropic with RMS angle 2 deg, full sigma while the label tip is > 10 mm outside the mouth,
    sigma/4 inside. Measured on the full-sigma steps: executed (after the env safety clamp) minus
    clean label: per-axis position std and RMS rotation angle must be within sigma +-25%."""
    from sim.expert import Expert, random_case, run_episode
    env = lite(PegEnv())
    sp, sr = 0.002, np.deg2rad(2.0)
    ex = Expert(env, sp, sr, seed=7)
    rng = np.random.default_rng(7)
    objs = train_objects()
    res = [run_episode(env, random_case(rng, objs[i % len(objs)]), ex) for i in range(30)]
    d = [x for r in res for x, n in zip(r["diffs"], r["noisy"]) if n]
    d_in = [x for r in res for x, n in zip(r["diffs"], r["noisy"]) if not n]
    dp, dr = np.array([x[0] for x in d]), np.array([x[1] for x in d])
    std_p = dp.std(0)
    rms_r = float(np.sqrt((np.linalg.norm(dr, axis=1) ** 2).mean()))
    n_ok = sum(r["success"] for r in res)
    fm = [r["f_max"] for r in res]
    print(f"success {n_ok}/30; full-sigma steps {len(d)} of {len(d) + len(d_in)}")
    print(f"executed - label (full-sigma steps): position std per axis mm {np.round(1e3 * std_p, 3)} "
          f"(target 2 +-0.5), rotation RMS angle {np.rad2deg(rms_r):.3f} deg (target 2 +-0.5)")
    if d_in:
        dpi = np.array([x[0] for x in d_in])
        print(f"[info] in-gate steps position std per axis mm {np.round(1e3 * dpi.std(0), 3)} (sigma/4 = 0.5)")
    print(f"per-episode max |F| median {np.median(fm):.2f} N, p90 {np.percentile(fm, 90):.2f} N, max {max(fm):.2f} N")
    ok = (n_ok >= 24 and np.all(np.abs(std_p - sp) <= 0.25 * sp) and abs(rms_r - sr) <= 0.25 * sr)
    return verdict("A7", ok, success=f"{n_ok}/30", pos_std_mm=np.round(1e3 * std_p, 2).tolist(),
                   rot_rms_deg=round(np.rad2deg(rms_r), 3), max_F=round(max(fm), 1))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "a1"
    t0 = time.time()
    ok = globals()[f"check_{which}"]()
    print(f"[{which}] {time.time() - t0:.1f}s")
    sys.exit(0 if ok else 1)
