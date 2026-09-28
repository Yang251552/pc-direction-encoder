"""Peg-in-hole env on FR3.  API: reset(case) / step(action9) / observe() / render_rgb(w, h).

action9 / agent_pos = peg-tip position (3) + first two columns of the peg-tip rotation matrix
(col0 then col1), world frame.  Control 10 Hz, physics dt 2 ms.
"""
import mujoco
import numpy as np

from sim import scene
from sim.features import compute_features

CONTROL_DT = 1.0 / scene.PROTOCOL["success"]["control_hz"]
SUBSTEPS = int(round(CONTROL_DT / scene.DT))
MAX_STEPS = scene.PROTOCOL["success"]["max_steps"]
MIN_DEPTH = scene.PROTOCOL["success"]["min_depth_mm"] / 1000.0
MAX_DPOS = 0.02  # safety clamp per control step, relative to the measured tip pose
MAX_DROT = np.deg2rad(4.0)  # 6 deg/step exceeded the FR3 wrist torque near its wrist singularity (extrap-069: 2.1 mm/3.4 deg)
BLEND_T = 0.03  # s, joint-velocity blend at the start of each control period
Q_HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
JNT_MARGIN = 0.03  # rad, IK keeps this far from joint limits
W_LIMIT = 0.1  # joint-limit barrier weight in the redundancy objective
NULL_STEP = 0.02  # rad per control step of null-space posture optimisation
RESET_NULL_ITERS = 60
QDOT_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])  # FR3 datasheet joint velocity limits
DEPTH_RES = 480
RGB_RES = 96
N_POINTS = 512
# Point-cloud crop, non-privileged: vertical cylinder of radius CROP_R around the *current tip*
# (proprioception) intersected with a fixed world z band (workspace constant). Sized from the box
# geometry over all 70 protocol cases + 300 random tilt<=30 deg cases at the reset, pre-insert and
# inserted tip poses: max tip->box-vertex horizontal distance 127.4 mm, box z in [0.188, 0.270] m.
# The band's lower edge removes the table and most of the stand; the upper edge keeps the lower
# ~2-8 cm of the peg and drops the flange / arm.
CROP_R = 0.14
CROP_Z = (0.183, 0.33)


def rot_log(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).flatten())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def rot_exp(v):
    q = np.zeros(4)
    th = np.linalg.norm(v)
    mujoco.mju_axisAngle2Quat(q, v / th if th > 0 else np.array([1.0, 0, 0]), th)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def pose_to_vec(p, R):
    return np.concatenate([p, R[:, 0], R[:, 1]])


def vec_to_pose(a):
    a = np.asarray(a, np.float64)
    x = a[3:6] / np.linalg.norm(a[3:6])
    y = a[6:9] - (x @ a[6:9]) * x
    y /= np.linalg.norm(y)
    return a[:3].copy(), np.stack([x, y, np.cross(x, y)], 1)


def fps(P, k, start=0):
    """Farthest point sampling (numpy, float32 distances). Deterministic given P."""
    x, y, z = (np.ascontiguousarray(P[:, i], np.float32) for i in range(3))
    idx = np.empty(k, np.int64)
    idx[0] = start
    d = np.full(len(P), np.inf, np.float32)
    t, dd = np.empty(len(P), np.float32), np.empty(len(P), np.float32)
    for i in range(1, k):
        j = idx[i - 1]
        np.subtract(x, x[j], out=t)
        np.multiply(t, t, out=dd)
        np.subtract(y, y[j], out=t)
        t *= t
        dd += t
        np.subtract(z, z[j], out=t)
        t *= t
        dd += t
        np.minimum(d, dd, out=d)
        idx[i] = int(d.argmax())
    return idx


class PegEnv:
    def __init__(self):
        self.object_id = None
        self._renderers = {}

    # ------------------------------------------------------------------ setup
    def _load(self, object_id):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}
        m = self.model = scene.build_model(object_id)
        self.data = mujoco.MjData(m)
        self.ik_data = mujoco.MjData(m)
        self.object_id, self.obj = object_id, scene.OBJECTS[object_id]
        self.tip = m.site("peg_tip").id
        self.ft_site = m.site("ft_site").id
        self.ft_body = m.body("ft").id
        self.cam = m.camera("bench").id
        self.box_mocap = m.body_mocapid[m.body("box").id]
        self.stand_mocap = m.body_mocapid[m.body("stand").id]
        self.jlo = m.jnt_range[:7, 0] + JNT_MARGIN
        self.jhi = m.jnt_range[:7, 1] - JNT_MARGIN
        self.kp = m.actuator_gainprm[:7, 0].copy()
        self.kv_over_kp = -m.actuator_biasprm[:7, 2] / self.kp
        self._M = np.zeros((m.nv, m.nv))
        a = m.sensor_adr
        self._f_adr, self._t_adr = a[m.sensor("ft_force").id], a[m.sensor("ft_torque").id]
        peg_body = m.body("peg").id
        self.peg_geoms = set(np.flatnonzero(m.geom_bodyid == peg_body))
        self.box_geoms = set(np.flatnonzero(m.geom_bodyid == m.body("box").id))
        # camera intrinsics (square image, fovy vertical)
        self.f_px = DEPTH_RES / 2 / np.tan(np.deg2rad(m.cam_fovy[self.cam]) / 2)
        jj, ii = np.meshgrid(np.arange(DEPTH_RES), np.arange(DEPTH_RES))
        self._ray_cam = np.stack([(jj + 0.5 - DEPTH_RES / 2) / self.f_px, -(ii + 0.5 - DEPTH_RES / 2) / self.f_px,
                                  -np.ones_like(jj, dtype=float)], -1).reshape(-1, 3)
        self.hole_r_max = np.linalg.norm(scene.peg_polygon(self.obj), axis=1).max() + 2 * scene.CLR

    def renderer(self, w, h, kind="bench"):
        """Cached mujoco.Renderer per (size, kind); kind in {"bench", "depth", "free"}."""
        key = (w, h, kind)
        if key not in self._renderers:
            r = mujoco.Renderer(self.model, height=h, width=w)
            if kind == "depth":
                r.enable_depth_rendering()
            elif kind == "bench":  # the first shadowed frame of a new GL context differs by <=2/255: warm up
                self._bench_scene(r, shadow=True)
                r.render()
            self._renderers[key] = r
        return self._renderers[key]

    def reset(self, case):
        if case["object_id"] != self.object_id:
            self._load(case["object_id"])
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        pivot, R, mouth = scene.box_pose(case)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        d.mocap_pos[self.box_mocap], d.mocap_quat[self.box_mocap] = pivot, q
        d.mocap_pos[self.stand_mocap], d.mocap_quat[self.stand_mocap] = pivot, [1, 0, 0, 0]
        self.ik_data.mocap_pos[:], self.ik_data.mocap_quat[:] = d.mocap_pos, d.mocap_quat
        self.case, self.R_box, self.mouth = case, R, mouth
        self.axis = -R[:, 2]  # insertion direction
        # random initial tip pose above the hole, peg roughly pointing down
        rng = np.random.default_rng(case["init_seed"])
        p0 = mouth + np.array([*rng.uniform(-0.03, 0.03, 2), rng.uniform(0.10, 0.14)])
        psi = np.deg2rad(45 + rng.uniform(-30, 30))
        R0 = np.array([[np.cos(psi), np.sin(psi), 0], [np.sin(psi), -np.cos(psi), 0], [0, 0, -1.0]])
        ax = rng.normal(size=3)
        ax[2] = 0
        R0 = rot_exp(ax / np.linalg.norm(ax) * np.deg2rad(rng.uniform(0, 8))) @ R0
        q0, err = self.ik(p0, R0, Q_HOME)
        for _ in range(RESET_NULL_ITERS):  # start from a well-conditioned redundancy posture
            q0, err = self.ik(p0, R0, q0, null_step=0.05)
        assert err[0] < 1e-4 and err[1] < 1e-3, f"init IK failed {err}"
        d.qpos[:7] = q0
        d.ctrl[:7] = q0
        self.q_cmd, self.v_cmd = q0.copy(), np.zeros(7)
        self.k = 0
        self.last_target = (p0, R0)
        mujoco.mj_forward(m, d)
        self._ray_w = self._ray_cam @ self.cam_pose()[1].T  # fixed camera: pixel rays in world frame
        return self.observe()

    # ------------------------------------------------------------------ kinematics
    def tip_pose(self, d=None):
        d = self.data if d is None else d
        return d.site_xpos[self.tip].copy(), d.site_xmat[self.tip].reshape(3, 3).copy()

    def _jac(self, q):
        m, d = self.model, self.ik_data
        d.qpos[:7] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.tip)
        return np.vstack([jp, jr])[:, :7]

    def posture_score(self, q):
        """Redundancy objective (higher = better): log manipulability of the tip Jacobian plus a
        log barrier on the joint limits."""
        J = self._jac(q)
        lo, hi = self.model.jnt_range[:7].T
        return np.linalg.slogdet(J @ J.T)[1] + W_LIMIT * np.sum(np.log((q - lo) * (hi - q)))

    def null_step(self, q, max_step):
        """One bounded gradient-ascent step on posture_score inside the Jacobian null space."""
        f0, h = self.posture_score(q), 1e-5
        g = np.array([(self.posture_score(q + h * e) - f0) / h for e in np.eye(7)])
        J = self._jac(q)
        dq = (np.eye(7) - np.linalg.pinv(J) @ J) @ g
        return q + dq * (max_step / max(np.abs(dq).max(), 1.0))

    def ik(self, p, R, q_seed, iters=200, lam=0.02, null_step=0.0):
        """Optional bounded null-space posture step, then damped least squares on the peg-tip site
        iterated to convergence."""
        m, d = self.model, self.ik_data
        q = np.clip(q_seed.copy(), self.jlo, self.jhi)
        if null_step > 0:
            q = np.clip(self.null_step(q, null_step), self.jlo, self.jhi)
        for _ in range(iters):
            J = self._jac(q)
            pc, Rc = self.tip_pose(d)
            e = np.concatenate([p - pc, rot_log(R @ Rc.T)])
            if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-5:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            dq *= min(1.0, 0.2 / (np.abs(dq).max() + 1e-12))
            q = np.clip(q + dq, self.jlo, self.jhi)
        d.qpos[:7] = q
        mujoco.mj_kinematics(m, d)
        pc, Rc = self.tip_pose(d)
        return q, (np.linalg.norm(p - pc), np.linalg.norm(rot_log(R @ Rc.T)))

    def aligned_rotation(self, R_ref):
        """Peg orientation with peg z = insertion axis and the cross-section matched to the hole,
        choosing the symmetric-equivalent twist closest to R_ref."""
        n = scene.symmetry_order(self.obj)
        if n == 0:  # round: shortest arc taking R_ref's z onto the axis
            z = R_ref[:, 2]
            c = np.cross(z, self.axis)
            ang = np.arctan2(np.linalg.norm(c), z @ self.axis)
            return rot_exp(c / (np.linalg.norm(c) + 1e-12) * ang) @ R_ref
        base = self.R_box @ np.diag([1.0, -1.0, -1.0])  # peg z = -box z, peg x = box x
        cands = [base @ rot_exp(np.array([0, 0, 2 * np.pi * k / n])) for k in range(n)]
        return min(cands, key=lambda R: np.linalg.norm(rot_log(R @ R_ref.T)))

    def aligned_pose(self, depth, R_ref):
        """Tip pose on the hole axis, `depth` metres past the mouth plane (negative = above)."""
        return self.mouth + depth * self.axis, self.aligned_rotation(R_ref)

    # ------------------------------------------------------------------ control
    def _apply_target_pose(self, p, R):
        """ALL driving logic: safety clamp (vs measured tip pose) -> IK with null-space posture step
        -> FR3 joint-velocity limit -> joint references streamed to the stock FR3 position servos
        over one control period (velocity-continuous profile + velocity and inertia feed-forward)."""
        m, d = self.model, self.data
        pc, Rc = self.tip_pose()
        dp = p - pc
        n = np.linalg.norm(dp)
        if n > MAX_DPOS:
            p = pc + dp * (MAX_DPOS / n)
        rv = rot_log(R @ Rc.T)
        th = np.linalg.norm(rv)
        if th > MAX_DROT:
            R = rot_exp(rv * (MAX_DROT / th)) @ Rc
        q_t, _ = self.ik(p, R, self.q_cmd, null_step=NULL_STEP)
        over = np.max(np.abs(q_t - self.q_cmd) / (QDOT_MAX * CONTROL_DT))
        if over > 1:  # FR3 joint-velocity limits (matters only near singularities): slow down, keep direction
            q_t = self.q_cmd + (q_t - self.q_cmd) / over
        # Joint reference: velocity-continuous piecewise-linear profile. The velocity ramps from the
        # previous segment's v0 to v1 over BLEND_T, then stays at v1; v1 is chosen so the segment
        # ends exactly at q_t. Streamed as ctrl = q_ref + (kv/kp) * v_ref (velocity feed-forward),
        # so the stock FR3 servos track without lag and without torque-saturating velocity jumps.
        q0, v0, tb = self.q_cmd, self.v_cmd, BLEND_T
        v1 = (q_t - q0 - v0 * tb / 2) / (CONTROL_DT - tb / 2)
        a_blend = (v1 - v0) / tb
        M = self._M
        for s in range(1, SUBSTEPS + 1):
            t = s * scene.DT
            if t <= tb:
                q_ref, v_ref = q0 + v0 * t + a_blend * t * t / 2, v0 + a_blend * t
                mujoco.mj_fullM(m, d, M)
                ff = M[:7, :7] @ a_blend / self.kp  # inertial (acceleration) feed-forward
            else:
                q_ref, v_ref, ff = q0 + (v0 + v1) * tb / 2 + v1 * (t - tb), v1, 0.0
            d.ctrl[:7] = q_ref + self.kv_over_kp * v_ref + ff
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)  # sensors / kinematics consistent with d.time
        self.q_cmd, self.v_cmd = q_t, v1
        self.last_target = (p, R)

    def step(self, action):
        p, R = vec_to_pose(action)
        self._apply_target_pose(p, R)
        self.k += 1
        obs = self.observe()
        depth, inside = self.insertion_depth()
        success = bool(inside and depth >= MIN_DEPTH)
        done = success or self.k >= MAX_STEPS
        return obs, done, {"success": success, "depth_mm": 1000.0 * depth}

    def insertion_depth(self):
        """(tip travel past the mouth plane along the hole axis [m], tip inside the hole's
        circumscribed cylinder). The radial guard keeps e.g. touching the plate beside the boss
        (30 mm below the mouth plane) from counting as an insertion."""
        rel = self.data.site_xpos[self.tip] - self.mouth
        along = float(rel @ self.axis)
        return along, bool(np.linalg.norm(rel - along * self.axis) <= self.hole_r_max)

    # ------------------------------------------------------------------ sensing
    def wrench(self):
        """Environment -> peg wrench at ft_site, in the sensor frame, gravity of the tool subtree removed."""
        m, d = self.model, self.data
        raw_f = d.sensordata[self._f_adr:self._f_adr + 3]
        raw_t = d.sensordata[self._t_adr:self._t_adr + 3]
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        mg = m.body_subtreemass[self.ft_body] * m.opt.gravity
        G_t = np.cross(d.subtree_com[self.ft_body] - d.site_xpos[self.ft_site], mg)
        return np.concatenate([-(raw_f + R.T @ mg), -(raw_t + R.T @ G_t)])

    def cam_pose(self):
        return self.data.cam_xpos[self.cam].copy(), self.data.cam_xmat[self.cam].reshape(3, 3).copy()

    def depth_cloud(self, depth=None):
        """Back-projected, cropped dense cloud. Returns (points (N,3), pixel flat indices (N,)).
        Uses only the depth image, the (fixed, calibrated) camera pose and the tip pose."""
        m = self.model
        if depth is None:
            r = self.renderer(DEPTH_RES, DEPTH_RES, kind="depth")
            self._bench_scene(r, shadow=False)
            depth = r.render()
        z = depth.reshape(-1)
        far = m.vis.map.zfar * m.stat.extent
        cp, _ = self.cam_pose()
        zw = cp[2] + z * self._ray_w[:, 2]
        pix = np.flatnonzero((z > 0) & (z < 0.99 * far) & (zw >= CROP_Z[0]) & (zw <= CROP_Z[1]))
        P = cp + self._ray_w[pix] * z[pix, None]
        keep = np.linalg.norm(P[:, :2] - self.tip_pose()[0][:2], axis=1) <= CROP_R
        return P[keep], pix[keep]

    def point_cloud(self):
        """(N_POINTS,14) features + the pixel index of each point (for checks). The only task input
        besides sensing is self.axis, the nominal insertion axis (given with the task)."""
        P, pix = self.depth_cloud()
        assert len(P) >= 32, f"only {len(P)} points after crop"
        sel = fps(P, min(N_POINTS, len(P)))
        if len(sel) < N_POINTS:  # peg far from the part (e.g. ablations): pad by repeating FPS points
            self.padded_frames = getattr(self, "padded_frames", 0) + 1
            sel = np.resize(sel, N_POINTS)
        feats = compute_features(P, self.cam_pose()[0], self.axis, query_idx=sel)
        return feats, pix[sel]

    def rgb(self):
        r = self.renderer(RGB_RES, RGB_RES)
        self._bench_scene(r, shadow=True)
        return r.render().copy()

    def _bench_scene(self, r, shadow):
        r.update_scene(self.data, camera=self.cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadow)

    def observe(self):
        pc, _ = self.point_cloud()
        p, R = self.tip_pose()
        return {"pc": pc, "rgb": self.rgb(), "agent_pos": pose_to_vec(p, R).astype(np.float32),
                "wrench": self.wrench().astype(np.float32), "t": float(self.data.time)}

    def render_rgb(self, w=320, h=240, azimuth=135.0, elevation=-25.0, distance=0.9):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.mouth + np.array([0, 0, 0.08])
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r = self.renderer(w, h, kind="free")
        r.update_scene(self.data, camera=cam)
        return r.render().copy()

    def contacts(self, geoms_a, geoms_b):
        c = self.data.contact[: self.data.ncon]
        return sum(1 for g1, g2 in zip(c.geom1, c.geom2)
                   if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b))
