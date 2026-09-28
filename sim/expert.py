"""Force-reactive peg-in-hole expert working in the hole frame, plus DART execution noise.

Perception model: per episode the expert believes the mouth centre is at the true centre plus a
lateral aiming error (hole-frame lateral plane, |e| ~ U(AIM_ERR), random direction, seeded by the
case's own init_seed). Hole axis direction and cross-section twist are known. The expert never
reads the true lateral hole position after reset; lateral corrections come only from the F/T.

approach: tip to the believed pre-insert pose (PRE_DIST outside the believed mouth on the axis,
          peg axis = hole axis, twist = nearest symmetric-equivalent alignment), <= APP_DPOS /
          APP_DROT per step.
insert:   advance along the axis through the believed centre + correction c. In contact
          (|F| > F_CONTACT near/inside the mouth) the lateral force (world frame, perpendicular to
          the axis; env -> peg, so a chamfer pushes towards the hole centre) moves the target:
          c += K_LAT * F_lat, clipped to DC_MAX per step (compliant centring). Axial force above
          F_AX_HOLD: stop advancing; above F_BACK_MAX: back off. Pressed on the mouth plane with
          no lateral force: small spiral search around the current estimate (force signal only).
          Never commands deeper than MAX_CMD_DEPTH (success at 20 mm, bottom at 30 mm).
DART:     executed = label (+) Gaussian noise: full sigma while the label tip is more than
          NOISE_GATE outside the mouth, IN_HOLE_SCALE * sigma inside the gate. Inside a 1 mm-
          clearance hole the stiff FR3 servos turn full 2 mm / 2 deg noise into 50-300 N wall
          forces (measured); sigma/4 keeps some real contact in the data at < 50 N.
"""
import numpy as np

from sim import scene
from sim.env import PegEnv, pose_to_vec, rot_exp, rot_log

PRE_DIST = 0.06
APP_DPOS = 0.010
APP_DROT = np.deg2rad(2.0)
SWITCH_TOL = (0.006, np.deg2rad(3.0))  # approach -> insert (loose: insert keeps correcting laterally)
ADV_FREE = 0.006  # axial advance per step while the tip is > NEAR outside the mouth (stops at -NEAR)
ADV_NEAR = 0.0005  # slow zone SLOW_LO <= depth < IN_DEPTH (mouth + chamfer): gentle first touch
SLOW_LO = -0.002  # >= 3 x the in-gate DART noise (0.5 mm) before the mouth plane
ADV_IN = 0.002  # inside the hole
IN_DEPTH = 0.003
ADV_CONTACT = 0.0005  # advance per step while in (light) contact
NEAR = 0.005
MAX_CMD_DEPTH = 0.025
AIM_ERR = (0.0005, 0.0018)  # lateral aiming error magnitude range [m]
F_CONTACT = 1.5  # N: contact detection (inertial transients in free motion stay below ~1 N)
K_LAT = 1e-4  # m/N compliant lateral correction gain
DC_MAX = 0.0006  # m, lateral correction per step
F_LAT_MIN = 0.5  # N: smaller lateral force carries no direction information
F_AX_HOLD = 6.0  # N: stop advancing
F_BACK_MAX = 15.0  # N: back off
BACK_OFF = 0.0005
ALIGN_TOL = 0.0005
SPIRAL_AFTER = 2  # consecutive "pressed, no lateral force, not going deeper" steps
SPIRAL_DTHETA, SPIRAL_R0, SPIRAL_DR = 0.8, 0.0004, 0.00012  # rad/step, m, m/step
NOISE_GATE = 0.010
IN_HOLE_SCALE = 0.25
TRAIN_TILTS = tuple(float(t) for t in scene.PROTOCOL["hole_pose"]["train_tilt_deg"])
PROTOCOL_SEEDS = {c["init_seed"] for c in scene.PROTOCOL["cases"]}


class Expert:
    def __init__(self, env: PegEnv, sigma_pos=0.0, sigma_rot=0.0, seed=0, aim_err=AIM_ERR):
        """sigma_pos: DART position noise, std per axis [m]; sigma_rot: DART rotation noise as RMS
        rotation angle [rad] (isotropic rotation vector, std sigma_rot/sqrt(3) per axis);
        aim_err: (lo, hi) lateral aiming error magnitude [m], or None for a perfect aim."""
        self.env, self.sigma_pos, self.sigma_rot, self.aim_err = env, sigma_pos, sigma_rot, aim_err
        self.rng = np.random.default_rng(seed)

    def reset(self):
        env = self.env
        self.R_goal = env.aligned_rotation(env.tip_pose()[1])
        self.phase = "approach"
        self.lat_basis = env.R_box[:, :2]  # hole-frame lateral plane
        r = np.random.default_rng([env.case["init_seed"], 7])  # the episode's own seed
        mag = r.uniform(*self.aim_err) if self.aim_err else 0.0
        ang = r.uniform(0, 2 * np.pi)
        self.aim = mag * (self.lat_basis @ np.array([np.cos(ang), np.sin(ang)]))
        self.mouth_hat = env.mouth + self.aim  # believed mouth centre (set once, never refreshed)
        self.c = np.zeros(3)  # force-based lateral correction
        self.corr_steps, self.stuck, self.spiral_k, self.spiral_off = 0, 0, 0, np.zeros(3)
        self.last_depth = -np.inf

    def label(self):
        """Expert target pose for the current state (uses the believed mouth + F/T only)."""
        env = self.env
        p, R = env.tip_pose()
        centre = self.mouth_hat + self.c
        if self.phase == "approach":
            dp = centre - PRE_DIST * env.axis - p
            rv = rot_log(self.R_goal @ R.T)
            if np.linalg.norm(dp) > SWITCH_TOL[0] or np.linalg.norm(rv) > SWITCH_TOL[1]:
                n = max(1.0, np.linalg.norm(dp) / APP_DPOS, np.linalg.norm(rv) / APP_DROT)
                return p + dp / n, rot_exp(rv / n) @ R
            self.phase = "insert"
        rel = p - centre
        depth = rel @ env.axis  # the aiming error is lateral, so this is the true depth
        lateral = np.linalg.norm(rel - depth * env.axis)
        F = env.data.site_xmat[env.ft_site].reshape(3, 3) @ env.wrench()[:3]  # env -> peg, world
        f_ax = F @ env.axis
        f_back = -f_ax
        F_lat = F - f_ax * env.axis
        contact = np.linalg.norm(F) > F_CONTACT and depth > -NEAR
        if contact and np.linalg.norm(F_lat) > F_LAT_MIN:  # compliant centring
            dc = K_LAT * F_lat
            n = np.linalg.norm(dc)
            self.c += dc * min(1.0, DC_MAX / n)
            self.corr_steps += 1
            self.stuck = 0
        elif contact and f_back > F_AX_HOLD and depth <= self.last_depth + 2e-4:
            self.stuck += 1  # pressed on the mouth plane without lateral information
        else:
            self.stuck = 0
        if self.stuck >= SPIRAL_AFTER:  # spiral search around the current estimate
            th = SPIRAL_DTHETA * self.spiral_k
            off = (SPIRAL_R0 + SPIRAL_DR * self.spiral_k) * (self.lat_basis @ np.array([np.cos(th), np.sin(th)]))
            self.c += off - self.spiral_off
            self.spiral_off, self.spiral_k = off, self.spiral_k + 1
            self.corr_steps += 1
        self.last_depth = depth
        if f_back > F_BACK_MAX:
            d_cmd = depth - BACK_OFF
        elif contact and f_back > F_AX_HOLD:
            d_cmd = depth  # hold while the lateral correction takes effect
        elif contact:
            d_cmd = min(depth + ADV_CONTACT, MAX_CMD_DEPTH)
        elif -(NEAR + ADV_FREE) < depth < 0 and lateral > ALIGN_TOL:
            d_cmd = depth  # at the mouth: centre on the (believed) axis before going in
        elif depth < SLOW_LO:  # free space: fast, but never jump past the start of the slow zone
            d_cmd = min(depth + (ADV_FREE if depth < -NEAR else ADV_IN), SLOW_LO + ADV_NEAR)
        else:
            d_cmd = min(depth + (ADV_NEAR if depth < IN_DEPTH else ADV_IN), MAX_CMD_DEPTH)
        return self.mouth_hat + self.c + d_cmd * env.axis, self.R_goal

    def true_lateral_error(self):
        """For logging only (never used by label()): lateral offset of the tip from the true axis."""
        rel = self.env.tip_pose()[0] - self.env.mouth
        return float(np.linalg.norm(rel - (rel @ self.env.axis) * self.env.axis))

    def perturb(self, p, R):
        """DART: executed pose = label (+) Gaussian noise. Returns (p_exec, R_exec, full_sigma)."""
        if self.sigma_pos == 0 and self.sigma_rot == 0:
            return p, R, False
        full = (p - self.env.mouth) @ self.env.axis < -NOISE_GATE
        k = 1.0 if full else IN_HOLE_SCALE
        return (p + self.rng.normal(0, k * self.sigma_pos, 3),
                rot_exp(self.rng.normal(0, k * self.sigma_rot / np.sqrt(3), 3)) @ R, full)


def random_case(rng, object_id, tilts=TRAIN_TILTS):
    """Fresh case in the protocol hole_xy range; yaw U(0,360); init_seed never a protocol seed."""
    (x0, x1), (y0, y1) = scene.PROTOCOL["hole_pose"]["hole_xy_range_m"]
    seed = int(rng.integers(0, 2**31 - 1))
    while seed in PROTOCOL_SEEDS:
        seed = int(rng.integers(0, 2**31 - 1))
    return {"object_id": object_id, "tilt_deg": float(rng.choice(tilts)), "yaw_deg": float(rng.uniform(0, 360)),
            "hole_xy": [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))], "init_seed": seed}


def run_episode(env, case, expert, record=False):
    """Returns dict(success, steps, depth_mm, diffs, noisy, f_max, frames, aim_mm, belief_err_mm,
    tip_lat_first_contact_mm, tip_lat_end_mm, corr_steps). diffs[k] = executed (after the env
    safety clamp) minus label, as (dp, rotation vector); noisy[k] = full-sigma step.
    frames (if record): list of (obs, label_vec), obs taken at the state the label was computed on."""
    obs = env.reset(case)
    expert.reset()
    frames, diffs, noisy, fmax, done, lat_first = [], [], [], 0.0, False, None
    info = {"success": False, "depth_mm": float("nan")}
    while not done:
        before = expert.corr_steps
        lat_now = expert.true_lateral_error()
        p, R = expert.label()
        if before == 0 and expert.corr_steps > 0:
            lat_first = lat_now
        if record:
            frames.append((obs, pose_to_vec(p, R).astype(np.float32)))
        pe, Re, nz = expert.perturb(p, R)
        obs, done, info = env.step(pose_to_vec(pe, Re))
        pe, Re = env.last_target  # what was actually executed (after the env safety clamp)
        diffs.append((pe - p, rot_log(Re @ R.T)))
        noisy.append(nz)
        fmax = max(fmax, float(np.linalg.norm(obs["wrench"][:3])))
    return {"success": info["success"], "steps": env.k, "depth_mm": info["depth_mm"], "diffs": diffs,
            "noisy": noisy, "f_max": fmax, "frames": frames, "aim_mm": 1e3 * float(np.linalg.norm(expert.aim)),
            "belief_err_mm": 1e3 * float(np.linalg.norm(expert.aim + expert.c)),
            "tip_lat_first_contact_mm": None if lat_first is None else 1e3 * lat_first,
            "tip_lat_end_mm": 1e3 * expert.true_lateral_error(), "corr_steps": expert.corr_steps,
            "spiral_steps": expert.spiral_k}
