"""Scene builder: FR3 + F/T body + parametric peg, and a tiltable parametric hole box.

Frames (all SI units):
- peg_tip site: origin at the peg's tip, +z = peg forward direction (flange -> tip), same
  orientation as FR3 attachment_site. agent_pos / action are this frame's pose.
- box body (mocap): origin = underside centre of the bottom plate (tilt pivot), +z = hole
  outward normal. Hole bottom at z=TB, mouth plane at z=TB+H (site "hole_mouth").
- Insertion axis (feature channel "axis") = -box z in world.
"""
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FR3_SCENE = ROOT / "third_party/menagerie/franka_fr3/scene.xml"
PROTOCOL = json.loads((ROOT / "eval/protocol.json").read_text())

# ---- fixed parameters (see checkpoint phaseA-sim.md for rationale) ----
CLR = PROTOCOL["geometry"]["clearance_mm_per_side"] / 1000.0  # 1 mm per side
H = PROTOCOL["success"]["hole_depth_mm"] / 1000.0  # 30 mm
TB = 0.010  # bottom plate thickness
TW = 0.020  # wall thickness around the hole
PLATE_HALF = 0.055  # square base plate half size (covers the largest boss: triangle, R_out=54 mm)
Z_MOUTH = 0.25  # world z of the mouth centre (kept fixed under tilt); 0.15 put tilted poses near the FR3 wrist singularity
DT = 0.002  # physics timestep
PEG_LEN = 0.10
FT_THICK = 0.02  # F/T sensor disk between flange and peg
N_ROUND = 32  # wall segments approximating a round hole
FRICTION = 0.3
SOLREF = (0.004, 1.0)
WRIST_KV = 40.0
PEG_DENSITY = 7800.0
CAM_POS = np.array([0.95, -0.30, 0.90])
CAM_LOOKAT = np.array([0.525, 0.0, 0.25])
CAM_FOVY = 40.0
OBJECTS = {o["object_id"]: o for o in PROTOCOL["objects"]["train"] + PROTOCOL["objects"]["heldout"]}
NSIDES = {"triangle": 3, "square": 4, "pentagon": 5, "hex": 6, "octagon": 8}


def peg_polygon(obj):
    """CCW cross-section vertices (n,2) in the peg frame; round -> N_ROUND-gon with apothem R
    (used only for the hole walls). Every shape is mirror-symmetric about the x axis."""
    s = obj["size_mm"]
    if obj["shape"] == "rect":  # long side along x
        hx, hy = max(s) / 2000.0, min(s) / 2000.0
        return np.array([[hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
    if obj["shape"] == "round":
        n, R = N_ROUND, s / 2000.0 / np.cos(np.pi / N_ROUND)
        ang = (np.arange(n) + 0.5) * 2 * np.pi / n  # edge normals at k*2pi/n
    else:
        n, R = NSIDES[obj["shape"]], s / 2000.0
        ang = np.arange(n) * 2 * np.pi / n  # vertex at angle 0
    return R * np.stack([np.cos(ang), np.sin(ang)], 1)


def symmetry_order(obj):
    """Rotational symmetry order of the cross-section about the peg axis (0 = continuous)."""
    return {"round": 0, "rect": 2}.get(obj["shape"], NSIDES.get(obj["shape"]))


def _edges(poly):
    """Outward unit normals and apothems of a CCW convex polygon."""
    e = np.roll(poly, -1, 0) - poly
    n = np.stack([e[:, 1], -e[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n, np.einsum("ij,ij->i", n, poly)


def _line_isect(n1, a1, n2, a2):
    return np.linalg.solve(np.array([n1, n2]), np.array([a1, a2]))


def _quat_z(theta):
    return [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]


def _look_quat(pos, target):
    f = target - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)  # camera looks along -z, y up
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, -f], 1).flatten())
    return q


def _add_hole(spec, body, obj):
    chamfer = obj["chamfer_mm"] / 1000.0
    nrm, apo = _edges(peg_polygon(obj))
    a_in, a_out = apo + CLR, apo + CLR + TW
    m = len(apo)
    zc, ztop = TB + H - chamfer, TB + H
    grey = [0.55, 0.6, 0.68, 1]
    body.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[PLATE_HALF, PLATE_HALF, TB / 2],
                  pos=[0, 0, TB / 2], rgba=[0.45, 0.5, 0.58, 1])
    for i in range(m):
        t = np.array([-nrm[i, 1], nrm[i, 0]])  # tangent (CCW)
        w0 = _line_isect(nrm[i - 1], a_out[i - 1], nrm[i], a_out[i])  # outer polygon vertices bounding edge i
        w1 = _line_isect(nrm[i], a_out[i], nrm[(i + 1) % m], a_out[(i + 1) % m])
        t0, t1 = sorted([t @ w0, t @ w1])
        tc, hl = (t0 + t1) / 2, (t1 - t0) / 2
        ang = np.arctan2(nrm[i, 1], nrm[i, 0])
        c2 = (a_in[i] + TW / 2) * nrm[i] + tc * t
        body.add_geom(name=f"wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[TW / 2, hl, (zc - TB) / 2],
                      pos=[c2[0], c2[1], (TB + zc) / 2], quat=_quat_z(ang), rgba=grey)
        # chamfered top of the wall: convex prism, cross-section (u,z) = (a,zc) (a+TW,zc) (a+TW,ztop) (a+c,ztop)
        uz = [(a_in[i], zc), (a_in[i] + TW, zc), (a_in[i] + TW, ztop), (a_in[i] + chamfer, ztop)]
        verts = [[*(u * nrm[i] + (tc + s * hl) * t), z] for (u, z) in uz for s in (-1, 1)]
        mname = f"{obj['object_id']}_chamf{i}"
        spec.add_mesh(name=mname, uservert=np.asarray(verts).flatten().tolist())
        body.add_geom(name=f"chamf{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname, rgba=grey)
    body.add_site(name="hole_mouth", pos=[0, 0, TB + H], size=[0.002, 0, 0], group=4)


def _add_peg(spec, link7, obj):
    ft = link7.add_body(name="ft", pos=[0, 0, 0.107])  # = attachment_site pose
    ft.add_site(name="ft_site", size=[0.004, 0, 0], group=4)
    ft.add_geom(name="ft_disk", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, FT_THICK / 2, 0],
                pos=[0, 0, FT_THICK / 2], contype=0, conaffinity=0, group=1, density=2700,
                rgba=[0.15, 0.15, 0.17, 1])
    peg = ft.add_body(name="peg", pos=[0, 0, FT_THICK])
    kw = dict(name="peg", friction=[FRICTION, 0.005, 0.0001], solref=list(SOLREF), density=PEG_DENSITY,
              rgba=[0.95, 0.55, 0.1, 1])
    if obj["shape"] == "round":
        peg.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[obj["size_mm"] / 2000.0, PEG_LEN / 2, 0],
                     pos=[0, 0, PEG_LEN / 2], **kw)
    else:
        poly = peg_polygon(obj)
        v = np.concatenate([np.c_[poly, np.zeros(len(poly))], np.c_[poly, np.full(len(poly), PEG_LEN)]])
        spec.add_mesh(name="peg_mesh", uservert=v.flatten().tolist())
        peg.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="peg_mesh", **kw)
    peg.add_site(name="peg_tip", pos=[0, 0, PEG_LEN], size=[0.002, 0, 0], group=4)
    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="ft_site")
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="ft_site")


@lru_cache(maxsize=None)
def build_model(object_id):
    obj = OBJECTS[object_id]
    spec = mujoco.MjSpec.from_file(str(FR3_SCENE))
    spec.option.timestep = DT
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 640
    spec.visual.quality.numslices = 64
    spec.visual.quality.shadowsize = 1024
    _add_peg(spec, spec.body("fr3_link7"), obj)
    for b in spec.bodies:  # arm + tool gravity compensation (the real FR3 controller does this too)
        if b.name.startswith("fr3_link") or b.name in ("ft", "peg"):
            b.gravcomp = 1.0
    # Wrist servos: kv 200 -> WRIST_KV. With kv=200 on ~0.08 kg m^2 at dt=2 ms, MuJoCo clamps the
    # *explicit* force (before the implicit damping correction) at the 12 Nm joint limit, which
    # stalls the wrist during fast rotations (tracking error ~1.8 mm, implicit torque up to 170 Nm).
    # kv=40 (damping ratio ~1.3-1.6) keeps the 12 Nm limit physical; kp unchanged.
    for j in (5, 6, 7):
        a = spec.actuator(f"fr3_joint{j}")
        a.biasprm = [a.biasprm[0], a.biasprm[1], -WRIST_KV] + list(a.biasprm[3:])
    box = spec.worldbody.add_body(name="box", mocap=True)
    _add_hole(spec, box, obj)
    for g in box.geoms:
        g.friction = [FRICTION, 0.005, 0.0001]
        g.solref = list(SOLREF)
    stand = spec.worldbody.add_body(name="stand", mocap=True)  # visual support post (top = box pivot)
    stand.add_geom(name="stand", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.2, 0], pos=[0, 0, -0.2],
                   contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.32, 1])
    spec.worldbody.add_camera(name="bench", pos=CAM_POS.tolist(), quat=_look_quat(CAM_POS, CAM_LOOKAT).tolist(),
                              fovy=CAM_FOVY)
    spec.worldbody.add_light(pos=[0.9, -0.6, 1.2], dir=[-0.5, 0.4, -0.8], diffuse=[0.5, 0.5, 0.5],
                              castshadow=False)
    return spec.compile()


def box_pose(case):
    """World pose of the box body (pivot) for a protocol case. R = Rz(yaw) @ Ry(tilt)."""
    t, y = np.deg2rad(case["tilt_deg"]), np.deg2rad(case["yaw_deg"])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    R = Rz @ Ry
    mouth = np.array([case["hole_xy"][0], case["hole_xy"][1], Z_MOUTH])
    return mouth - R @ np.array([0, 0, TB + H]), R, mouth
