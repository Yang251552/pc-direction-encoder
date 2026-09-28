"""14-channel per-point features (world frame):
xyz(3) | normal(3) | pcurv_dir(3) | pcurv_mag(2: |k1| >= |k2|) | axis(3)

- normal: Open3D estimate_normals (KNN=20), oriented towards the camera.
- principal curvature (approximate): per point, least-squares fit of the 2x2 shape operator
  S in the tangent plane from the neighbours' normal differences projected on that plane
  (dn_t ~= S dc_t), then eigen-decomposition. A PCA normal is the normal at its neighbourhood
  centroid, so each neighbour normal is paired with its neighbourhood centroid c (pairing with
  the raw point biased k1 by ~10% on cylinders). pcurv_mag = eigenvalue magnitudes sorted
  descending; pcurv_dir = eigenvector of the larger-magnitude curvature. Flat points
  (|k1| < KAPPA_FLAT) get dir = 0.
- pcurv_dir sign: dot(dir, axis) >= 0; if that dot is ~0 use dot(dir, world x) >= 0 (then y).
- axis: unit insertion axis (direction the peg advances into the hole), broadcast.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

KNN = 20
KAPPA_FLAT = 20.0  # 1/m: radius > 50 mm counts as flat (smallest designed radius 8 mm = 125 /m)
SIGN_EPS = 1e-3


def normals_o3d(points, cam_pos):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(KNN))
    pcd.orient_normals_towards_camera_location(np.asarray(cam_pos, np.float64))
    return np.asarray(pcd.normals)


def principal_curvature(points, normals, query_idx, tree=None):
    """Curvature at points[query_idx] using KNN neighbours from the full (dense) cloud.
    Returns (dir (m,3) unsigned, mag (m,2) with mag[:,0] >= mag[:,1] >= 0)."""
    tree = tree or cKDTree(points)
    _, nb = tree.query(points[query_idx], k=KNN, workers=2)
    uniq, inv = np.unique(nb, return_inverse=True)
    _, nb2 = tree.query(points[uniq], k=KNN, workers=2)
    cent = points[nb2].mean(1)[inv.reshape(nb.shape)]  # (m,k,3) neighbourhood centroid of each neighbour
    n = normals[query_idx]
    # tangent basis per point
    a = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    b1 = np.cross(n, a)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = np.cross(n, b1)
    B = np.stack([b1, b2], 2)  # (m,3,2)
    dp = cent - points[nb].mean(1)[:, None]  # (m,k,3) centroid differences
    dn = normals[nb] - n[:, None]
    X = dp @ B  # (m,k,2)
    Y = dn @ B
    XtX = X.transpose(0, 2, 1) @ X + 1e-12 * np.eye(2)
    St = np.linalg.solve(XtX, X.transpose(0, 2, 1) @ Y)  # (m,2,2), Y ~= X St
    S = 0.5 * (St + St.transpose(0, 2, 1))
    w, v = np.linalg.eigh(S)  # ascending
    order = np.argsort(-np.abs(w), axis=1)
    mag = np.take_along_axis(np.abs(w), order, 1)
    v1 = np.take_along_axis(v, order[:, None, :1], 2)[..., 0]  # (m,2) eigenvector of max |k|
    return np.einsum("mij,mj->mi", B, v1), mag


def compute_features(points, cam_pos, axis, query_idx=None):
    """points: dense (N,3) world cloud; query_idx: indices of the output points (default all).
    Returns (m,14) float32."""
    points = np.asarray(points, np.float64)
    query_idx = np.arange(len(points)) if query_idx is None else np.asarray(query_idx)
    normals = normals_o3d(points, cam_pos)
    d, mag = principal_curvature(points, normals, query_idx)
    d[mag[:, 0] < KAPPA_FLAT] = 0.0
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    d = sign_convention(d, axis)
    m = len(query_idx)
    return np.concatenate([points[query_idx], normals[query_idx], d, mag, np.broadcast_to(axis, (m, 3))],
                          1).astype(np.float32)


def sign_convention(d, axis):
    """dot(d, axis) >= 0; where that dot is ~0 fall back to world x, then world y."""
    d = d.copy()
    todo = np.linalg.norm(d, axis=1) > 0
    for ref in (axis, np.array([1.0, 0, 0]), np.array([0, 1.0, 0])):
        s = d @ ref
        decided = todo & (np.abs(s) > SIGN_EPS)
        d[decided & (s < 0)] *= -1
        todo &= ~decided
    return d
