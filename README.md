# Direction-aware point cloud encoder for force-aware peg insertion (simulation prototype)

I built a minimal direction-aware point cloud encoder: input a point cloud with per-point normals, principal curvatures (direction and magnitudes) and the nominal insertion axis (N×14) → output a trained encoder whose geometric embedding conditions a diffusion policy.

![closed-loop insertion on a held-out hole pose](figures/demo_f3-010.gif)

*Case f3-010, closed loop with the expert disabled: a Franka FR3 inserts a hexagonal peg into a hole tilted 15°, a tilt never seen in training. Left: the scene. Middle: what the encoder sees (512 points, normals, insertion axis). Right: wrist force Fz and insertion depth (success = 20 mm within 15 s).*

**Results** (frozen protocol, one config, every attempt in `results/results.json`): held-out hole pose f3 6/20 · training poses train_control 6/10 · novel cross-sections f4 0/30 (a limitation, see below) · 30° extrapolation extrap 1/10 (reported only).

What is hard-coded: simulation only (MuJoCo 3.10, Franka FR3 model), ground-truth camera extrinsics, one fixed workspace depth camera, a scripted expert that knows the hole pose up to a 0.5–1.8 mm aiming error and corrects it with wrist force, one parametric peg/hole family, and the nominal insertion axis given as a task input.

## What is implemented

The anchor chain, end to end:

1. **Scene**: Franka FR3 (MuJoCo Menagerie), a peg fixed to the flange through a wrist force/torque sensor, and a tiltable part with a matching hole (1 mm clearance per side, 30 mm deep, chamfered).
2. **Demonstrations**: 480 expert insertions over 8 training geometries (round, square, hexagon, octagon × 2 sizes), with DART noise and force-guided lateral correction.
3. **Point cloud**: depth image → back-projection → crop around the pin tip (proprioception only, no ground-truth part pose) → farthest point sampling to 512 points.
4. **Direction features** (14 channels per point): xyz, surface normal (Open3D, k = 20), principal-curvature direction and both curvature magnitudes, nominal insertion axis.
5. **Encoder**: DP3-style PointNet encoder taking all 14 channels. This is the trained model the project delivers (`weights/send_pc/encoder.pt`, loadable on its own).
6. **Policy**: encoder embedding + end-effector pose + wrist-wrench history → conditional 1-D U-Net diffusion head (DDPM training, 10-step DDIM sampling). Positions and point coordinates are expressed relative to the pin tip.
7. **Closed-loop evaluation** on a protocol frozen before any training: 70 cases, each attempted once.

## Results

| Group | What is held out | Successes / attempts |
|---|---|---|
| train_control | nothing (training geometries and tilts, new poses) | train_control 6/10 |
| f3 | hole tilt 15° (never in training), training geometries | f3 6/20 |
| f4 | cross-section shape (rectangle, triangle, pentagon; never in training) | f4 0/30 |
| extrap | hole tilt 30° (outside the training range) | extrap 1/10 |

Checks that the trained encoder is actually used:

- Closed loop, f3 with a mismatched point cloud (another case's scene): f3 0/20, versus f3 6/20 with the real one.
- Closed loop, f3 with the wrench input zeroed: f3 4/20. This is reported only, not claimed as a gain.
- Training-window loss with another episode's point cloud: 101.8× the loss with its own.
- Swapping only the direction channels (normals, curvatures) for another frame's moves the predicted positions by 3.0 mm on average; zeroing the wrench in contact moves them by 0.9 mm. The pass threshold is 0.5 mm (half the per-side clearance); it replaced an earlier threshold in normalised action units that was mis-scaled, and was set after that first measurement. The full check output is in `results/check_floor_d1.txt`.

## How to reproduce

Tested on macOS x86_64 with Python 3.12.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python third_party/fetch_menagerie.py                                  # Franka FR3 model, pinned commit
bash scripts/smoke_test.sh                                                        # fresh clone -> load committed weights -> one inference
.venv/bin/python -m policy.check_encoder weights/send_pc/encoder.pt              # the trained encoder on its own
.venv/bin/python -m eval.run --policy weights/send_pc --all --out results/pc.json   # full closed-loop evaluation, about 20 min on CPU
```

`eval.check_floor` also recomputes the sensitivity checks, so it needs the demonstrations (`data/demos.npz`, not in git): run `python -m sim.gen_demos` first (about 15 min with 3 processes), then `python -m eval.check_floor results/pc.json --ablation results/send_pc_ablate_pc_f3.json`.

Training from scratch:

- `python -m sim.gen_demos` generates the demonstrations, about 15 min with 3 processes.
- `python -m policy.train --config configs/send_pc.json --device cuda` trains the policy. It took 7 min on one T4 GPU; `policy/requirements-train-gpu.txt` lists the GPU environment.

## Limitations

- **Novel cross-sections do not work yet (f4 0/30).** The policy does not rotate the peg about the insertion axis to match an unseen polygon: the twist error reached 17–56° on the rectangle and triangle. The expert always knew the aligned twist, so the demonstrations never showed a twist search.
- Tilts outside the training range mostly fail (extrap 1/10).
- Everything is in simulation, with a privileged scripted expert, one camera with ground-truth extrinsics, one random seed, and a small U-Net sized for CPU.

## Next steps

- Real depth streams on the TACTO setup, with automatic extrinsic calibration of wrist and workspace cameras (hand-eye AX = XB).
- Novel shapes: an expert that searches the twist from the wrist torque about the insertion axis, then retrain and rerun f4.
- RGB-only baseline: configured in `configs/send_rgb.json`, not yet run, so no comparison is claimed.
- Ablations of the encoder input (xyz only, xyz + axis, all 14 channels), and expressing the input in the insertion-axis frame.
- A ROS2 interface to the Bota wrist sensor.

## Provenance and licenses

- `third_party/dp3/`: encoder and U-Net from 3D Diffusion Policy (MIT) at commit 47385d9. Every modification is listed in `third_party/dp3/SOURCE.md`.
- Franka FR3 model: MuJoCo Menagerie (Apache-2.0) at commit c96a32d, fetched by `third_party/fetch_menagerie.py`.
