"""Visual encoders. PointCloudEncoder is the anchor deliverable (G2); RGBEncoder is the f5 baseline.

PointCloudEncoder input is always the data-contract point cloud, raw, world frame:
(B, N, 14) = xyz 0-2 | normal 3-5 | pcurv_dir 6-8 | pcurv_mag (k1, k2) 9-10 | axis 11-13.
It picks its channel subset (14 = all / 6 = xyz + axis / 3 = xyz, for the G4 comparison),
applies fixed dataset-level preprocessing stored as buffers, and runs the vendored DP3
PointNet (PointNetEncoderXYZ, same architecture for every C, DP3 widths, out 64).

Preprocessing (fitted once on the training clouds, saved in encoder.pt):
- xyz: per-axis centre + ONE isotropic scale (angles between geometry and the unit
  direction channels are preserved). No per-frame centring: the hole position is only
  visible in the point cloud and actions are absolute world poses.
- direction channels (normal, pcurv_dir, axis) are unit or zero vectors: passed through.
- pcurv_mag: one scale for both k (p99 of |k| over non-zero entries), then clamp +-3;
  zero stays zero (flat-region convention). Clamp bounds estimator outliers at
  edges; revisit if real-data k is heavy-tailed beyond 3x p99 on many points.

Reference frame: with the policy's `relative_pos` on, xyz is expressed relative to the last
observed pin tip BEFORE this encoder (to_reference_frame); the fitted xyz affine is then in that
frame too. encoder.pt records it as `xyz_frame` ("pin_tip" or "world").

Stand-alone use:  enc, ckpt = load_encoder("runs/<name>/encoder.pt")
                  emb = enc(to_reference_frame(pc, pin_tip) if ckpt["xyz_frame"] == "pin_tip" else pc)
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision

from third_party.dp3.pointnet_extractor import PointNetEncoderXYZ

PC_CHANNELS = 14
DIRECTION_CHANNELS = slice(3, 11)  # normal, pcurv_dir, pcurv_mag: what the f1 check swaps (protocol amendment)
KAPPA = slice(9, 11)
CHANNEL_SETS = {14: list(range(14)), 6: [0, 1, 2, 11, 12, 13], 3: [0, 1, 2]}


def to_reference_frame(pc, ref):
    """Translate the xyz channels by -ref (ref broadcasts over the point axis, e.g. pc (B, To, N, 14)
    with ref (B, 1, 1, 3)); normal / pcurv / axis channels are translation invariant and untouched."""
    return torch.cat([pc[..., :3] - ref, pc[..., 3:]], dim=-1)


class PointCloudEncoder(nn.Module):
    kind = "pc"

    def __init__(self, channels=14, out_dim=64, use_layernorm=True, final_norm="layernorm"):
        super().__init__()
        self.config = dict(kind="pc", channels=channels, out_dim=out_dim,
                           use_layernorm=use_layernorm, final_norm=final_norm)
        self.out_dim = out_dim
        self.idx = CHANNEL_SETS[channels]
        self.register_buffer("xyz_center", torch.zeros(3))
        self.register_buffer("xyz_scale", torch.ones(()))
        self.register_buffer("kappa_scale", torch.ones(()))
        # DP3 defaults (dp3.yaml): out 64, layernorm, final layernorm
        self.pointnet = PointNetEncoderXYZ(in_channels=channels, out_channels=out_dim,
                                           use_layernorm=use_layernorm, final_norm=final_norm)

    @torch.no_grad()
    def fit(self, pc):
        """pc: (..., 14) training point clouds (np or torch)."""
        pc = np.asarray(pc, dtype=np.float32).reshape(-1, PC_CHANNELS)
        lo, hi = pc[:, :3].min(0), pc[:, :3].max(0)
        self.xyz_center.copy_(torch.from_numpy((lo + hi) / 2))
        self.xyz_scale.fill_(max(float((hi - lo).max()) / 2, 1e-6))
        k = np.abs(pc[:, KAPPA])
        k = k[k > 0]
        self.kappa_scale.fill_(max(float(np.percentile(k, 99)), 1e-6) if k.size else 1.0)

    def preprocess(self, pc):
        x = torch.cat([(pc[..., :3] - self.xyz_center) / self.xyz_scale, pc[..., 3:9],
                       (pc[..., KAPPA] / self.kappa_scale).clamp(-3, 3), pc[..., 11:14]], dim=-1)
        return x[..., self.idx]

    def forward(self, pc):
        assert pc.shape[-1] == PC_CHANNELS, f"expects contract point cloud (B, N, 14), got {tuple(pc.shape)}"
        return self.pointnet(self.preprocess(pc))


class RGBEncoder(nn.Module):
    """f5 baseline: ResNet18, every BatchNorm -> GroupNorm(C // 16) (as in Diffusion Policy),
    random init (weights=None: nothing is downloaded), fc -> Linear(512, out) + LayerNorm
    (mirrors DP3's final projection so both visual embeddings enter the UNet alike).
    crop (Diffusion Policy standard): in train() mode each image gets an independent random
    crop x crop window, in eval() mode the centre crop. The point cloud branch has no augmentation;
    this asymmetry is part of the f5 setup (recorded in configs/send_rgb.json)."""
    kind = "rgb"

    def __init__(self, out_dim=64, crop=None):
        super().__init__()
        self.config = dict(kind="rgb", out_dim=out_dim, crop=crop)
        self.out_dim, self.crop = out_dim, crop
        self.net = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
        self.net.fc = nn.Sequential(nn.Linear(512, out_dim), nn.LayerNorm(out_dim))

    def fit(self, rgb):
        pass  # fixed [0, 255] -> [-1, 1]; no data statistics

    def forward(self, rgb):
        """rgb: (B, H, W, 3) uint8 (contract layout)."""
        assert rgb.shape[-1] == 3, f"expects (B, H, W, 3), got {tuple(rgb.shape)}"
        x = rgb.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        if self.crop:
            c, (H, W) = self.crop, x.shape[-2:]
            if self.training:
                dy = torch.randint(0, H - c + 1, (len(x),)).tolist()
                dx = torch.randint(0, W - c + 1, (len(x),)).tolist()
                x = torch.stack([im[:, y:y + c, z:z + c] for im, y, z in zip(x, dy, dx)])
            else:
                x = x[..., (H - c) // 2:(H - c) // 2 + c, (W - c) // 2:(W - c) // 2 + c]
        return self.net(x)


def build_encoder(config):
    config = dict(config)
    kind = config.pop("kind")
    return {"pc": PointCloudEncoder, "rgb": RGBEncoder}[kind](**config)


def load_encoder(path):
    """Load ONLY the encoder from encoder.pt (no policy code needed beyond this file)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    enc = build_encoder(ckpt["config"])
    enc.load_state_dict(ckpt["state_dict"])
    return enc.eval(), ckpt
