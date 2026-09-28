# DP3 vendor record

Upstream: https://github.com/YanjieZe/3D-Diffusion-Policy @ `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97` (master, 2025-10-17), MIT License (copied as `LICENSE`).
Only the point cloud encoder and the conditional UNet1D are vendored; the policy wrapper, normalizer,
sampler and dp3.yaml were read for defaults only (not vendored).

The `upstream sha256` column is the sha256 of the file at that commit (checked against
raw.githubusercontent.com on 2026-09-28). `.venv/bin/python -m policy.checks b1` rebuilds every upstream
file by reverse-applying the diff below to the vendored file and compares the sha256, so the vendored
files differ from upstream by exactly the registered changes and nothing else.

| vendored file | upstream path | upstream sha256 |
|---|---|---|
| LICENSE | LICENSE | d62cd477abcbf7329c15d3e740c9466d07dc39ef3a212e4c9b237675c8bc7a7d |
| pointnet_extractor.py | 3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py | d1fbe151475a8b055af588533206249881f1387c3f12ee0f069d0d30838ea5d7 |
| conditional_unet1d.py | 3D-Diffusion-Policy/diffusion_policy_3d/model/diffusion/conditional_unet1d.py | a5f58a998f38f027aaaa92e1085b1fee99abd51d4f6479d0ffecef3b1228ec12 |
| conv1d_components.py | 3D-Diffusion-Policy/diffusion_policy_3d/model/diffusion/conv1d_components.py | ea2aa10e297cf253bc1a1919ba47fe6ae32049b5eaaab292ed9cc145d8d2adc0 |
| positional_embedding.py | 3D-Diffusion-Policy/diffusion_policy_3d/model/diffusion/positional_embedding.py | 2266c75ecc9c9ac8ca126839d39a7bafbdd7af8800d6de903f66815e64cc366c |

## Registered changes (all marked `[pc-direction-encoder]` in the code)

pointnet_extractor.py
1. Drop `import torchvision` (imported but unused upstream; not needed by the encoder).
2. Replace `from termcolor import cprint` by a no-op `cprint` (termcolor not installed; DP3 logging silenced).
3. `PointNetEncoderXYZ`: remove `assert in_channels == 3`, so C = 3 / 6 / 14 share the same architecture
   (only the first Linear's input width changes).
4. `DP3Encoder`: `pointcloud_encoder_cfg.in_channels = self.point_cloud_shape[-1]` instead of `= 3`
   (consistency; this project builds `PointNetEncoderXYZ` directly via `policy/encoder.py`).

conditional_unet1d.py
5. Drop the `einops` / `termcolor` imports (not installed); package-relative imports of
   `conv1d_components` and `positional_embedding` instead of `diffusion_policy_3d.model.diffusion.*`.
6. Define `Rearrange('batch t -> batch t 1')` as `nn.Unflatten(1, (-1, 1))` (the only pattern used).
7. Three `einops.rearrange` calls ('b h t -> b t h' and back) become `.transpose(1, 2)`.

LICENSE, conv1d_components.py, positional_embedding.py: unchanged.

## Defaults taken from upstream `dp3.yaml` (read, not vendored)

Kept: encoder out 64, `use_layernorm`, final LayerNorm; state MLP (64, 64); UNet kernel 5, n_groups 8,
diffusion-step embedding 128, FiLM on down/mid/up; DDPM 100 train steps, DDIM 10 inference steps,
`squaredcos_cap_v2`, `clip_sample`, `prediction_type: sample`, `set_alpha_to_one`; AdamW lr 1e-4,
betas (0.95, 0.999), eps 1e-8, wd 1e-6; cosine schedule, 500 warmup steps; EMA power 0.75, max 0.9999;
min-max ('limits') normalisation of action / state.
Changed here (see `configs/send_*.json`): horizon / obs / action steps 8 / 2 / 4 (SPEC 5) instead of
16 / 2 / 8; UNet `down_dims` (64, 128, 256) instead of (512, 1024, 2048) (CPU budget); batch 64 instead
of 128; point cloud 14 channels with fixed isotropic xyz scaling instead of per-channel min-max of xyz;
wrench (not in DP3) is min-max normalised after clipping to its training [p1, p99] per dimension.
The RGB-only baseline (not DP3) uses Diffusion Policy's random crop 96->88 / centre crop at inference;
see `notes` in `configs/send_rgb.json`.

## Diff (vendored vs upstream; unified, 1 line of context)

```diff
--- a/pointnet_extractor.py
+++ b/pointnet_extractor.py
@@ -3,3 +3,2 @@
 import torch.nn.functional as F
-import torchvision
 import copy
@@ -7,3 +6,4 @@
 from typing import Optional, Dict, Tuple, Union, List, Type
-from termcolor import cprint
+def cprint(*args, **kwargs):  # [pc-direction-encoder] termcolor not installed; DP3 logging silenced
+    pass
 
@@ -133,3 +133,3 @@
         
-        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
+        # [pc-direction-encoder] 3-channel assert removed: C = 3 / 6 / 14 share this architecture
        
@@ -242,3 +242,3 @@
             else:
-                pointcloud_encoder_cfg.in_channels = 3
+                pointcloud_encoder_cfg.in_channels = self.point_cloud_shape[-1]  # [pc-direction-encoder] was 3
                 self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
--- a/conditional_unet1d.py
+++ b/conditional_unet1d.py
@@ -5,8 +5,10 @@
 import torch.nn.functional as F
-import einops
-from einops.layers.torch import Rearrange
-from termcolor import cprint
-from diffusion_policy_3d.model.diffusion.conv1d_components import (
+from .conv1d_components import (
     Downsample1d, Upsample1d, Conv1dBlock)
-from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb
+from .positional_embedding import SinusoidalPosEmb
+
+
+def Rearrange(pattern):  # [pc-direction-encoder] einops not installed; only 'batch t -> batch t 1' is used
+    assert pattern == 'batch t -> batch t 1'
+    return nn.Unflatten(1, (-1, 1))
 
@@ -273,3 +275,3 @@
         """
-        sample = einops.rearrange(sample, 'b h t -> b t h')
+        sample = sample.transpose(1, 2)  # [pc-direction-encoder] was einops 'b h t -> b t h'
 
@@ -295,3 +297,3 @@
         if local_cond is not None:
-            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
+            local_cond = local_cond.transpose(1, 2)  # [pc-direction-encoder] was einops
             resnet, resnet2 = self.local_cond_encoder
@@ -343,3 +345,3 @@
 
-        x = einops.rearrange(x, 'b t h -> b h t')
+        x = x.transpose(1, 2)  # [pc-direction-encoder] was einops 'b t h -> b h t'
 
```
