"""Diffusion policy around the visual encoder (DP3-style conditional UNet1D, CPU-sized).

Global condition = three NAMED embeddings concatenated in COND_ORDER (G3):
  visual    : encoder(frame) for each of the n_obs_steps frames      -> To * 64
              ('pc' = PointCloudEncoder, or 'rgb' = RGBEncoder for the f5 baseline)
  agent_pos : MLP(normalised pose) for each frame (DP3 state MLP)     -> To * 64
  wrench    : MLP over the normalised wrench HISTORY of the same To frames -> 64

relative_pos (translation equivariance, C3 repair r1): the reference point is the last observed pin
tip, ref = agent_pos[:, -1, :3]. Inside the model, point cloud xyz and the agent_pos positions are
taken relative to ref, and position actions are displacements from ref (normalisers are fitted on
these relative quantities); rotations (6D) and all direction / wrench channels stay as they are.
agent_pos keeps its relative positions rather than being dropped: the last frame is 0 by construction
and the earlier frame gives the recent motion, so no absolute position enters the network.
sample_chunk() / predict() add ref back: callers always see ABSOLUTE target poses.

Training: DDPMScheduler (100 steps). Inference: DDIMScheduler (10 steps, eta = 0).
prediction_type 'sample' + squaredcos_cap_v2 + clip_sample, as in DP3's dp3.yaml.
Horizon 8 / obs 2 / execute n_action_steps: UNet outputs (B, 8, 9), predict returns (n_action_steps, 9)
(SPEC 5 had 4; send_pc uses 2 since C3 repair r2). Point clouds may have any number of points N.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler

from policy.encoder import build_encoder, to_reference_frame
from third_party.dp3.conditional_unet1d import ConditionalUnet1D
from third_party.dp3.pointnet_extractor import create_mlp

COND_ORDER = ("visual", "agent_pos", "wrench")


class MinMax(nn.Module):
    """Per-dim min-max to [-1, 1]; near-constant dims (range < 1e-4) are only shifted to 0
    (DP3 LinearNormalizer 'limits' rule). pct < 100: robust variant, the range is the training
    [100 - pct, pct] percentiles and inputs are clipped to it first (wrench: contact spikes at the
    hole bottom, ~99 N, would otherwise squash the free-space range)."""

    def __init__(self, dim, pct=100.0):
        super().__init__()
        self.pct = pct
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("offset", torch.zeros(dim))
        self.register_buffer("lo", torch.full((dim,), -torch.inf))
        self.register_buffer("hi", torch.full((dim,), torch.inf))

    @torch.no_grad()
    def fit(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.scale.numel())
        lo, hi = (torch.from_numpy(np.percentile(x, q, axis=0).astype(np.float32)) for q in (100 - self.pct, self.pct))
        if self.pct < 100:
            self.lo.copy_(lo)
            self.hi.copy_(hi)
        flat = (hi - lo) < 1e-4
        scale = 2.0 / torch.where(flat, torch.full_like(lo, 2.0), hi - lo)
        self.scale.copy_(scale)
        self.offset.copy_(torch.where(flat, -lo, -1.0 - scale * lo))

    def normalize(self, x):
        return torch.maximum(torch.minimum(x, self.hi), self.lo) * self.scale + self.offset

    def unnormalize(self, y):
        return (y - self.offset) / self.scale


def mlp(in_dim, sizes):
    """DP3 state-MLP convention: hidden = sizes[:-1], out = sizes[-1], no final activation."""
    return nn.Sequential(*create_mlp(in_dim, sizes[-1], list(sizes[:-1])))


class DiffusionPolicy(nn.Module):
    def __init__(self, visual="pc", channels=14, horizon=8, n_obs_steps=2, n_action_steps=4,
                 action_dim=9, agent_pos_dim=9, wrench_dim=6, enc_out_dim=64, state_mlp=(64, 64),
                 wrench_mlp=(64, 64), wrench_clip_pct=99.0, rgb_crop=None, down_dims=(64, 128, 256),
                 kernel_size=5, n_groups=8, dsed=128, num_train_timesteps=100, num_inference_steps=10,
                 prediction_type="sample", relative_pos=False):
        super().__init__()
        self.config = dict(visual=visual, channels=channels, horizon=horizon, n_obs_steps=n_obs_steps,
                           n_action_steps=n_action_steps, action_dim=action_dim, agent_pos_dim=agent_pos_dim,
                           wrench_dim=wrench_dim, enc_out_dim=enc_out_dim, state_mlp=list(state_mlp),
                           wrench_mlp=list(wrench_mlp), wrench_clip_pct=wrench_clip_pct, rgb_crop=rgb_crop,
                           down_dims=list(down_dims), kernel_size=kernel_size,
                           n_groups=n_groups, dsed=dsed, num_train_timesteps=num_train_timesteps,
                           num_inference_steps=num_inference_steps, prediction_type=prediction_type,
                           relative_pos=relative_pos)
        assert horizon % 4 == 0, "UNet with 3 levels down/up-samples twice"
        assert n_obs_steps - 1 + n_action_steps <= horizon
        self.visual_key, self.relative_pos = visual, relative_pos
        self.horizon, self.n_obs_steps, self.n_action_steps = horizon, n_obs_steps, n_action_steps
        self.num_inference_steps, self.prediction_type = num_inference_steps, prediction_type

        enc_cfg = (dict(kind="pc", channels=channels, out_dim=enc_out_dim) if visual == "pc"
                   else dict(kind="rgb", out_dim=enc_out_dim, crop=rgb_crop))
        self.encoder = build_encoder(enc_cfg)
        self.agent_pos_mlp = mlp(agent_pos_dim, state_mlp)
        self.wrench_mlp = mlp(wrench_dim * n_obs_steps, wrench_mlp)
        self.cond_dims = {"visual": enc_out_dim * n_obs_steps, "agent_pos": state_mlp[-1] * n_obs_steps,
                          "wrench": wrench_mlp[-1]}
        self.norm = nn.ModuleDict({"action": MinMax(action_dim), "agent_pos": MinMax(agent_pos_dim),
                                   "wrench": MinMax(wrench_dim, pct=wrench_clip_pct)})

        self.unet = ConditionalUnet1D(
            input_dim=action_dim, global_cond_dim=sum(self.cond_dims.values()),
            diffusion_step_embed_dim=dsed, down_dims=list(down_dims), kernel_size=kernel_size,
            n_groups=n_groups, condition_type="film")
        sched = dict(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2",
                     clip_sample=True, prediction_type=prediction_type)
        self.train_scheduler = DDPMScheduler(**sched)
        self.infer_scheduler = DDIMScheduler(**sched, set_alpha_to_one=True, steps_offset=0)

    @torch.no_grad()
    def fit_normalizers(self, data, windows, max_visual=4096, seed=0):
        """Fit every normaliser on exactly what the network sees: the model-frame quantities of the
        training windows (all windows for low-dim, a fixed subsample for the point cloud)."""
        To = self.n_obs_steps
        obs = {k: torch.from_numpy(data[k][windows[:, :To]]) for k in ("agent_pos", "wrench")}
        self.norm["agent_pos"].fit(self.relative_obs(obs)["agent_pos"])
        self.norm["wrench"].fit(obs["wrench"])
        self.norm["action"].fit(self.action_to_model(torch.from_numpy(data["action"][windows]), obs["agent_pos"]))
        if self.visual_key == "pc":  # RGBEncoder has nothing to fit
            sub = np.random.default_rng(seed).choice(len(windows), min(max_visual, len(windows)), replace=False)
            v = {"pc": torch.from_numpy(data["pc"][windows[sub, :To]]), "agent_pos": obs["agent_pos"][sub]}
            self.encoder.fit(self.relative_obs(v)["pc"].numpy())

    def ref(self, agent_pos):
        """(B, To, 9) raw -> (B, 3): last observed pin tip (reference point)."""
        return agent_pos[:, -1, :3]

    def relative_obs(self, obs):
        """Raw obs -> model frame: pc xyz and agent_pos positions minus ref (identity if not relative_pos)."""
        if not self.relative_pos:
            return obs
        ref = self.ref(obs["agent_pos"])
        out = dict(obs, agent_pos=to_reference_frame(obs["agent_pos"], ref[:, None]))
        if "pc" in obs:
            out["pc"] = to_reference_frame(obs["pc"], ref[:, None, None])
        return out

    def action_to_model(self, action, agent_pos):
        """Raw absolute actions (B, H, 9) -> model frame (position minus ref), NOT yet normalised."""
        return to_reference_frame(action, self.ref(agent_pos)[:, None]) if self.relative_pos else action

    def action_from_model(self, action, agent_pos):
        """Model-frame actions (B, H, 9), unnormalised -> absolute."""
        return to_reference_frame(action, -self.ref(agent_pos)[:, None]) if self.relative_pos else action

    def normalized_target(self, action, agent_pos):
        return self.norm["action"].normalize(self.action_to_model(action, agent_pos))

    def cond_parts(self, obs):
        """obs: {visual_key: (B, To, ...), 'agent_pos': (B, To, 9), 'wrench': (B, To, 6)}, raw.
        Returns the named embeddings {name: (B, dim)}."""
        obs = self.relative_obs(obs)
        v = obs[self.visual_key]
        B, To = v.shape[:2]
        return {
            "visual": self.encoder(v.flatten(0, 1)).reshape(B, -1),
            "agent_pos": self.agent_pos_mlp(self.norm["agent_pos"].normalize(obs["agent_pos"])).reshape(B, -1),
            "wrench": self.wrench_mlp(self.norm["wrench"].normalize(obs["wrench"]).reshape(B, -1)),
        }

    def global_cond(self, obs):
        parts = self.cond_parts(obs)
        return torch.cat([parts[k] for k in COND_ORDER], dim=-1)

    def obs_keys(self):
        return (self.visual_key, "agent_pos", "wrench")

    def compute_loss(self, batch):
        cond = self.global_cond({k: batch[k] for k in self.obs_keys()})
        x0 = self.normalized_target(batch["action"], batch["agent_pos"])
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.train_scheduler.config.num_train_timesteps, (x0.shape[0],)).to(x0.device)
        pred = self.unet(self.train_scheduler.add_noise(x0, noise, t), t, global_cond=cond)
        return F.mse_loss(pred, x0 if self.prediction_type == "sample" else noise)

    @torch.no_grad()
    def sample_chunk(self, obs, noise=None, normalized=False):
        """obs: raw (B, To, ...) tensors. noise: optional (B, horizon, action_dim) initial x_T
        (same noise + DDIM eta=0 => deterministic). Returns (B, horizon, action_dim): ABSOLUTE poses,
        or with normalized=True the network's normalised model-frame output."""
        cond = self.global_cond(obs)
        shape = (cond.shape[0], self.horizon, self.norm["action"].scale.numel())
        x = (torch.randn(shape) if noise is None else noise.clone()).to(cond.device)
        sched = self.infer_scheduler
        sched.set_timesteps(self.num_inference_steps)
        for t in sched.timesteps:
            x = sched.step(self.unet(x, t, global_cond=cond), t, x, eta=0.0).prev_sample
        return x if normalized else self.action_from_model(self.norm["action"].unnormalize(x), obs["agent_pos"])

    def executed(self, chunk):
        """The n_action_steps actions that are executed: indices To-1 .. To-1+Ta-1."""
        return chunk[..., self.n_obs_steps - 1: self.n_obs_steps - 1 + self.n_action_steps, :]

    @torch.no_grad()
    def predict(self, obs_history, noise=None):
        """obs_history: {visual_key: (T, ...), 'agent_pos': (T, 9), 'wrench': (T, 6)}, oldest
        first, T >= 1, raw (pc float (T, N, 14) / rgb uint8 (T, 96, 96, 3)). Uses the last
        n_obs_steps frames; at episode start the first frame is repeated (training padding).
        Returns np (n_action_steps, 9): absolute target poses (pos + raw 6D; orthonormalise
        downstream)."""
        To = self.n_obs_steps
        obs = {}
        for k in self.obs_keys():
            x = torch.as_tensor(np.asarray(obs_history[k]))[-To:]
            if x.dtype == torch.float64:
                x = x.float()
            if x.shape[0] < To:
                x = torch.cat([x[:1].expand(To - x.shape[0], *x.shape[1:]), x])
            obs[k] = x[None]
        return self.executed(self.sample_chunk(obs, noise)[0]).numpy()


def load_policy(run_dir):
    """Load runs/<name>/policy.pt and put the ACCEPTED encoder.pt weights into it, so the
    policy used in eval is exactly the checkpoint that check_encoder verified."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=True)
    enc = torch.load(run_dir / "encoder.pt", map_location="cpu", weights_only=True)
    assert ckpt["config_hash"] == enc["config_hash"], "policy.pt and encoder.pt come from different runs"
    model = DiffusionPolicy(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.encoder.load_state_dict(enc["state_dict"])
    model.config_hash = ckpt["config_hash"]
    return model.eval()
