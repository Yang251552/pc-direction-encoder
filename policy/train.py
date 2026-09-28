"""Train the diffusion policy; the product is runs/<name>/encoder.pt.

  .venv/bin/python -m policy.train --config configs/send_pc.json     # point cloud (send version)
  .venv/bin/python -m policy.train --config configs/send_rgb.json    # RGB-only baseline (f5)
  overrides (checks / smoke runs): --data PATH --name NAME --steps N --episodes K
  --device auto|cpu|cuda (default auto = cuda if available). Everything is saved as CPU tensors, so
  runs trained on a GPU load locally with map_location="cpu".

Outputs in runs/<name>/:
  encoder.pt  construct config + EMA state_dict + init state_dict + config_hash + train steps
  policy.pt   model config + EMA state_dict + config_hash
  loss.csv    step, mean loss / mean encoder grad norm since the previous row, lr, elapsed
  config.json full config + config_hash + data sha256 + result (incl. train_seconds)
config_hash = sha256(json(model + train sections without the runtime-only keys threads / log_every /
device, sorted keys) + sha256(data file)). (Before C3 repair r2 threads and log_every were hashed too.)
"""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from policy.model import DiffusionPolicy

ROOT = Path(__file__).resolve().parents[1]
LOW_DIM = {"agent_pos": 9, "wrench": 6, "action": 9}


def load_data(path, visual, n_episodes=None):
    """Read data/demos.npz for one visual variant and check the contract (SPEC 5).
    Trust boundary: another stage writes this file."""
    with np.load(path) as d:
        missing = {visual, "episode_ends", *LOW_DIM} - set(d.files)
        assert not missing, f"{path}: missing keys {sorted(missing)}"
        data = {k: np.asarray(d[k], dtype=np.float32) for k in LOW_DIM}
        data[visual] = np.asarray(d[visual])
        ends = np.asarray(d["episode_ends"], dtype=np.int64)
    S = data["action"].shape[0]
    for k, dim in LOW_DIM.items():
        assert data[k].shape == (S, dim), f"{k} {data[k].shape}, want ({S}, {dim})"
        assert np.isfinite(data[k]).all(), f"non-finite values in {k}"
    v = data[visual]
    if visual == "pc":
        assert v.ndim == 3 and v.shape[0] == S and v.shape[2] == 14, f"pc {v.shape}, want ({S}, N, 14)"
        v = data["pc"] = v.astype(np.float32, copy=False)
        assert np.isfinite(v).all(), "non-finite values in pc"
    else:
        assert v.ndim == 4 and v.shape[0] == S and v.shape[3] == 3 and v.dtype == np.uint8, \
            f"rgb {v.shape} {v.dtype}, want ({S}, H, W, 3) uint8"
    assert ends.ndim == 1 and len(ends) and ends[-1] == S and np.all(np.diff(np.r_[0, ends]) > 0), \
        "bad episode_ends"
    data["episode_ends"] = ends
    if n_episodes is not None and n_episodes < len(ends):
        S = int(ends[n_episodes - 1])
        data = {k: (x[:n_episodes] if k == "episode_ends" else x[:S]) for k, x in data.items()}
    return data


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


RUNTIME_KEYS = ("threads", "log_every", "device")  # how a run executes, not what it learns


def config_hash(cfg, data_sha):
    train = {k: v for k, v in cfg["train"].items() if k not in RUNTIME_KEYS}
    core = json.dumps({"model": cfg["model"], "train": train}, sort_keys=True)
    return hashlib.sha256((core + data_sha).encode()).hexdigest()


def make_windows(episode_ends, horizon, pad_before, pad_after):
    """(M, horizon) global frame indices + (M,) episode id. Same windows and edge padding as
    DP3's SequenceSampler: starts run from -pad_before to L - horizon + pad_after; frames
    outside the episode repeat its first/last frame."""
    rows, eps, start = [], [], 0
    for e, end in enumerate(episode_ends):
        L = end - start
        for i in range(-pad_before, L - horizon + pad_after + 1):
            rows.append(np.clip(start + i + np.arange(horizon), start, end - 1))
            eps.append(e)
        start = end
    return np.asarray(rows, dtype=np.int64), np.asarray(eps, dtype=np.int64)


def windows_for(data, mcfg):
    return make_windows(data["episode_ends"], mcfg["horizon"],
                        pad_before=mcfg["n_obs_steps"] - 1, pad_after=mcfg["n_action_steps"] - 1)


def get_batch(data, visual, windows, rows, To, visual_rows=None):
    idx = windows[rows]
    vidx = idx if visual_rows is None else windows[visual_rows]
    return {visual: torch.from_numpy(data[visual][vidx[:, :To]]),
            "agent_pos": torch.from_numpy(data["agent_pos"][idx[:, :To]]),
            "wrench": torch.from_numpy(data["wrench"][idx[:, :To]]),
            "action": torch.from_numpy(data["action"][idx])}


def other_episode_rows(rng, win_ep, rows):
    """For each sample, a random window from a DIFFERENT episode (inference-time swap controls)."""
    v = rng.integers(0, len(win_ep), len(rows))
    while (bad := win_ep[v] == win_ep[rows]).any():
        v[bad] = rng.integers(0, len(win_ep), int(bad.sum()))
    return v


def grad_norm(module):
    g = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(g).norm().item() if g else 0.0


def resolve_device(device="auto"):
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                        "cpu" if device == "auto" else device)


def cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def train(cfg, data, out_dir, data_sha, quiet=False, device="auto"):
    """cfg: {'model': DiffusionPolicy kwargs, 'train': {...}}. Returns summary + models.
    Effective warmup = min(train.warmup, steps // 10) (recorded as result.warmup_steps).
    Normalisers are fitted on the CPU, then model (incl. normaliser buffers), batches and EMA live on
    `device`; saved state_dicts are CPU tensors."""
    device = resolve_device(device)
    mcfg, tcfg = cfg["model"], cfg["train"]
    steps, bs = int(tcfg["steps"]), int(tcfg["batch_size"])
    torch.set_num_threads(int(tcfg["threads"]))
    torch.manual_seed(tcfg["seed"])
    rng = np.random.default_rng(tcfg["seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chash = config_hash(cfg, data_sha)

    model = DiffusionPolicy(**mcfg)
    windows, _ = windows_for(data, model.config)
    model.fit_normalizers(data, windows)
    enc_init = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    model.to(device)
    visual, To = model.visual_key, model.n_obs_steps
    warmup = min(int(tcfg["warmup"]), steps // 10)  # short runs keep >= 90% of steps after warmup

    # DP3 dp3.yaml optimiser / schedule / EMA
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    lr_sched = get_scheduler("cosine", opt, num_warmup_steps=warmup, num_training_steps=steps)
    ema = EMAModel(model.parameters(), decay=0.9999, min_decay=0.0, use_ema_warmup=True, inv_gamma=1.0, power=0.75)

    log_f = open(out_dir / "loss.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "loss", "enc_grad_norm", "lr", "elapsed_s"])
    losses, gnorms, step_s, last = [], [], [], 0
    log_every = int(tcfg.get("log_every", 100))
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        ts = time.perf_counter()
        rows = rng.integers(0, len(windows), bs)
        batch = {k: v.to(device, non_blocking=True) for k, v in get_batch(data, visual, windows, rows, To).items()}
        loss = model.compute_loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms.append(grad_norm(model.encoder))
        opt.step()
        lr_sched.step()
        ema.step(model.parameters())
        losses.append(loss.item())
        step_s.append(time.perf_counter() - ts)
        if step % log_every == 0 or step == steps or step == 1:
            row = [step, float(np.mean(losses[last:])), float(np.mean(gnorms[last:])), lr_sched.get_last_lr()[0],
                   round(time.time() - t0, 1)]
            last = step
            log.writerow(row)
            log_f.flush()
            if not quiet:
                print("step %d loss %.5f enc_grad %.4g lr %.2e %.0fs" % tuple(row), flush=True)
    train_s = time.time() - t0
    log_f.close()

    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model.parameters())
    ema_model.eval()
    n_frames = int(data["episode_ends"][-1])
    summary = dict(steps=steps, batch_size=bs, train_seconds=round(train_s, 1),
                   sec_per_step=round(train_s / steps, 4), first_loss=losses[0],
                   final_loss_mean50=float(np.mean(losses[-50:])), n_windows=len(windows),
                   n_episodes=len(data["episode_ends"]), n_frames=n_frames,
                   epochs=round(steps * bs / len(windows), 1), warmup_steps=warmup,
                   threads=int(tcfg["threads"]), device=str(device))
    torch.save(dict(config=model.encoder.config, state_dict=cpu_state(ema_model.encoder),
                    init_state_dict=enc_init, weights="ema", config_hash=chash, train_steps=steps,
                    xyz_frame="pin_tip" if model.relative_pos else "world", run=out_dir.name),
               out_dir / "encoder.pt")
    torch.save(dict(config=model.config, state_dict=cpu_state(ema_model), weights="ema",
                    config_hash=chash), out_dir / "policy.pt")
    (out_dir / "config.json").write_text(json.dumps(dict(
        config=cfg, config_hash=chash, data_sha256=data_sha, result=summary), indent=2))
    return dict(summary, config_hash=chash, model=model, ema_model=ema_model, losses=losses,
                gnorms=gnorms, step_s=step_s, windows=windows, out_dir=out_dir)


def load_config(path, **overrides):
    """Config JSON + CLI/check overrides (data, name, steps). steps must be set (B5 fills it)."""
    cfg = json.loads(Path(path).read_text())
    for k in ("data", "name"):
        if overrides.get(k):
            cfg[k] = str(overrides[k])
    if overrides.get("steps"):
        cfg["train"]["steps"] = int(overrides["steps"])
    assert cfg["train"].get("steps"), f"{path}: train.steps not set (run policy.checks b5 first)"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data")
    ap.add_argument("--name")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--episodes", type=int, help="use only the first k episodes")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    cfg = load_config(a.config, data=a.data, name=a.name, steps=a.steps)
    data_path = ROOT / cfg["data"]
    data = load_data(data_path, cfg["model"]["visual"], a.episodes)
    if a.episodes:
        cfg["train"]["episodes"] = a.episodes  # enters config_hash
    r = train(cfg, data, ROOT / "runs" / cfg["name"], file_sha256(data_path), device=a.device)
    print(json.dumps({k: v for k, v in r.items() if isinstance(v, (int, float, str, bool))}))
    print("DONE")


if __name__ == "__main__":
    main()
