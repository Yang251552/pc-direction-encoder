"""Phase B acceptance checks (PLAN v2.1, B1-B7). Each prints PASS/FAIL and its key numbers.

  .venv/bin/python -m policy.checks b1        # ... b7, in order (b6/b7 use the b3/b4 runs)
  .venv/bin/python -m policy.checks c1 --data data/demos.npz   # b3's checks on real demos

Thresholds come from eval/protocol.json 'thresholds' (criteria as amended 09-28, commit 4f26ca8):
  overfit        : first 5 episodes, `overfit_steps` steps; mean loss of the last 50 steps
                   <= overfit_loss_ratio x mean of the first 10
  encoder grad   : encoder gradient norm > 0 at every step (overfit run + full run)
  inference swap : full send-config training (train.steps, set by b5) on all episodes; the trained (EMA)
                   policy's diffusion loss on 1024 fixed training windows (fixed noise and t) with each
                   window's point cloud replaced by one from another episode
                   >= inference_swap_loss_ratio x the loss with its own point cloud
  sensitivity    : same trained policy; 64 fixed training windows, one fixed initial noise, DDIM eta = 0;
                   mean |delta a| over the executed (n_action_steps, 9) actions in normalised action space
                   >= sensitivity_mean_abs_da, (a) raw wrench zeroed, (b) direction channels
                   normal / pcurv_dir / pcurv_mag (3:11) replaced, point by point index, by those of a
                   frame from another episode (in-distribution; zeroing was dropped)
Synthetic data: runs/synthetic/demos_synth.npz (policy/make_synthetic.py, seed 0, 100 episodes).
"""
import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from policy.encoder import DIRECTION_CHANNELS, PointCloudEncoder
from policy.model import DiffusionPolicy, load_policy
from policy.train import (file_sha256, get_batch, load_config, load_data, make_windows, other_episode_rows,
                          train, windows_for)

ROOT = Path(__file__).resolve().parents[1]
TH = json.loads((ROOT / "eval" / "protocol.json").read_text())["thresholds"]
SYNTH = ROOT / "runs" / "synthetic" / "demos_synth.npz"
DP3 = ROOT / "third_party" / "dp3"
CFG = {"pc": ROOT / "configs" / "send_pc.json", "rgb": ROOT / "configs" / "send_rgb.json"}
THREADS = 2  # another agent runs the simulator in parallel
OVERFIT_EPISODES = 5  # PLAN C1: overfit on 5 demos


def report(name, ok, **nums):
    print(f"{name}: {'PASS' if ok else 'FAIL'} {json.dumps(nums)}", flush=True)
    return bool(ok)


def synth_path():
    if not SYNTH.exists():
        from policy.make_synthetic import make
        SYNTH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(SYNTH, **make(100, seed=0))
    return SYNTH


# ---------------------------------------------------------------- B1: vendoring + encoder shapes
def parse_source_md():
    md = (DP3 / "SOURCE.md").read_text()
    table = {m[0]: m[1] for m in re.findall(r"^\| (\S+) \| \S+ \| ([0-9a-f]{64}) \|$", md, re.M)}
    diff = md.split("```diff\n", 1)[1].split("```", 1)[0]
    return table, diff


def reverse_apply(text, file_diff):
    """Undo one file's unified diff: vendored text -> upstream text (hunks must match exactly)."""
    lines, out, i = text.splitlines(True), [], 0
    for h in re.split(r"^(?=@@ )", file_diff, flags=re.M)[1:]:
        head, *body = h.splitlines(True)
        start = int(re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", head)[1]) - 1
        new = [ln[1:] for ln in body if ln[0] in " +"]
        old = [ln[1:] for ln in body if ln[0] in " -"]
        assert lines[start:start + len(new)] == new, f"hunk {head.strip()} does not match the vendored file"
        out += lines[i:start] + old
        i = start + len(new)
    return "".join(out + lines[i:])


def b1():
    table, diff = parse_source_md()
    per_file = {m[0]: m[1] for m in re.findall(r"^--- a/(\S+)\n\+\+\+ b/\S+\n(.*?)(?=^--- a/|\Z)", diff, re.M | re.S)}
    vendored = {p.name for p in DP3.iterdir() if p.is_file() and p.name != "SOURCE.md"}
    rebuilt_ok = {}
    for f, sha in table.items():
        up = reverse_apply((DP3 / f).read_text(), per_file.get(f, ""))
        rebuilt_ok[f] = hashlib.sha256(up.encode()).hexdigest() == sha
    orig = DP3 / "_orig"
    diff_equal = None
    if orig.exists():  # before _orig is deleted: the registered diff must be exactly the current diff
        cur = "".join(line for f in table for line in difflib.unified_diff(
            (orig / f).read_text().splitlines(True), (DP3 / f).read_text().splitlines(True), "a/" + f, "b/" + f, n=1))
        diff_equal = cur == diff
    shapes = {}
    for C in (3, 6, 14):
        enc = PointCloudEncoder(channels=C)
        shapes[C] = list(enc(torch.randn(4, 256, 14)).shape)
    ok = (vendored == set(table) and all(rebuilt_ok.values()) and diff_equal is not False
          and all(s == [4, 64] for s in shapes.values()))
    return report("B1", ok, files_registered=sorted(table), unregistered=sorted(vendored - set(table)),
                  upstream_rebuilt_sha_ok=rebuilt_ok, n_hunks=diff.count("\n@@ "), orig_present=orig.exists(),
                  registered_diff_equals_current=diff_equal, encoder_out={f"C={c}": s for c, s in shapes.items()})


# ---------------------------------------------------------------- B2: named condition -> UNet
def b2():
    torch.manual_seed(0)
    B, out = 4, {}
    obs = {"pc": torch.randn(B, 2, 256, 14), "rgb": torch.randint(0, 256, (B, 2, 96, 96, 3), dtype=torch.uint8),
           "agent_pos": torch.randn(B, 2, 9), "wrench": torch.randn(B, 2, 6)}
    ok = True
    for vis in ("pc", "rgb"):
        m = DiffusionPolicy(**load_config(CFG[vis], steps=1)["model"])
        parts = m.cond_parts({k: obs[k] for k in m.obs_keys()})
        cond = m.global_cond({k: obs[k] for k in m.obs_keys()})
        y = m.unet(torch.randn(B, m.horizon, 9), torch.zeros(B, dtype=torch.long), global_cond=cond)
        out[vis] = dict(encoder=type(m.encoder).__name__, parts={k: list(v.shape) for k, v in parts.items()},
                        global_cond=list(cond.shape), unet_out=list(y.shape),
                        params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
        ok &= (list(parts) == ["visual", "agent_pos", "wrench"] and list(y.shape) == [B, 8, 9]
               and cond.shape[1] == sum(v.shape[1] for v in parts.values()))
    if out["rgb"]["encoder"] == "RGBEncoder":
        m = DiffusionPolicy(**load_config(CFG["rgb"], steps=1)["model"])
        gn = sum(isinstance(x, torch.nn.GroupNorm) for x in m.encoder.modules())
        bn = sum(isinstance(x, torch.nn.BatchNorm2d) for x in m.encoder.modules())
        out["rgb"].update(groupnorm_layers=gn, batchnorm_layers=bn)
        ok &= bn == 0 and gn > 0
    return report("B2", ok, **out)


# ---------------------------------------------------------------- B3 / B4: training checks
def run(vis, name, data, data_path, steps):
    cfg = load_config(CFG[vis], steps=steps, name=name, data=data_path)
    cfg["train"].update(threads=THREADS, log_every=10)
    t = time.time()
    r = train(cfg, data, ROOT / "runs" / name, file_sha256(data_path), quiet=True)
    print(f"  trained {name}: {r['steps']} steps in {time.time() - t:.0f}s", flush=True)
    return r


def overfit_numbers(losses):
    n = TH["overfit_steps"]
    first, last = float(np.mean(losses[:10])), float(np.mean(losses[n - 50:n]))
    return first, last, last / first


@torch.no_grad()
def swap_loss(model, data, n=1024, seed=0, chunk=128):
    """Diffusion loss on n fixed training windows (fixed noise and t ~ U[0, T)): own visual
    observation vs the visual observation of a window from another episode."""
    windows, win_ep = windows_for(data, model.config)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(windows), min(n, len(windows)), replace=False)
    vrows = other_episode_rows(rng, win_ep, rows)
    g = torch.Generator().manual_seed(seed)
    T = model.train_scheduler.config.num_train_timesteps
    own = swapped = 0.0
    for i in range(0, len(rows), chunk):
        r, v = rows[i:i + chunk], vrows[i:i + chunk]
        b = get_batch(data, model.visual_key, windows, r, model.n_obs_steps)
        x0 = model.normalized_target(b["action"], b["agent_pos"])
        noise, t = torch.randn(x0.shape, generator=g), torch.randint(0, T, (len(r),), generator=g)
        xt = model.train_scheduler.add_noise(x0, noise, t)
        target = x0 if model.prediction_type == "sample" else noise
        for vis_rows, acc in ((None, "own"), (v, "swapped")):
            bb = get_batch(data, model.visual_key, windows, r, model.n_obs_steps, vis_rows)
            cond = model.global_cond({k: bb[k] for k in model.obs_keys()})
            sse = float(((model.unet(xt, t, global_cond=cond) - target) ** 2).mean(dim=(1, 2)).sum())
            if acc == "own":
                own += sse
            else:
                swapped += sse
    return own / len(rows), swapped / len(rows)


def sensitivity(model, data, n=128, seed=0):
    """Physical-unit sensitivity (criterion amended 09-28, user-approved; the old normalised 9-D |delta a|
    was mis-scaled: 0.05 normalised = 4-8 mm, far above the 1 mm clearance). Mean |delta p| in mm of the
    executed positions, one fixed initial noise, DDIM eta = 0:
      swap_direction_mm : random windows; direction channels (normal, pcurv_dir, pcurv_mag) replaced point
                          by point with a frame from another episode (in-distribution)
      zero_wrench_mm    : contact windows only (|F| > sensitivity_contact_force_n at the last observed
                          frame); raw wrench zeroed"""
    windows, win_ep = windows_for(data, model.config)
    rng = np.random.default_rng(seed)
    To = model.n_obs_steps
    F = np.linalg.norm(data["wrench"][:, :3], axis=1)[windows[:, To - 1]]
    out = {}

    def run(rows, perturb):
        idx = windows[rows][:, :To]
        obs = {k: torch.from_numpy(data[k][idx]) for k in model.obs_keys()}
        noise = torch.randn((len(rows), model.horizon, 9), generator=torch.Generator().manual_seed(seed))
        act = lambda o: model.executed(model.sample_chunk(o, noise))[..., :3]  # noqa: E731  raw metres
        base = act(obs)
        out["determinism_max_abs"] = max(out.get("determinism_max_abs", 0.0), float((act(obs) - base).abs().max()))
        return float((act(perturb(obs, rows)) - base).norm(dim=-1).mean() * 1000)

    def swap_dirs(obs, rows):
        other = windows[other_episode_rows(rng, win_ep, rows)][:, :To]
        pc = obs["pc"].clone()
        pc[..., DIRECTION_CHANNELS] = torch.from_numpy(data["pc"][other][..., DIRECTION_CHANNELS])
        return dict(obs, pc=pc)

    if "pc" in model.obs_keys():
        out["swap_direction_mm"] = run(rng.choice(len(windows), min(n, len(windows)), replace=False), swap_dirs)
    contact = np.flatnonzero(F > TH["sensitivity_contact_force_n"])
    out["contact_windows"] = int(len(contact))
    out["zero_wrench_mm"] = run(rng.choice(contact, min(n, len(contact)), replace=False),
                                lambda o, r: dict(o, wrench=torch.zeros_like(o["wrench"]))) if len(contact) else 0.0
    return out


def b3(data_path=None, overfit_episodes=OVERFIT_EPISODES, tag="B3", run_dir=None):
    """overfit + encoder grad on a small set; inference swap + sensitivity on the full run's policy
    (loaded from disk exactly as eval will load it)."""
    torch.set_num_threads(THREADS)
    data_path = Path(data_path or synth_path())
    small, full = load_data(data_path, "pc", overfit_episodes), load_data(data_path, "pc")
    steps = TH["overfit_steps"]
    pre = tag.lower()
    fit = run("pc", f"{pre}_pc_overfit", small, data_path, steps)
    if run_dir is None:
        normal = run("pc", f"{pre}_pc", full, data_path, None)
        policy = load_policy(ROOT / "runs" / f"{pre}_pc")
    else:  # C1 on the already-trained send run (no duplicate full training); grads/losses from its loss.csv
        import csv
        rows = list(csv.DictReader(open(Path(run_dir) / "loss.csv")))
        policy = load_policy(run_dir)
        normal = {"gnorms": [float(r["enc_grad_norm"]) for r in rows], "n_episodes": int(len(full["episode_ends"])),
                  "steps": int(rows[-1]["step"]), "final_loss_mean50": float(np.mean([float(r["loss"]) for r in rows[-5:]])),
                  "config_hash": policy.config_hash}
    first, last, ratio = overfit_numbers(fit["losses"])
    g = fit["gnorms"] + normal["gnorms"]
    own, swapped = swap_loss(policy, full)
    sens = sensitivity(policy, full)
    th = TH["sensitivity_dpos_mm"]
    oks = [report(f"{tag}.overfit", ratio <= TH["overfit_loss_ratio"], loss_first10=round(first, 5),
                  loss_last50=round(last, 5), ratio=round(ratio, 4), steps=steps, episodes=fit["n_episodes"],
                  windows=fit["n_windows"]),
           report(f"{tag}.encoder_grad", min(g) > 0, min=float(min(g)), mean_last50=float(np.mean(g[-50:]))),
           report(f"{tag}.inference_swap", swapped / own >= TH["inference_swap_loss_ratio"],
                  episodes=normal["n_episodes"], steps=normal["steps"], train_final_loss=round(normal["final_loss_mean50"], 5),
                  loss_own_pc=round(own, 5), loss_swapped_pc=round(swapped, 5), ratio=round(swapped / own, 2),
                  threshold=TH["inference_swap_loss_ratio"]),
           report(f"{tag}.sensitivity", sens["determinism_max_abs"] == 0 and sens["swap_direction_mm"] >= th
                  and sens["zero_wrench_mm"] >= th, **{k: round(v, 4) for k, v in sens.items()}, threshold_mm=th)]
    return report(tag, all(oks), config_hash=normal["config_hash"][:16], data=str(data_path))


def b4():
    torch.set_num_threads(THREADS)
    r = run("rgb", "b4_rgb", load_data(synth_path(), "rgb", OVERFIT_EPISODES), synth_path(), TH["overfit_steps"])
    first, last, ratio = overfit_numbers(r["losses"])
    return report("B4", ratio <= TH["overfit_loss_ratio"] and min(r["gnorms"]) > 0, loss_first10=round(first, 5),
                  loss_last50=round(last, 5), ratio=round(ratio, 4), steps=TH["overfit_steps"],
                  episodes=r["n_episodes"], enc_grad_min=float(min(r["gnorms"])), sec_per_step=r["sec_per_step"])


# ---------------------------------------------------------------- B5: timing -> steps in both configs
BUDGET_S, MARGIN, WARM, TIMED = 7200, 0.9, 3, 20
PROTOCOL_EPISODES, PROTOCOL_LEN = 120, (100, 150)  # 120 demos x ~100-150 steps (protocol.json 'demos')


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def b5():
    load_before, sec = uptime(), {}
    for vis in ("pc", "rgb"):
        data, sha = load_data(synth_path(), vis), file_sha256(synth_path())
        cfg = load_config(CFG[vis], steps=WARM + TIMED, name=f"b5_{vis}", data=SYNTH)
        cfg["train"]["threads"] = THREADS
        r = train(cfg, data, ROOT / "runs" / f"b5_{vis}", sha, quiet=True)
        sec[vis] = float(np.mean(r["step_s"][WARM:]))
    load_after = uptime()
    steps = int(BUDGET_S * MARGIN / max(sec.values()) // 500 * 500)
    hours = {v: round(steps * s / 3600, 3) for v, s in sec.items()}
    E, (lo, hi) = PROTOCOL_EPISODES, PROTOCOL_LEN
    windows = [len(make_windows(np.cumsum([L] * E), 8, 1, 3)[0]) for L in (lo, hi)]  # horizon 8, To 2, Ta 4
    timing = dict(measured=time.strftime("%Y-%m-%d %H:%M"), batch_size=64, threads=THREADS, warm_steps=WARM,
                  timed_steps=TIMED, sec_per_step={v: round(s, 4) for v, s in sec.items()}, budget_s=BUDGET_S,
                  margin=MARGIN, projected_hours=hours, protocol_windows=windows,
                  epochs=[round(steps * 64 / w, 1) for w in windows[::-1]], uptime_before=load_before,
                  uptime_after=load_after, note="measured while the simulator agent ran; re-measure when idle")
    for vis, p in CFG.items():
        cfg = json.loads(p.read_text())
        cfg["train"]["steps"] = steps
        cfg["timing"] = timing
        p.write_text(json.dumps(cfg, indent=2) + "\n")
    written = {v: json.loads(p.read_text())["train"]["steps"] for v, p in CFG.items()}
    ok = all(h <= BUDGET_S / 3600 for h in hours.values()) and len(set(written.values())) == 1 and steps > 0
    return report("B5", ok, steps=steps, steps_in_configs=written, **timing)


# ---------------------------------------------------------------- B6: stand-alone encoder (fresh process)
def b6(runs=("runs/b3_pc", "runs/b4_rgb")):
    oks = []
    for run in runs:
        p = subprocess.run([sys.executable, "-m", "policy.check_encoder", str(ROOT / run / "encoder.pt")],
                           capture_output=True, text=True, cwd=ROOT)
        print(f"  {run}: {p.stdout.strip()}{p.stderr.strip()[-500:]}")
        oks.append(p.returncode == 0 and p.stdout.startswith("PASS"))
    return report("B6", all(oks), runs=list(runs))


# ---------------------------------------------------------------- B7: predict interface + latency
def b7(n=20, pc_run="runs/b3_pc", data_path=None):
    """Gate: the point cloud policy (default: the B3 normal model). The RGB policy (runs/b4_rgb) is
    timed for information only (its latency sets D3's wall time, not a B7 criterion)."""
    torch.set_num_threads(THREADS)
    out, ok = {}, True
    for vis, run in (("pc", pc_run), ("rgb", "runs/b4_rgb")):
        data = load_data(data_path or synth_path(), vis, 1)
        policy = load_policy(ROOT / run)
        hist = {k: data[k][5:7] for k in policy.obs_keys()}  # T = 2 frames, raw numpy
        a = policy.predict(hist)
        a1 = policy.predict({k: v[:1] for k, v in hist.items()})  # episode start: T = 1, padded
        ms = []
        for _ in range(n):
            t = time.perf_counter()
            policy.predict(hist)
            ms.append(1000 * (time.perf_counter() - t))
        out[vis] = dict(shape=list(a.shape), shape_T1=list(a1.shape), finite=bool(np.isfinite(a).all()),
                        median_ms=round(float(np.median(ms)), 1), max_ms=round(max(ms), 1),
                        ddim_steps=policy.num_inference_steps)
        if vis == "pc":
            want = (policy.n_action_steps, 9)
            ok = a.shape == want and a1.shape == want and out[vis]["finite"] and np.median(ms) < 200
    return report("B7", ok, threads=THREADS, calls=n, load=uptime().split("averages:")[-1].strip(),
                  pc=out["pc"], rgb_info_only=out["rgb"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["b1", "b2", "b3", "b4", "b5", "b6", "b7", "c1", "sens"])
    ap.add_argument("--run", help="sens / b6 / b7: trained run dir to check instead of the B3 default (no retraining)")
    ap.add_argument("--data")
    ap.add_argument("--episodes", type=int)
    a = ap.parse_args()
    if a.check == "sens":
        torch.set_num_threads(THREADS)
        sens = sensitivity(load_policy(a.run), load_data(a.data or ROOT / "data" / "demos.npz", "pc"))
        th = TH["sensitivity_dpos_mm"]
        ok = report("SENS", sens["determinism_max_abs"] == 0 and sens["swap_direction_mm"] >= th
                    and sens["zero_wrench_mm"] >= th, run=a.run, **{k: round(v, 4) for k, v in sens.items()},
                    threshold_mm=th)
    elif a.check == "c1":
        ok = b3(a.data or ROOT / "data" / "demos.npz", a.episodes or OVERFIT_EPISODES, tag="C1", run_dir=a.run)
    elif a.check == "b3":
        ok = b3(a.data, a.episodes or OVERFIT_EPISODES)
    elif a.check == "b6" and a.run:
        ok = b6((a.run,))
    elif a.check == "b7" and a.run:
        ok = b7(pc_run=a.run, data_path=a.data)
    else:
        ok = globals()[a.check]()
    print("DONE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
