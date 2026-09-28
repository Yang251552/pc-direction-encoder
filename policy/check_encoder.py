"""B6 / C2: load ONLY the encoder from encoder.pt in a fresh process and check it is a trained model.

  .venv/bin/python -m policy.check_encoder runs/<name>/encoder.pt

PASS needs: finite (B, out_dim) embedding on a contract-shaped input; ||theta - theta_init|| > 0
(theta_init = this run's encoder right before its first gradient step, stored in encoder.pt);
loss.csv next to it whose last step equals the stored train_steps; config_hash consistent with
config.json (recomputed from its config + data sha256).
"""
import csv
import json
import sys
from pathlib import Path

import torch

from policy.encoder import load_encoder


def main(path):
    path = Path(path)
    enc, ckpt = load_encoder(path)
    torch.manual_seed(0)
    cfg = ckpt["config"]
    x = (torch.randn(2, 256, 14) if cfg["kind"] == "pc" else torch.randint(0, 256, (2, 96, 96, 3), dtype=torch.uint8))
    with torch.no_grad():
        emb = enc(x)
    finite = bool(torch.isfinite(emb).all())
    init = ckpt["init_state_dict"]
    diff = sum(((v.float() - init[k].float()) ** 2).sum() for k, v in ckpt["state_dict"].items()) ** 0.5
    ref = sum((v.float() ** 2).sum() for v in init.values()) ** 0.5

    loss_csv = path.parent / "loss.csv"
    rows = list(csv.DictReader(loss_csv.open())) if loss_csv.exists() else []
    last_step = int(rows[-1]["step"]) if rows else None

    hash_ok, cfg_json = None, path.parent / "config.json"
    if cfg_json.exists():
        from policy.train import config_hash  # only for the consistency check
        info = json.loads(cfg_json.read_text())
        hash_ok = info["config_hash"] == ckpt["config_hash"] == config_hash(info["config"], info["data_sha256"])

    out = dict(kind=cfg["kind"], xyz_frame=ckpt.get("xyz_frame", "world"), emb_shape=list(emb.shape), finite=finite, param_diff_norm=round(float(diff), 4),
               rel_diff=round(float(diff / ref), 4), train_steps=ckpt["train_steps"], loss_csv=str(loss_csv),
               loss_csv_last_step=last_step, loss_first=float(rows[0]["loss"]) if rows else None,
               loss_last=float(rows[-1]["loss"]) if rows else None, config_hash=ckpt["config_hash"][:16],
               config_hash_consistent=hash_ok)
    ok = (finite and list(emb.shape) == [2, cfg["out_dim"]] and diff > 0 and last_step == ckpt["train_steps"]
          and hash_ok is True)
    print(("PASS " if ok else "FAIL ") + json.dumps(out))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
