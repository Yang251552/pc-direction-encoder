"""D4: merge eval.run output files into one results file, keeping each input's attempts and
metadata verbatim, plus a (policy, ablate type, group) success/attempts summary.

  .venv/bin/python -m eval.merge results/pc.json results/send_pc_ablate_pc_f3.json --out results/results.json
  .venv/bin/python -m eval.merge --selftest

Errors (exit 1) if inputs disagree on protocol_sha256, or if the same (policy, ablate type)
attempts the same case_id twice (across files or within one).
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/results.json"


class MergeError(Exception):
    pass


def ablate_type(rec):
    flags = [name for name, key in (("pc", "ablate_pc"), ("wrench", "ablate_wrench")) if rec.get(key)]
    return "+".join(flags) or "none"


def check_protocol_consistency(records, sources):
    protos = {r["protocol_sha256"] for r in records}
    if len(protos) > 1:
        detail = ", ".join(f"{s}={r['protocol_sha256']}" for r, s in zip(records, sources))
        raise MergeError(f"protocol_sha256 mismatch across inputs: {detail}")
    return protos.pop()


def check_no_duplicate_case_ids(records, sources):
    seen = {}  # (policy, ablate_type, case_id) -> source file
    for rec, src in zip(records, sources):
        key_prefix = (rec["policy"], ablate_type(rec))
        for att in rec["attempts"]:
            key = key_prefix + (att["case_id"],)
            if key in seen:
                raise MergeError(f"duplicate case_id {att['case_id']!r} for policy={key_prefix[0]} "
                                  f"ablate={key_prefix[1]}: already seen in {seen[key]}, again in {src}")
            seen[key] = src


def build_summary(records):
    """{policy: {ablate_type: {group: {success, attempts}}}}"""
    summary = {}
    for rec in records:
        by_ablate = summary.setdefault(rec["policy"], {}).setdefault(ablate_type(rec), {})
        for att in rec["attempts"]:
            g = by_ablate.setdefault(att["group"], {"success": 0, "attempts": 0})
            g["attempts"] += 1
            g["success"] += int(att["success"])
    return summary


def merge(records, sources):
    protocol_sha256 = check_protocol_consistency(records, sources)
    check_no_duplicate_case_ids(records, sources)
    return {
        "protocol_sha256": protocol_sha256,
        "summary_by_policy_ablate_group": build_summary(records),
        "inputs": [{"source_file": src, **rec} for rec, src in zip(records, sources)],
    }


def _run_cli(args):
    """Invoke this module's own CLI as a subprocess (used by --selftest)."""
    import subprocess
    return subprocess.run([sys.executable, "-m", "eval.merge", *args], cwd=ROOT,
                           capture_output=True, text=True)


def selftest():
    import tempfile

    base = json.loads((ROOT / "results/send_pc_train_control.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        main_p = td / "main.json"
        main_p.write_text(json.dumps(base))

        # --- positive: base + a disjoint ablation file (different case_ids, different ablate flag)
        abl_attempts = [dict(a, case_id=a["case_id"] + "-abl") for a in base["attempts"][:2]]
        abl = dict(base, ablate_pc=True, attempts=abl_attempts,
                   summary={"train_control": {"success": sum(a["success"] for a in abl_attempts),
                                               "attempts": len(abl_attempts)}})
        abl_p = td / "abl.json"
        abl_p.write_text(json.dumps(abl))
        out_p = td / "out.json"
        r = _run_cli([str(main_p), str(abl_p), "--out", str(out_p)])
        assert r.returncode == 0, f"positive case should succeed, got: {r.stderr}"
        merged = json.loads(out_p.read_text())
        sbag = merged["summary_by_policy_ablate_group"]["send_pc"]
        assert sbag["none"]["train_control"] == base["summary"]["train_control"], sbag["none"]
        assert sbag["pc"]["train_control"]["attempts"] == 2, sbag["pc"]
        assert len(merged["inputs"]) == 2
        print(f"PASS positive: merged {main_p.name}+{abl_p.name} -> {out_p.name}, "
              f"none/train_control={sbag['none']['train_control']}")

        # --- negative 1: duplicate case_id (merge main with itself under the same ablate type)
        r = _run_cli([str(main_p), str(main_p), "--out", str(td / "dup.json")])
        assert r.returncode != 0, "expected failure for duplicate case_id"
        assert "duplicate case_id" in r.stderr, r.stderr
        print("PASS negative: duplicate case_id rejected ->", r.stderr.strip().splitlines()[-1])

        # --- negative 2: protocol_sha256 mismatch (hash inconsistency)
        bad_p = td / "bad_protocol.json"
        bad_p.write_text(json.dumps(dict(base, protocol_sha256="deadbeef00000000")))
        r = _run_cli([str(main_p), str(bad_p), "--out", str(td / "badmerge.json")])
        assert r.returncode != 0, "expected failure for protocol_sha256 mismatch"
        assert "protocol_sha256 mismatch" in r.stderr, r.stderr
        print("PASS negative: protocol_sha256 mismatch rejected ->", r.stderr.strip().splitlines()[-1])

    print("eval.merge selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="eval.run output JSON files")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    if not a.inputs:
        ap.error("the following arguments are required: inputs")

    records = [json.loads(Path(p).read_text()) for p in a.inputs]
    try:
        merged = merge(records, a.inputs)
    except MergeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=1))
    print(f"merged {len(records)} file(s) -> {out_path}")


if __name__ == "__main__":
    main()
