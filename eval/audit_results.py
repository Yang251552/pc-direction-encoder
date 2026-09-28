"""D4/E1 gate: audit a merged results file (eval.merge output) for coverage and provenance.
Prints PASS/FAIL and the specific reasons. Exit 0 on PASS, 1 on FAIL.

  .venv/bin/python -m eval.audit_results results/results.json
  .venv/bin/python -m eval.audit_results results/results.json --gif figures/f3_success.gif --gif-case f3-004
  .venv/bin/python -m eval.audit_results --selftest

Checks (non-ablation "main" inputs = those with ablate_pc=False and ablate_wrench=False):
  - every eval/protocol.json case appears exactly once across the main inputs' attempts
  - per-group attempt counts match the protocol
  - all main inputs share exactly one config_hash, and it matches weights/send_pc/config.json
  - all main inputs share exactly one weights_sha256, and it matches the real sha256 (first 16
    hex chars, same algorithm as eval/run.py) of weights/send_pc/{encoder,policy}.pt
  - with --gif: the file exists, and --gif-case is a success in the main results
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights/send_pc"
PROTOCOL_PATH = ROOT / "eval/protocol.json"
DEFAULT_RESULTS = ROOT / "results/results.json"


def ablate_type(rec):
    flags = [name for name, key in (("pc", "ablate_pc"), ("wrench", "ablate_wrench")) if rec.get(key)]
    return "+".join(flags) or "none"


def weights_sha256(weights_dir=WEIGHTS):
    return {f: hashlib.sha256((weights_dir / f).read_bytes()).hexdigest()[:16] for f in ("encoder.pt", "policy.pt")}


def audit(merged, protocol, weights_dir=WEIGHTS, gif=None, gif_case=None):
    fails = []

    def need(ok, msg):
        if not ok:
            fails.append(msg)

    main_inputs = [i for i in merged.get("inputs", []) if ablate_type(i) == "none"]
    need(len(main_inputs) > 0, "no non-ablation ('main') inputs found in results")
    if not main_inputs:
        return fails

    attempts = [a for i in main_inputs for a in i["attempts"]]
    case_ids = [a["case_id"] for a in attempts]
    need(len(case_ids) == len(set(case_ids)), "a case_id appears more than once in the main results")

    want_ids = {c["case_id"] for c in protocol["cases"]}
    got_ids = set(case_ids)
    if got_ids != want_ids:
        missing, extra = sorted(want_ids - got_ids), sorted(got_ids - want_ids)
        need(False, f"case coverage mismatch: missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                    f"extra {extra[:5]}{'...' if len(extra) > 5 else ''}")

    by_id = {c["case_id"]: c for c in protocol["cases"]}
    for a_ in attempts:  # an attempt must be the protocol case it claims to be
        c = by_id.get(a_["case_id"])
        if c is not None:
            need(all(a_[k] == c[k] for k in ("group", "object_id", "tilt_deg")),
                 f"attempt {a_['case_id']} group/object/tilt differ from the protocol case")
    from eval.merge import build_summary
    from eval.protocol_util import protocol_sha
    need(merged.get("summary_by_policy_ablate_group") == build_summary(merged.get("inputs", [])),
         "summary_by_policy_ablate_group does not match the attempts")
    for inp in merged.get("inputs", []):
        own = {}
        for a_ in inp["attempts"]:
            g = own.setdefault(a_["group"], {"success": 0, "attempts": 0})
            g["attempts"] += 1; g["success"] += int(a_["success"])
        need(inp.get("summary") == own, f"input summary {inp.get('summary')} != its attempts {own}")
        need(inp.get("protocol_sha256") == protocol_sha(protocol),
             f"input protocol_sha256 {inp.get('protocol_sha256')} != live protocol {protocol_sha(protocol)}")
    want_groups = Counter(c["group"] for c in protocol["cases"])
    got_groups = Counter(a["group"] for a in attempts)
    need(got_groups == want_groups, f"group counts {dict(got_groups)} != protocol {dict(want_groups)}")

    config_hashes = {i["config_hash"] for i in main_inputs}
    need(len(config_hashes) == 1, f"main results span {len(config_hashes)} distinct config_hash values")

    wshas = {tuple(sorted(i["weights_sha256"].items())) for i in main_inputs}
    need(len(wshas) == 1, f"main results span {len(wshas)} distinct weights_sha256 values")

    if not weights_dir.exists():
        need(False, f"{weights_dir} not found")
    else:
        cfg_path = weights_dir / "config.json"
        if len(config_hashes) == 1 and cfg_path.exists():
            actual_hash = json.loads(cfg_path.read_text())["config_hash"]
            recorded_hash = next(iter(config_hashes))
            need(recorded_hash == actual_hash,
                 f"config_hash {recorded_hash} != {cfg_path.relative_to(ROOT)} ({actual_hash})")
        if len(wshas) == 1:
            actual_wsha = weights_sha256(weights_dir)
            recorded_wsha = dict(next(iter(wshas)))
            need(recorded_wsha == actual_wsha,
                 f"weights_sha256 {recorded_wsha} != actual sha256 of {weights_dir.relative_to(ROOT)} ({actual_wsha})")

    if gif is not None:
        need(Path(gif).is_file(), f"gif not found: {gif}")
        need(gif_case is not None, "--gif-case is required together with --gif")
        if gif_case is not None:
            match = [a for a in attempts if a["case_id"] == gif_case]
            need(len(match) == 1, f"--gif-case {gif_case!r} does not appear exactly once in the main results")
            need(bool(match) and match[0].get("success") is True,
                 f"--gif-case {gif_case!r} is not a recorded success in the main results")

    return fails


def _run_cli(args):
    import subprocess
    return subprocess.run([sys.executable, "-m", "eval.audit_results", *args], cwd=ROOT,
                           capture_output=True, text=True)


def selftest():
    import tempfile

    train_control = json.loads((ROOT / "results/send_pc_train_control.json").read_text())
    protocol = json.loads(PROTOCOL_PATH.read_text())
    actual_wsha = weights_sha256()
    actual_config_hash = json.loads((WEIGHTS / "config.json").read_text())["config_hash"]

    def full_coverage_input(wsha=None, config_hash=None):
        # Real train_control attempts (from the committed fixture) + synthetic success=True
        # attempts for every other protocol case, so the "main results" cover all 70 cases.
        attempts = list(train_control["attempts"])
        have = {a["case_id"] for a in attempts}
        for c in protocol["cases"]:
            if c["case_id"] not in have:
                attempts.append({"case_id": c["case_id"], "group": c["group"], "object_id": c["object_id"],
                                  "tilt_deg": c["tilt_deg"], "success": True, "depth_mm": 21.0,
                                  "steps": 50, "max_force_n": 10.0})
        summary = {}
        for a_ in attempts:
            g = summary.setdefault(a_["group"], {"success": 0, "attempts": 0})
            g["attempts"] += 1; g["success"] += int(a_["success"])
        return {"policy": "send_pc", "visual": "pc", "config_hash": config_hash or actual_config_hash,
                "weights_sha256": wsha or actual_wsha, "protocol_sha256": _live_sha(),
                "ablate_pc": False, "ablate_wrench": False, "attempts": attempts, "summary": summary}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # --- positive: full protocol coverage, real config_hash/weights_sha256, plus a gif check
        from eval.merge import build_summary
        good = {"protocol_sha256": _live_sha(), "inputs": [full_coverage_input()]}
        good["summary_by_policy_ablate_group"] = build_summary(good["inputs"])
        good_p = td / "good.json"
        good_p.write_text(json.dumps(good))
        gif_p = td / "f3.gif"
        gif_p.write_bytes(b"GIF89a")
        success_f3_case = next(a["case_id"] for a in good["inputs"][0]["attempts"]
                                if a["group"] == "f3" and a["success"])
        r = _run_cli([str(good_p), "--gif", str(gif_p), "--gif-case", success_f3_case])
        assert r.returncode == 0, f"positive case should PASS, got: {r.stdout}{r.stderr}"
        assert "PASS" in r.stdout, r.stdout
        print(f"PASS positive: full coverage + real weights sha256 + gif case {success_f3_case} -> PASS")

        # --- negative 1: duplicate case_id / broken coverage (two main inputs share a case_id)
        dup = {"protocol_sha256": _live_sha(), "summary_by_policy_ablate_group": {},
               "inputs": [full_coverage_input(), full_coverage_input()]}
        dup_p = td / "dup.json"
        dup_p.write_text(json.dumps(dup))
        r = _run_cli([str(dup_p)])
        assert r.returncode == 1 and "FAIL" in r.stdout, r.stdout
        assert "more than once" in r.stdout, r.stdout
        print("PASS negative: duplicate case_id across main inputs rejected")

        # --- negative 2: weights_sha256 doesn't match the real weights/send_pc files
        bad_wsha = {"encoder.pt": "0" * 16, "policy.pt": "0" * 16}
        bad = {"protocol_sha256": _live_sha(), "summary_by_policy_ablate_group": {},
               "inputs": [full_coverage_input(wsha=bad_wsha)]}
        bad_p = td / "bad_wsha.json"
        bad_p.write_text(json.dumps(bad))
        r = _run_cli([str(bad_p)])
        assert r.returncode == 1 and "FAIL" in r.stdout, r.stdout
        assert "weights_sha256" in r.stdout, r.stdout
        print("PASS negative: weights_sha256 mismatch against weights/send_pc rejected")

        # --- negative 3 (E5 P1-3): a summary that disagrees with the attempts
        forged = json.loads(json.dumps(good))
        forged["summary_by_policy_ablate_group"]["send_pc"]["none"]["f3"]["success"] -= 1
        forged_p = td / "forged_summary.json"
        forged_p.write_text(json.dumps(forged))
        r = _run_cli([str(forged_p)])
        assert r.returncode == 1 and "does not match the attempts" in r.stdout, r.stdout
        print("PASS negative: summary not matching the attempts rejected")

        # --- negative 4 (E5 P2): an attempt relabelled into another group
        swapped = json.loads(json.dumps(good))
        att = next(a for a in swapped["inputs"][0]["attempts"] if a["group"] == "f4")
        att["group"] = "f3"
        swapped["inputs"][0]["summary"] = None
        swapped_p = td / "swapped_group.json"
        swapped_p.write_text(json.dumps(swapped))
        r = _run_cli([str(swapped_p)])
        assert r.returncode == 1 and "differ from the protocol case" in r.stdout, r.stdout
        print("PASS negative: attempt relabelled into another group rejected")

        # --- negative 5 (E5 P1-2): results produced under a different protocol
        stale = json.loads(json.dumps(good))
        stale["inputs"][0]["protocol_sha256"] = "0" * 16
        stale_p = td / "stale_protocol.json"
        stale_p.write_text(json.dumps(stale))
        r = _run_cli([str(stale_p)])
        assert r.returncode == 1 and "live protocol" in r.stdout, r.stdout
        print("PASS negative: protocol_sha256 mismatch rejected")

    print("eval.audit_results selftest: ALL PASS")


def _live_sha():
    from eval.protocol_util import protocol_sha
    return protocol_sha(json.loads((ROOT / "eval/protocol.json").read_text()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(DEFAULT_RESULTS))
    ap.add_argument("--gif")
    ap.add_argument("--gif-case")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    merged = json.loads(Path(a.results).read_text())
    protocol = json.loads(PROTOCOL_PATH.read_text())
    fails = audit(merged, protocol, gif=a.gif, gif_case=a.gif_case)
    print("FAIL\n  " + "\n  ".join(fails) if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
