"""E4 gate: audit README.md against a merged results file. Prints PASS/FAIL and reasons.
Exit 0 on PASS, 1 on FAIL.

  .venv/bin/python -m eval.audit_readme README.md results/results.json
  .venv/bin/python -m eval.audit_readme --selftest

Checks:
  - every "N/M" success-rate token in the README has a matching (group, success, attempts) triple
    somewhere in results.json's group summaries, found via a group name (e.g. "f3", "train_control")
    on the same line -- group names come from whatever the results file actually reports,
    no hardcoded list, so this stays correct if groups change.
  - the "front matter" (everything before the first "## " heading) contains: the opening-sentence
    keyword "minimal direction-aware point cloud encoder", a .gif reference, at least one "N/M",
    and a line mentioning "hard-coded".
  - no line containing "RGB" is missing a qualifier ("not yet run" / "configured") -- guards against
    README wording that RGB has already been compared (PROJECT_SPEC.md Non-goals / f5 decision).
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/results.json"
NM_RE = re.compile(r"(?<!\d)(\d+)\s*(?:/|\bof\b)\s*(\d+)(?!\d)")  # "6/20" and "6 of 20"
OPEN_SENTENCE_KEYWORD = "minimal direction-aware point cloud encoder"
# Small fixed synonym list per PROJECT_SPEC wording ("not yet run" / "configured");
# extend here if README phrasing legitimately varies.
RGB_QUALIFIERS = ("not yet run", "configured")


def extract_group_pairs(doc):
    """group name -> set of (success, attempts) pairs seen anywhere in this results doc.
    Prefers eval.merge's summary_by_policy_ablate_group; falls back to a raw eval.run "summary"
    or a list of per-input "summary" dicts, so this also works on an un-merged results file."""
    pairs = {}

    def add(group, succ, att):
        pairs.setdefault(group, set()).add((succ, att))

    if "summary_by_policy_ablate_group" in doc:
        for by_ablate in doc["summary_by_policy_ablate_group"].values():
            for groups in by_ablate.values():
                for g, s in groups.items():
                    add(g, s["success"], s["attempts"])
    elif "summary" in doc:
        for g, s in doc["summary"].items():
            add(g, s["success"], s["attempts"])
    elif "inputs" in doc:
        for i in doc["inputs"]:
            for g, s in i.get("summary", {}).items():
                add(g, s["success"], s["attempts"])
    return pairs


def front_matter(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("## "):
            return "\n".join(lines[:i])
    return text


def check_success_rates(text, pairs):
    """Match each 'N/M' to the group name closest before it on the same line (not just any group
    name anywhere on the line) -- a line can legitimately report two groups, e.g.
    'f3: 5/20. train_control: 3/10.', and each number must line up with its own group."""
    fails, found_any = [], False
    group_names = list(pairs)
    for lineno, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        for m in NM_RE.finditer(line):
            found_any = True
            succ, att = int(m.group(1)), int(m.group(2))
            prefix = low[:m.start()]
            group, group_pos = None, -1
            for g in group_names:
                pos = prefix.rfind(g.lower())
                if pos > group_pos:
                    group, group_pos = g, pos
            if group is None:
                fails.append(f"line {lineno}: '{m.group(0)}' has no recognizable group name before it on the same line")
            elif (succ, att) not in pairs[group]:
                fails.append(f"line {lineno}: '{m.group(0)}' (matched to group '{group}') not found in results "
                             f"(known for '{group}': {sorted(pairs[group])})")
    return fails, found_any


def check_rgb_wording(text):
    fails = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if "RGB" in line and not any(q in line.lower() for q in RGB_QUALIFIERS):
            fails.append(f"line {lineno}: 'RGB' appears without a qualifier like "
                         f"{RGB_QUALIFIERS!r} -- reads as an already-run comparison")
    return fails


def check_front_matter(front, pairs):
    fails = []
    if OPEN_SENTENCE_KEYWORD not in front.lower():
        fails.append(f"front matter missing opening-sentence keyword {OPEN_SENTENCE_KEYWORD!r}")
    if not re.search(r"\S+\.gif", front, re.IGNORECASE):
        fails.append("front matter missing a .gif reference")
    _, found_any = check_success_rates(front, pairs)
    if not found_any:
        fails.append("front matter missing at least one 'N/M' success-rate")
    if "hard-coded" not in front.lower():
        fails.append("front matter missing a line mentioning 'hard-coded'")
    return fails


OVERCLAIM_RE = re.compile(r"outperform|superior|better than|state of the art|generali[sz]\w*\b.{0,40}\b(novel|new|unseen)\b", re.I)
PERCENT_RE = re.compile(r"\d+(\.\d+)?\s?%")


def check_claims(text):
    """Overclaims (unsupported by the frozen results: PROJECT_SPEC Non-goals) and percentages instead of N/M."""
    fails = []
    for i, line in enumerate(text.splitlines(), 1):
        if OVERCLAIM_RE.search(line) and not re.search(r"\b(not|no|never|without)\b", line, re.I):
            fails.append(f"line {i}: overclaim wording without a negation: {line.strip()[:80]!r}")
        if PERCENT_RE.search(line):
            fails.append(f"line {i}: success rates must be written as N/M, not percentages: {line.strip()[:80]!r}")
    return fails


def audit_readme(readme_text, results_doc):
    pairs = extract_group_pairs(results_doc)
    nm_fails, _ = check_success_rates(readme_text, pairs)
    fails = (nm_fails + check_rgb_wording(readme_text) + check_claims(readme_text)
             + check_front_matter(front_matter(readme_text), pairs))
    return list(dict.fromkeys(fails))  # de-dup (front-matter N/M lines get scanned twice)


def _run_cli(args):
    import subprocess
    return subprocess.run([sys.executable, "-m", "eval.audit_readme", *args], cwd=ROOT,
                           capture_output=True, text=True)


def selftest():
    import tempfile

    train_control = json.loads((ROOT / "results/send_pc_train_control.json").read_text())
    tc_succ, tc_att = (train_control["summary"]["train_control"]["success"],
                       train_control["summary"]["train_control"]["attempts"])
    results_doc = {
        "summary_by_policy_ablate_group": {
            "send_pc": {"none": {
                "train_control": {"success": tc_succ, "attempts": tc_att},  # from the real fixture
                "f3": {"success": 5, "attempts": 20},
                "f4": {"success": 3, "attempts": 30},
                "extrap": {"success": 1, "attempts": 10},
            }},
        },
    }

    good_readme = f"""# Direction-Aware Point Cloud Encoder

I built a minimal direction-aware point cloud encoder: input a point cloud with per-point
normals, principal curvatures and the nominal insertion axis -> output a trained encoder.

![demo](figures/f3_success.gif)

Held-out poses (f3): 5/20. Training-pose control (train_control): {tc_succ}/{tc_att}.

Camera extrinsics and the success threshold are hard-coded, not learned; see PROJECT_SPEC.md.

## Results

Novel objects (f4): 3/30. Extrapolation (extrap): 1/10.

RGB-only baseline is configured but not yet run (interview evidence phase).

## Next steps
"""

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        results_p = td / "results.json"
        results_p.write_text(json.dumps(results_doc))

        # --- positive
        good_p = td / "README.md"
        good_p.write_text(good_readme)
        r = _run_cli([str(good_p), str(results_p)])
        assert r.returncode == 0 and "PASS" in r.stdout, f"positive case should PASS, got: {r.stdout}{r.stderr}"
        print("PASS positive: well-formed README against results.json -> PASS")

        # --- negative 1: README number has no provenance in results.json
        no_provenance = good_readme.replace("Held-out poses (f3): 5/20", "Held-out poses (f3): 7/20")
        bad1_p = td / "README_bad_number.md"
        bad1_p.write_text(no_provenance)
        r = _run_cli([str(bad1_p), str(results_p)])
        assert r.returncode == 1 and "FAIL" in r.stdout, r.stdout
        assert "not found in results" in r.stdout, r.stdout
        print("PASS negative: README number without provenance in results.json rejected")

        # --- negative 2: bare "RGB" claim without a qualifier
        bare_rgb = good_readme.replace(
            "RGB-only baseline is configured but not yet run (interview evidence phase).",
            "The point cloud encoder outperforms the RGB baseline.")
        bad2_p = td / "README_bare_rgb.md"
        bad2_p.write_text(bare_rgb)
        r = _run_cli([str(bad2_p), str(results_p)])
        assert r.returncode == 1 and "FAIL" in r.stdout, r.stdout
        assert "RGB" in r.stdout and "qualifier" in r.stdout, r.stdout
        print("PASS negative: bare RGB comparison claim rejected")

    print("eval.audit_readme selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readme", nargs="?", default=str(ROOT / "README.md"))
    ap.add_argument("results", nargs="?", default=str(DEFAULT_RESULTS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    text = Path(a.readme).read_text()
    doc = json.loads(Path(a.results).read_text())
    fails = audit_readme(text, doc)
    print("FAIL\n  " + "\n  ".join(fails) if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
