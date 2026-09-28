#!/usr/bin/env bash
# E3' (send-line item 1): smoke test on a fresh clone of the COMMITTED repo -- proves README's
# environment setup + the committed weights/send_pc actually work, without a full retrain.
#
#   clone (git, local path, committed content only) -> plain venv (exactly as README) -> pip install
#   requirements.txt -> python third_party/fetch_menagerie.py -> load weights/send_pc, predict()
#   + run its actions through env.step -> python -m policy.check_encoder weights/send_pc/encoder.pt
#
#
# Usage:
#   bash scripts/smoke_test.sh            # full run
#   bash scripts/smoke_test.sh --dry-run  # print the steps only, run nothing
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEM_PY=/usr/local/bin/python3.12
SCRATCH="$(mktemp -d -t pc-direction-encoder-smoke.XXXXXX)"
REPO="$SCRATCH/repo"

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

announce() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[dry-run] $*"
    else
        echo "==> $*"
    fi
}

run() {
    # run <description> -- <command...>
    local desc="$1"; shift
    announce "$desc"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '    $ %s\n' "$*"
        return 0
    fi
    "$@"
}

START_TS=$(date +%s)

run "clone the committed repo (no uncommitted changes) into $REPO" \
    git clone --quiet "$ROOT" "$REPO"

run "create a plain venv (exactly as README)" \
    "$SYSTEM_PY" -m venv "$REPO/.venv"   # same as README: no system site-packages

VENV_PY="$REPO/.venv/bin/python"

# Everything below runs with cwd=$REPO: fetch_menagerie.py and the -m module invocations are
# all written relative to the repo root, matching how README's own commands are meant to run.
if [[ "$DRY_RUN" == 0 ]]; then
    cd "$REPO"
fi

if [[ "$DRY_RUN" == 1 ]]; then
    announce "install requirements: (cd $REPO &&) $VENV_PY -m pip install -r requirements.txt"
    printf '    $ %s\n' "$VENV_PY -m pip install -r requirements.txt"
else
    if [[ ! -f requirements.txt ]]; then
        echo "FAIL: $REPO/requirements.txt not found (not committed at HEAD yet)." >&2
        echo "      orchestrator: add requirements.txt at the repo root and commit it." >&2
        exit 1
    fi
    echo "==> install requirements"
    "$VENV_PY" -m pip install -r requirements.txt
fi

run "fetch the FR3 menagerie model (~57 MB, @c96a32d)" \
    "$VENV_PY" third_party/fetch_menagerie.py

announce "load weights/send_pc, predict() on the protocol's first f3 case, run its actions through env.step"
if [[ "$DRY_RUN" == 1 ]]; then
    cat <<'EOF'
    $ python - <<'PY'
    import json, numpy as np
    from policy.model import load_policy
    from sim.env import PegEnv
    protocol = json.loads(open("eval/protocol.json").read())
    case = next(c for c in protocol["cases"] if c["group"] == "f3")
    policy = load_policy("weights/send_pc")
    env = PegEnv()
    obs = env.reset(case)
    oh = {k: np.asarray(obs[k])[None] for k in policy.obs_keys()}
    for a in policy.predict(oh):
        obs, done, info = env.step(a)
    PY
EOF
else
    "$VENV_PY" - <<'PY'
import json
import numpy as np
from policy.model import load_policy
from sim.env import PegEnv

protocol = json.loads(open("eval/protocol.json").read())
case = next(c for c in protocol["cases"] if c["group"] == "f3")
policy = load_policy("weights/send_pc")
env = PegEnv()
obs = env.reset(case)
oh = {k: np.asarray(obs[k])[None] for k in policy.obs_keys()}
actions = policy.predict(oh)
for a in actions:
    obs, done, info = env.step(a)
print(f"smoke inference OK: case={case['case_id']} n_actions={len(actions)} success={info['success']}")
PY
fi

run "check the committed encoder" \
    "$VENV_PY" -m policy.check_encoder weights/send_pc/encoder.pt

END_TS=$(date +%s)
echo "smoke test finished in $((END_TS - START_TS))s"
