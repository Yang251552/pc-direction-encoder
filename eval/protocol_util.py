"""Shared protocol hash. Covers the frozen content only: `frozen_at` and the `amendments` log are excluded,
so appending an amendment note does not change the hash (changes to cases/criteria/thresholds do)."""
import hashlib, json


def protocol_sha(p):
    return hashlib.sha256(json.dumps({k: v for k, v in p.items() if k not in ("frozen_at", "amendments")},
                                     sort_keys=True).encode()).hexdigest()[:16]
