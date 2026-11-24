"""Create input signatures and manifests for reusable analysis outputs."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_signature(payload):
    canonical = _canonical_json(payload)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def path_fingerprint(path):
    path = Path(path)
    info = {
        "path": str(path.resolve()),
        "exists": path.exists(),
    }
    if not path.exists():
        return info

    stat = path.stat()
    info.update(
        {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    return info


def load_manifest(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def outputs_exist(paths):
    for path in paths:
        if path is None:
            continue
        if not Path(path).exists():
            return False
    return True


def write_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data.setdefault("generated_at_utc", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
