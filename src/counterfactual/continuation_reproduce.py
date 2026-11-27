"""Rebuild each split in an isolated directory and check numerical identities."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .architecture_data import digest
from .competition_audit import compare_manifests
from .continuation_data import DATA, PROTOCOL, REPORT, build
from .continuation_experiment import write


def run():
    cfg = json.loads(PROTOCOL.read_text())
    seen = set()
    out = {}
    with TemporaryDirectory() as temp:
        for name, spec in {"pilot": [70000, 48, 400], **cfg["splits"]}.items():
            source = DATA / name
            old = json.loads((source / "manifest.json").read_text())
            new = build(
                name, *spec, Path(temp), "construction-pilot" if name == "pilot" else digest(PROTOCOL)
            )
            archive = digest(source / "branches.npz")
            if new["archive"] != archive:
                raise ValueError(f"Different states, contexts or labels in {name}")
            count = compare_manifests(old, new)
            seeds = {p["scene"]["seed"] for p in old["records"]}
            if seeds & seen or len(seeds) != spec[1]:
                raise ValueError("Parent overlap")
            seen.update(seeds)
            out[name] = {
                "parents": len(seeds),
                "archive": archive,
                "manifest": digest(source / "manifest.json"),
                "simultaneous_callback_reorderings": count,
            }
    write(REPORT / "reproduction.json", out)


if __name__ == "__main__":
    run()
