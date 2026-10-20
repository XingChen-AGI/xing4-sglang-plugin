#!/usr/bin/env python3
"""Compare two golden output directories collected by collect_golden.py.

Semantics:
* Greedy cases (temperature=0): choices and usage must match token for token.
* Tool-call cases: server-generated ``call_<hex>`` ids are normalized away
  before comparing (they are random per request).
* Sampling cases (a non-zero temperature, e.g. 05_recommended_sampling):
  only the shape is compared (finish_reason and completion_tokens), because
  sampled text is not reproducible across sglang builds even with a fixed
  seed (EAGLE acceptance changes the RNG consumption sequence).

Usage:
    python compare_golden.py <golden_dir_a> <golden_dir_b>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_CALL_ID = re.compile(r"call_[0-9a-f]+")


def _normalize(obj):
    if isinstance(obj, str):
        return _CALL_ID.sub("call_X", obj)
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    return obj


def _is_sampling_case(name: str) -> bool:
    # collect_golden.py does not persist the request body; sampling cases are
    # identified by their file name convention.
    return "sampling" in name


def compare(name: str, a: dict, b: dict) -> list[str]:
    problems: list[str] = []
    if _is_sampling_case(name):
        for key in ("finish_reason",):
            va = a["choices"][0].get(key)
            vb = b["choices"][0].get(key)
            if va != vb:
                problems.append(f"{key}: {va!r} != {vb!r}")
        ca = a["usage"].get("completion_tokens")
        cb = b["usage"].get("completion_tokens")
        if ca is not None and ca != cb:
            problems.append(
                f"completion_tokens: {ca} != {cb} (sampled text may differ, "
                "but length should match for length-capped cases)"
            )
        return problems

    for key in ("choices", "usage", "model"):
        va, vb = _normalize(a.get(key)), _normalize(b.get(key))
        if va != vb:
            problems.append(f"{key} differs")
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    dir_a, dir_b = Path(sys.argv[1]), Path(sys.argv[2])
    failures = 0
    for path in sorted(dir_a.glob("*.json")):
        if path.name == "model_info.json":
            continue
        other = dir_b / path.name
        if not other.exists():
            print(f"[MISSING] {path.name} not in {dir_b}")
            failures += 1
            continue
        problems = compare(path.name, json.loads(path.read_text()), json.loads(other.read_text()))
        if problems:
            failures += 1
            print(f"[MISMATCH] {path.name}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"[MATCH]    {path.name}")
    print()
    print("GOLDEN MATCH: PASS" if failures == 0 else "GOLDEN MATCH: FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
