#!/usr/bin/env python3
"""Collect golden outputs from a running Xing4.0 SGLang server.

Used for A/B comparison between the in-tree support_xing4_0 branch and the
out-of-tree plugin (Phase 0 baseline vs Phase 3 verification).

Usage:
    python collect_golden.py --url http://localhost:8000 --out-dir ./p0_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

MODEL = "Xing4.0-29B-A4B"

CASES = [
    # (name, request_body)
    (
        "01_plain_greedy",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
            "temperature": 0,
            "max_tokens": 64,
        },
    ),
    (
        "02_reasoning_greedy",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "9.11 和 9.9 哪个大？"}],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": True},
            "skip_special_tokens": False,
        },
    ),
    (
        "03_tool_call_greedy",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "查一下北京今天的天气"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询指定城市的当前天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "temperature": 0,
            "max_tokens": 256,
        },
    ),
    (
        "04_code_greedy",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "写一个 Python 函数计算斐波那契数列第 n 项，只输出代码。"}],
            "temperature": 0,
            "max_tokens": 256,
        },
    ),
    (
        "05_recommended_sampling",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "用三句话解释什么是张量并行。"}],
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "max_tokens": 256,
            "seed": 42,
        },
    ),
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    import pathlib

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Request metadata snapshot (for A/B provenance). Local filesystem paths are
    # scrubbed so that a collected snapshot can be shared without leaking the
    # machine layout or the exact checkpoint directory name.
    with urllib.request.urlopen(args.url.rstrip("/") + "/get_model_info", timeout=30) as r:
        model_info = json.loads(r.read())
    for key in ("model_path", "tokenizer_path"):
        if key in model_info:
            model_info[key] = "<redacted>"
    (out / "model_info.json").write_text(json.dumps(model_info, ensure_ascii=False, indent=2))

    failures = 0
    for name, body in CASES:
        try:
            result = post(args.url, body)
            (out / f"{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            msg = result["choices"][0]["message"]
            flags = []
            if msg.get("reasoning_content"):
                flags.append(f"reasoning[{len(msg['reasoning_content'])}c]")
            if msg.get("tool_calls"):
                flags.append(f"tool_calls={len(msg['tool_calls'])}")
            text = (msg.get("content") or "")[:60].replace("\n", " ")
            print(f"[OK]   {name}: finish={result['choices'][0]['finish_reason']} "
                  f"tokens={result['usage']['completion_tokens']} {' '.join(flags)}")
            print(f"       text: {text}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {e}", file=sys.stderr)

    print(f"\nDone: {len(CASES) - failures}/{len(CASES)} succeeded -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
