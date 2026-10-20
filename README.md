# SGLang Xing4_0 Plugin

> This document targets Linux systems and Bash-compatible shells.

This package provides XingChen-Xing4_0 (e.g. `XingChen-AGI/Xing4.0-29B-A4B`)
support for [SGLang](https://github.com/sgl-project/sglang) as an out-of-tree
plugin. It enables loading and serving Xing4_0 models **without modifying the
SGLang source tree**. The plugin registers:

- `Xing4_0ForCausalLM`, the Xing4_0 model implementation;
- the `xing4_0` Transformers configuration (`model_type: "xing4_0"`);
- the `xing4_0` reasoning parser (think-block extraction);
- the `xing4_0` tool-call parser (JSON, name+JSON and tag-based calls);
- EAGLE / NextN speculative decoding, defaulting the draft model
  path to the target checkpoint (Xing4_0 bundles its MTP head);
- MLA-attention + MoE routing recognition via the compat hooks.

The implementation is vendored from SGLang PR
[#39793](https://github.com/sgl-project/sglang/pull/39793) (branch
`support_xing4_0`, baseline commit `aa737beda5`) and targets SGLang 0.5.x. It
does not ship custom CUDA kernels; it reuses SGLang's existing MLA / MoE /
DeepGEMM / FlashInfer runtime.

## Install

Install SGLang first, then install this package into the **same Python
environment**. The editable install is important because SGLang discovers the
plugin through the `sglang.srt.plugins` entry-point metadata.

```bash
pip install -e . --no-deps
```

`--no-deps` prevents the plugin from replacing the SGLang installation already
selected for the environment. Verify that the entry point is visible to the
interpreter used to launch SGLang:

```bash
python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='sglang.srt.plugins')])"
```

The output should include:

```
('xing4_0', 'sglang_xing4_0_plugin:register')
```

Adding `src/` to `PYTHONPATH` alone is **not** sufficient for normal plugin
discovery because entry points come from installed package metadata.

## Usage

Start the OpenAI-compatible server:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1

sglang serve /path/to/xing4_0-model \
  --trust-remote-code \
  --tp-size 2 \
  --context-length 262144
```

Reasoning extraction (think blocks):

```bash
sglang serve /path/to/xing4_0-model \
  --trust-remote-code \
  --reasoning-parser xing4_0
```

Tool calling:

```bash
sglang serve /path/to/xing4_0-model \
  --trust-remote-code \
  --tool-call-parser xing4_0 \
  --chat-template /path/to/chat_template.jinja
```

EAGLE speculative decoding (fully automatic — set only `speculative-algorithm`):

```bash
sglang serve /path/to/xing4_0-model \
  --trust-remote-code \
  --speculative-algorithm EAGLE
```

> **Gotcha:** SGLang 0.5.x requires either fully automatic (no
> `num_steps`/`topk`/`draft_tokens`), fully manual (all three), or adaptive
> (`--speculative-adaptive`). Setting only two of the three triggers an
> `AssertionError`.

If SGLang already provides `Xing4_0ForCausalLM` in tree, the plugin logs a
stand-down message and leaves the in-tree implementation active. To
explicitly use the vendored implementation:

```bash
XING4_0_PLUGIN_OVERRIDE=1 sglang serve /path/to/xing4_0-model ...
```

## Reasoning format

The chat template injects the think-start marker into the prompt via
`add_generation_prompt`; the model itself does not emit it again. Generated
output therefore looks like:

- thinking enabled: `<think>reasoning ... </think> content`
- thinking disabled (via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`): pure content, no markers.

## Tool-call format

Tool calls are emitted inside the model's tool-call sentinel tags
(`<tool_call>` ... `</tool_call>`, exposed as `TOOL_START_TOKEN` /
`TOOL_END_TOKEN` in `sglang_xing4_0_plugin.xing4_0_tool_parser`). Three payload
forms are recognized:

1. JSON: `{"name": "func", "arguments": {...}}`
2. name + JSON arguments: `func{"arguments": {...}}`
3. tag-based:

```
func
<param_key>arg1</param_key>
<param_value>value1</param_value>
...
```

Values declared as integer / number / boolean / JSON in the tool schema are
converted to their JSON types; string-typed values are kept verbatim.

## Plugin loading

SGLang 0.5.x loads plugins automatically at three process boundaries (the CLI
entry point, the engine launcher, and the scheduler subprocess). The entry
point group is `sglang.srt.plugins`. SGLang does not currently expose an
allowlist env var for this group; set `PYTHONPATH` or uninstall the plugin
if you need to suppress it.

## Compatibility and maintenance

The vendored model imports SGLang's internal runtime-layer APIs. The
`TESTED_SGLANG` value in `src/sglang_xing4_0_plugin/__init__.py` records the
tested revision (`aa737beda5` / 0.5.22). When upgrading SGLang, re-run the
tests and check for drift.

Beyond the model class, the upstream PR#39793 touches several SGLang internals
(frozenset architecture lists, `ModelConfig` draft remapping, the `mhc_pre`
split-k kernel, and the EAGLE draft path auto-default). These are replicated
as lazy, idempotent `HookRegistry` hooks under `sglang_xing4_0_plugin.compat`.
Every item is safe to install on an already-patched build and can be killed
individually with `XING4_0_COMPAT_OFF=1` (whole layer) or
`XING4_0_COMPAT_<ITEM>=0` (single item). As PR#39793 merges piece by piece,
items should be deleted one at a time; once everything is in-tree,
`register()` stands down automatically.

The plugin logs both the tested revision and the running SGLang version when
it registers, making version drift visible in the server log.

## Tests

The plugin ships a smoke test (`tests/test_smoke.py`) and parameterised golden
regressions (`tests/golden/`, comparing canned prompt output against two
architectural scenarios — pristine upstream SGLang vs. a fully-patched fork).
Neither requires a GPU or model weights.

```bash
pip install pytest
python -m pytest tests/ -q
```

## Project layout

```
xing4-sglang-plugin/
├── src/sglang_xing4_0_plugin/
│   ├── __init__.py                       # register() entry point
│   ├── config.py                         # Xing4_0Config
│   ├── models/
│   │   ├── __init__.py
│   │   └── xing4_0.py                    # vendored Xing4_0 model
│   ├── xing4_0_reasoning_parser.py       # vendored reasoning parser
│   ├── xing4_0_tool_parser.py            # vendored tool-call parser
│   └── compat/                           # transitional HookRegistry hooks
│       ├── __init__.py
│       ├── arg_groups.py
│       ├── model_config.py
│       ├── mhc_pre.py
│       └── serving.py
├── tests/
│   ├── conftest.py
│   ├── test_smoke.py
│   ├── test_compat.py
│   └── golden/
└── pyproject.toml
```

## License

Apache-2.0. The vendored SGLang implementation retains its upstream copyright
headers.
