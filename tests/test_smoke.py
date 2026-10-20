# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for sglang_xing4_0_plugin.

Runs against the *current* sglang environment:

- On a tree that already has in-tree Xing4_0 support (e.g. the
  ``support_xing4_0`` branch), these tests exercise the stand-down path and
  verify the vendored artifacts independently.
- On pristine sglang main (Phase 3), the same tests exercise the full
  registration path.
"""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest

import sglang_xing4_0_plugin
from sglang_xing4_0_plugin import (
    ARCHITECTURE,
    MODELS_PACKAGE,
    OVERRIDE_ENV,
    PARSER_NAME,
)


@pytest.fixture(scope="module", autouse=True)
def _registered():
    sglang_xing4_0_plugin.register()


def test_entry_point_registered():
    eps = importlib.metadata.entry_points(group="sglang.srt.plugins")
    matches = [ep for ep in eps if ep.name == PARSER_NAME]
    assert matches, "entry point sglang.srt.plugins::xing4_0 not found"
    assert matches[0].value == "sglang_xing4_0_plugin:register"


def test_register_is_idempotent():
    sglang_xing4_0_plugin.register()
    sglang_xing4_0_plugin.register()  # must not raise


def test_parsers_available_and_instantiable():
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    assert PARSER_NAME in ReasoningParser.DetectorMap
    assert PARSER_NAME in FunctionCallParser.ToolCallParserEnum
    # Instantiable standalone (no server required)
    ReasoningParser.DetectorMap[PARSER_NAME](stream_reasoning=True)
    FunctionCallParser.ToolCallParserEnum[PARSER_NAME]()


def test_model_arch_registered():
    from sglang.srt.models.registry import ModelRegistry

    assert ModelRegistry.is_arch_supported(ARCHITECTURE)


def test_config_registry_entry():
    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

    from sglang_xing4_0_plugin.config import Xing4_0Config

    assert _CONFIG_REGISTRY["xing4_0"] is not None
    assert Xing4_0Config.model_type == "xing4_0"
    assert _CONFIG_REGISTRY["xing4_0"].model_type == "xing4_0"


def test_vendored_model_module_importable():
    mod = importlib.import_module(MODELS_PACKAGE + ".xing4_0")
    assert mod.EntryClass == [mod.Xing4_0ForCausalLM]
    assert mod.Xing4_0ForCausalLM.__name__ == ARCHITECTURE


def test_no_override_env_by_default(monkeypatch):
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    sglang_xing4_0_plugin.register()  # stand-down path on in-tree envs
