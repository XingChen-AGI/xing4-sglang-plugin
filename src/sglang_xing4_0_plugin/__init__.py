# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree XingChen Xing4_0 support for SGLang.

Entry point group: ``sglang.srt.plugins``. SGLang loads plugins at four
process points (cli/serve, launch_server, engine subprocess launcher, and
the scheduler process), so registration must be cheap, lazy, and safe to
call more than once.

The vendored code originates from sglang PR
https://github.com/sgl-project/sglang/pull/39793 (branch ``support_xing4_0``).
Once that PR merges, this plugin stands down automatically.

Note (Phase 2): the in-tree patch also touches ModelConfig internals, arg
group arch lists, and the mhc_pre splitk kernel; those pieces are carried by
``sglang_xing4_0_plugin.compat`` hooks until the upstream PR merges. The
compat layer installs itself at the end of :func:`register` (i.e. only
when not standing down) and is idempotent on top of in-tree builds.
"""

from __future__ import annotations

import logging
import os

__all__ = [
    "ARCHITECTURE",
    "MODELS_PACKAGE",
    "OVERRIDE_ENV",
    "PARSER_NAME",
    "TESTED_SGLANG",
    "register",
]

logger = logging.getLogger(__name__)

ARCHITECTURE = "Xing4_0ForCausalLM"
PARSER_NAME = "xing4_0"
MODELS_PACKAGE = "sglang_xing4_0_plugin.models"
# support_xing4_0 branch tip == upstream PR sglang-project/sglang#39793 head;
# the model layer is ported to the 0.5.x stage-boundary API (append_stages).
TESTED_SGLANG = "aa737beda5 (0.4.x) / 0.5.22"
OVERRIDE_ENV = "XING4_0_PLUGIN_OVERRIDE"

_TRUE = frozenset({"1", "true", "yes", "on"})


def _override_requested() -> bool:
    """Return whether the operator explicitly requested an override."""
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in _TRUE


def _register_model(overwrite: bool) -> None:
    """Register the vendored model package with SGLang's ModelRegistry.

    strict=True so a broken vendored import fails loudly at load time
    instead of silently leaving the architecture unregistered.
    """
    from sglang.srt.models.registry import ModelRegistry

    ModelRegistry.register(MODELS_PACKAGE, overwrite=overwrite, strict=True)


def _register_config() -> None:
    """Expose Xing4_0Config through sglang's config registry."""
    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

    from sglang_xing4_0_plugin.config import Xing4_0Config

    _CONFIG_REGISTRY[Xing4_0Config.model_type] = Xing4_0Config


def _register_reasoning_parser() -> None:
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    from sglang_xing4_0_plugin.xing4_0_reasoning_parser import Xing4_0Detector

    ReasoningParser.DetectorMap[PARSER_NAME] = Xing4_0Detector


def _register_tool_parser() -> None:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    from sglang_xing4_0_plugin.xing4_0_tool_parser import Xing4_0Detector

    FunctionCallParser.ToolCallParserEnum[PARSER_NAME] = Xing4_0Detector


def _install_compat() -> None:
    """Install the transitional compat hooks (see :mod:`sglang_xing4_0_plugin.compat`).

    Only reached when not standing down. Every item is idempotent, so
    installing over an in-tree-aware build (OVERRIDE mode) is safe.
    """
    from sglang_xing4_0_plugin import compat

    compat.install_all()


def register() -> None:
    """Register Xing4_0 with SGLang's ``sglang.srt.plugins`` entry point."""
    from sglang.srt.models.registry import ModelRegistry

    in_tree = ARCHITECTURE in ModelRegistry.get_supported_archs()
    if in_tree and not _override_requested():
        logger.info(
            "sglang-xing4_0-plugin: %s is already supported in tree; standing "
            "down. Set %s=1 to override with the vendored copy.",
            ARCHITECTURE,
            OVERRIDE_ENV,
        )
        return

    # in_tree + override: register with overwrite so we replace the in-tree
    # class instead of raising "already registered".
    _register_model(overwrite=in_tree)
    _register_config()
    _register_reasoning_parser()
    _register_tool_parser()
    _install_compat()
    logger.info(
        "sglang-xing4_0-plugin: registered %s (models=%s, parser=%s, "
        "vendored from sglang commit %s)",
        ARCHITECTURE,
        MODELS_PACKAGE,
        PARSER_NAME,
        TESTED_SGLANG,
    )
