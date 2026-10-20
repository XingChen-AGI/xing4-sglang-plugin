# SPDX-License-Identifier: Apache-2.0
"""OpenAI-serving compat for Xing4_0's reasoning markers.

The Xing4_0 reasoning delimiters (``</_think>`` ...) are *special* tokens, so
the OpenAI-serving default ``skip_special_tokens=True`` strips them from the
detokenized stream before the reasoning parser can split thinking from the
answer: every thinking-mode request then lands, unsplit, in ``content``.

sglang main added ``_patch_reasoning_skip_special_tokens`` for exactly this
class of parsers (kimi_k3, apertus2509, inkling, muse — their delimiters are
special tokens too); the vendored PR#39793 predates that list, so the plugin
adds itself here. With this hook, ``skip_special_tokens`` is forced to False
for requests routed through the ``xing4_0`` reasoning parser, and thinking
mode splits correctly without every client passing the flag by hand.
"""

from __future__ import annotations

import logging

from sglang_xing4_0_plugin import PARSER_NAME

logger = logging.getLogger(__name__)

ITEM = "REASONING_SKIP"
_MARKER = "_xing4_0_reasoning_skip_patched"


def install() -> list[str]:
    from sglang_xing4_0_plugin.compat import compat_enabled

    if not compat_enabled(ITEM):
        return []

    try:
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    except Exception:
        # Not the HTTP-serving process (or the layout changed): nothing to do.
        return []

    if getattr(OpenAIServingChat, _MARKER, False):
        return [ITEM]

    original = getattr(
        OpenAIServingChat, "_patch_reasoning_skip_special_tokens", None
    )
    if original is None:
        # The layout changed; clients can still pass skip_special_tokens=False.
        return []

    def _patched(self, request) -> None:
        if getattr(self, "reasoning_parser", None) == PARSER_NAME:
            # Xing4_0's </_think> is a special token: keep it in the stream
            # so the reasoning parser can find the split point.
            request.skip_special_tokens = False
        original(self, request)

    try:
        _patched.__name__ = original.__name__
        _patched.__qualname__ = original.__qualname__
        _patched.__doc__ = original.__doc__
        OpenAIServingChat._patch_reasoning_skip_special_tokens = _patched
        setattr(OpenAIServingChat, _MARKER, True)
        logger.debug(
            "sglang-xing4_0-plugin: keep special reasoning markers in the "
            "OpenAI stream (skip_special_tokens=False for parser %s)",
            PARSER_NAME,
        )
        return [ITEM]
    except Exception:
        logger.exception(
            "sglang-xing4_0-plugin: failed to install the %s compat hook", ITEM
        )
        return []
