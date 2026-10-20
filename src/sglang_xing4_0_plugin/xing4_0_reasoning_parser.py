# SPDX-License-Identifier: Apache-2.0
"""Reasoning format detector for Xing4_0 models.

Vendored from sglang PR https://github.com/sgl-project/sglang/pull/39793
(branch support_xing4_0, commit aa737beda5), where it lives inline in
``sglang/srt/parser/reasoning_parser.py``.
"""

from __future__ import annotations

from sglang.srt.parser.reasoning_parser import BaseReasoningFormatDetector


class Xing4_0Detector(BaseReasoningFormatDetector):
    """Detector for Xing4_0 models.

    Xing4_0 models use<think> and </think> as reasoning markers.
    The chat template injects these markers into the prompt via ``enable_thinking`` parameter.

    When ``enable_thinking=True``:
      The model generates reasoning content followed by </think> and then the answer.
      Format: "reasoning content</think>answer"

    When ``enable_thinking=False``:
      The model generates the answer directly without any markers.
      Format: "answer"

    ``force_nonempty_content=True`` acts as a safety net: if the model omits
    the ``</think>`` close tag, the accumulated reasoning is reclassified as
    normal text so the answer is never lost in the text field.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = True,
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )
