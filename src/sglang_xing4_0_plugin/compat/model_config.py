# SPDX-License-Identifier: Apache-2.0
"""ModelConfig compat hooks for Xing4_0.

Replicates the ``configs/model_config.py`` hunks of the in-tree
``support_xing4_0`` patch (upstream PR
https://github.com/sgl-project/sglang/pull/39793, baseline commit
``aa737beda5``) as method-level hooks, so no class replacement (and no
pickle concerns) is needed:

* ``is_deepseek_dsa`` recognizes Xing4_0 (still gated on ``index_topk``).
* ``ModelConfig._config_draft_model`` remaps a draft Xing4_0 to
  ``DeepseekV3ForCausalLMNextN`` with ``num_nextn_predict_layers=1``
  (bundled DeepSeek-V3-compatible MTP head). Installed as a BEFORE hook so
  the remap happens at the same point in the flow as the in-tree patch and
  the original method's downstream NextN handling sees the remapped arch.
* ``ModelConfig._derive_model_shapes`` derives MLA shapes (head_dim=256) for
  the target model and restores ``hc_hidden_size = hidden_size * hc_mult``
  when ``hc_contract_for_draft`` is set. Installed as an AROUND hook that
  fixes up after the original; nothing between the in-tree patch points and
  the end of the method reads those attributes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ARCHITECTURE = "Xing4_0ForCausalLM"
DRAFT_ARCHITECTURE = "DeepseekV3ForCausalLMNextN"

_MODEL_CONFIG_MODULE = "sglang.srt.configs.model_config"


def install() -> list[str]:
    """Register the ModelConfig compat items; returns installed item names."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from sglang_xing4_0_plugin.compat import compat_enabled

    installed: list[str] = []

    if compat_enabled("DSA"):
        HookRegistry.register(
            f"{_MODEL_CONFIG_MODULE}.is_deepseek_dsa",
            _is_deepseek_dsa_around,
            HookType.AROUND,
        )
        installed.append("DSA")

    if compat_enabled("DRAFT_REMAP"):
        HookRegistry.register(
            f"{_MODEL_CONFIG_MODULE}.ModelConfig._config_draft_model",
            _config_draft_model_before,
            HookType.BEFORE,
        )
        installed.append("DRAFT_REMAP")

    if compat_enabled("MODEL_SHAPES"):
        HookRegistry.register(
            f"{_MODEL_CONFIG_MODULE}.ModelConfig._derive_model_shapes",
            _derive_model_shapes_around,
            HookType.AROUND,
        )
        installed.append("MODEL_SHAPES")

    return installed


def _is_deepseek_dsa_around(original_fn, config, *args, **kwargs):
    """AROUND is_deepseek_dsa: Xing4_0 with index_topk counts as DSA."""
    if original_fn(config, *args, **kwargs):
        return True
    from sglang.srt.configs.model_config import _hf_arch, _hf_attr

    return (
        _hf_arch(config) == ARCHITECTURE
        and _hf_attr(config, "index_topk") is not None
    )


def _config_draft_model_before(self, *args, **kwargs):
    """BEFORE ModelConfig._config_draft_model: remap draft Xing4_0 to NextN.

    Returning None leaves the call arguments untouched; the remap mutates
    ``self.hf_config`` in place, exactly like the in-tree patch. On an
    in-tree-aware build the original's own Xing4_0 check then no-ops.
    """
    if not getattr(self, "is_draft_model", False):
        return None
    hf_config = getattr(self, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if architectures and architectures[0] == ARCHITECTURE:
        architectures[0] = DRAFT_ARCHITECTURE
        # The bundled MTP head is a single layer; force it for checkpoints
        # that ship num_nextn_predict_layers=0/None.
        hf_config.num_nextn_predict_layers = 1
        logger.info(
            "sglang-xing4_0-plugin: draft Xing4_0 remapped to %s "
            "(bundled MTP head, num_nextn_predict_layers=1)",
            DRAFT_ARCHITECTURE,
        )
    return None


def _derive_model_shapes_around(original_fn, self, *args, **kwargs):
    """AROUND ModelConfig._derive_model_shapes: Xing4_0 MLA + mHC fixups."""
    result = original_fn(self, *args, **kwargs)
    architectures = (
        getattr(getattr(self, "hf_config", None), "architectures", None) or []
    )
    if ARCHITECTURE not in architectures:
        return result
    _apply_mla_shapes(self)
    _apply_hc_hidden_size(self)
    return result


def _apply_mla_shapes(self) -> None:
    """Replicate the in-tree MLA branch of _derive_model_shapes.

    On pristine sglang an unknown arch falls through to the MHA else-branch;
    Xing4_0 is MLA with head_dim=256. Skipped when MLA shapes were already
    derived (an in-tree build, or the draft model after the NextN remap).
    """
    from sglang.srt.configs.model_config import (
        AttentionArch,
        get_dsa_index_head_dim,
        is_deepseek_dsa,
    )

    if getattr(self, "attention_arch", None) == AttentionArch.MLA:
        return
    text_config = self.hf_text_config
    self.head_dim = 256
    self.attention_arch = AttentionArch.MLA
    self.kv_lora_rank = text_config.kv_lora_rank
    self.qk_nope_head_dim = text_config.qk_nope_head_dim
    self.qk_rope_head_dim = text_config.qk_rope_head_dim
    self.v_head_dim = text_config.v_head_dim
    self.swa_kv_lora_rank = self.kv_lora_rank
    self.swa_qk_rope_head_dim = self.qk_rope_head_dim
    self.index_head_dim = (
        get_dsa_index_head_dim(text_config)
        if is_deepseek_dsa(text_config)
        else None
    )
    rope_scaling = getattr(text_config, "rope_parameters", None) or getattr(
        text_config, "rope_scaling", None
    )
    self._init_mla_scaling(rope_scaling)


def _apply_hc_hidden_size(self) -> None:
    """Replicate the in-tree hc_contract_for_draft restore.

    Models that declare ``hc_contract_for_draft`` (Xing4_0) keep
    ``spec_hidden_size`` contracted but still carry an mHC-internal residual
    stream of ``hidden_size * hc_mult`` used to size activation buffers.
    """
    text_config = self.hf_text_config
    hc_mult = getattr(text_config, "hc_mult", 1)
    if hc_mult > 1 and getattr(text_config, "hc_contract_for_draft", False):
        self.hc_hidden_size = self.hidden_size * hc_mult
