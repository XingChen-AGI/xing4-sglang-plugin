# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution compat hooks for Xing4_0.

Replicates the ``arg_groups`` hunks of the in-tree ``support_xing4_0`` patch
(upstream PR https://github.com/sgl-project/sglang/pull/39793, baseline
commit ``aa737beda5``):

* ``overrides._DEEPSEEK_FAMILY_ARCHS`` / ``_FLASHINFER_ALLREDUCE_FUSION_ARCHS``
  gain ``Xing4_0ForCausalLM`` (direct module-attribute rebind -- frozensets
  are data, not callables, so they cannot go through HookRegistry).
* ``model_hook.handle_model_specific_adjustments`` disables the DeepGEMM
  tf32 hc-prenorm path for Xing4_0 (torch.compile cannot trace the raw C
  extension; the TileLang splitk path supports hc_hidden_size=14336).
* ``overrides._deterministic_is_deepseek_model`` recognizes Xing4_0.
* ``speculative_hook._handle_eagle_family`` defaults the EAGLE draft path to
  the target checkpoint (Xing4_0 bundles its MTP head).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ARCHITECTURE = "Xing4_0ForCausalLM"

_OVERRIDES_MODULE = "sglang.srt.arg_groups.overrides"


def install() -> list[str]:
    """Register the arg-group compat items; returns installed item names."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from sglang_xing4_0_plugin.compat import compat_enabled

    installed: list[str] = []

    if compat_enabled("ARCH_SETS"):
        _install_arch_sets()
        installed.append("ARCH_SETS")

    if compat_enabled("MODEL_HOOK"):
        HookRegistry.register(
            "sglang.srt.arg_groups.model_hook.handle_model_specific_adjustments",
            _handle_model_specific_adjustments_around,
            HookType.AROUND,
        )
        installed.append("MODEL_HOOK")

    if compat_enabled("DETERMINISTIC"):
        HookRegistry.register(
            f"{_OVERRIDES_MODULE}._deterministic_is_deepseek_model",
            _deterministic_is_deepseek_model_around,
            HookType.AROUND,
        )
        installed.append("DETERMINISTIC")

    if compat_enabled("EAGLE_FAMILY"):
        HookRegistry.register(
            "sglang.srt.arg_groups.speculative_hook._handle_eagle_family",
            _handle_eagle_family_around,
            HookType.AROUND,
        )
        installed.append("EAGLE_FAMILY")

    return installed


def _install_arch_sets() -> None:
    """Add Xing4_0ForCausalLM to the two architecture frozensets.

    All consumers read the module global at call time from inside
    ``overrides.py`` itself, so rebinding the defining module is sufficient;
    the sys.modules sweep repairs any stale ``from`` import as belt and
    braces. Skipped when the arch is already present (in-tree build).
    """
    import sglang.srt.arg_groups.overrides as overrides_mod
    from sglang.srt.plugins.hook_registry import _propagate_patch

    for name in ("_DEEPSEEK_FAMILY_ARCHS", "_FLASHINFER_ALLREDUCE_FUSION_ARCHS"):
        current = getattr(overrides_mod, name, None)
        if current is None:
            logger.warning(
                "sglang-xing4_0-plugin: %s.%s not found; skipping "
                "(renamed upstream?)",
                _OVERRIDES_MODULE,
                name,
            )
            continue
        if ARCHITECTURE in current:
            continue
        patched = current | {ARCHITECTURE}
        setattr(overrides_mod, name, patched)
        _propagate_patch(current, patched, overrides_mod)
        logger.info(
            "sglang-xing4_0-plugin: added %s to %s.%s",
            ARCHITECTURE,
            _OVERRIDES_MODULE,
            name,
        )


def _handle_model_specific_adjustments_around(original_fn, server_args, *args, **kwargs):
    """AROUND handle_model_specific_adjustments: disable DeepGEMM hc-prenorm.

    The in-tree patch adds an ``elif model_arch in ["Xing4_0ForCausalLM"]``
    branch; running after the original is equivalent because the pristine
    elif chain never matches Xing4_0 and the explicit ``set(False)`` wins
    regardless of ordering.
    """
    result = original_fn(server_args, *args, **kwargs)
    try:
        from sglang.srt.arg_groups.overrides import model_config_of
        from sglang.srt.environ import envs

        arch = model_config_of(server_args).hf_config.architectures[0]
        if arch == ARCHITECTURE:
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
    except Exception:
        logger.debug(
            "sglang-xing4_0-plugin: model-hook compat skipped", exc_info=True
        )
    return result


def _deterministic_is_deepseek_model_around(original_fn, view, *args, **kwargs):
    """AROUND _deterministic_is_deepseek_model: Xing4_0 counts as DeepSeek."""
    # The in-tree body swallows lookup errors and returns False; keep that
    # contract for both the original call and the Xing4_0 probe.
    try:
        if original_fn(view, *args, **kwargs):
            return True
    except Exception:
        pass
    try:
        from sglang.srt.arg_groups.overrides import model_config_of

        hf_config = model_config_of(view).hf_config
        return hf_config.architectures[0] == ARCHITECTURE
    except Exception:
        return False


def _handle_eagle_family_around(original_fn, server_args, *args, **kwargs):
    """AROUND _handle_eagle_family: default the draft path to the checkpoint.

    The in-tree patch adds Xing4_0 to the bundled-MTP family list. Declaring
    *after* the original keeps precedence for in-tree-aware builds and for
    an explicit ``--speculative-draft-model-path``.
    """
    result = original_fn(server_args, *args, **kwargs)
    try:
        from sglang.srt.arg_groups.overrides import (
            declare_resolution,
            model_config_of,
            resolving_view,
        )

        arch = model_config_of(server_args).hf_config.architectures[0]
        if arch != ARCHITECTURE:
            return result
        cfg = resolving_view(server_args)
        if cfg.speculative_draft_model_path is None:
            declare_resolution(
                server_args,
                "_handle_eagle_family",
                speculative_draft_model_path=cfg.model_path,
                speculative_draft_model_revision=cfg.revision,
            )
            logger.info(
                "sglang-xing4_0-plugin: Xing4_0 bundles its MTP head; "
                "defaulting speculative draft model to the target checkpoint."
            )
    except Exception:
        logger.debug(
            "sglang-xing4_0-plugin: eagle-family compat skipped", exc_info=True
        )
    return result
