# SPDX-License-Identifier: Apache-2.0
"""Transitional compat hooks for Xing4_0 on pristine sglang builds.

The in-tree ``support_xing4_0`` patch (upstream PR
https://github.com/sgl-project/sglang/pull/39793, vendored baseline commit
``aa737beda5``) touches a handful of core-resolution spots beyond the model
class itself: server-arg arch lists, ``ModelConfig`` draft remapping and MLA
shape derivation, and the ``mhc_pre`` splitk kernel. Until that PR merges,
the modules in this package install HookRegistry hooks (plus one direct
module-attribute rebind for arch frozensets) that replicate those hunks on a
pristine sglang. Each item is idempotent, so running on top of an
in-tree-aware build (e.g. the vendored fork with ``XING4_0_PLUGIN_OVERRIDE=1``)
is safe. Delete items one by one as PR#39793 merges; when everything is in
tree, :func:`sglang_xing4_0_plugin.register` stands down and none of this
runs.

Kill switches (checked once at install time, not per call):

* ``XING4_0_COMPAT_OFF=1`` disables the whole compat layer.
* ``XING4_0_COMPAT_<ITEM>=0`` disables a single item, where ``<ITEM>`` is one
  of: ``ARCH_SETS``, ``MODEL_HOOK``, ``DETERMINISTIC``, ``EAGLE_FAMILY``,
  ``DSA``, ``DRAFT_REMAP``, ``MODEL_SHAPES``, ``MHC_PRE``.
"""

from __future__ import annotations

import logging
import os

__all__ = ["compat_enabled", "install_all"]

logger = logging.getLogger(__name__)

_COMPAT_OFF_ENV = "XING4_0_COMPAT_OFF"
_ITEM_ENV_PREFIX = "XING4_0_COMPAT_"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

# Items installed in this process (install_all is idempotent).
_INSTALLED: set[str] = set()


def compat_enabled(item: str) -> bool:
    """Whether compat item *item* should install. Default: enabled."""
    if os.environ.get(_COMPAT_OFF_ENV, "").strip().lower() in _TRUE:
        return False
    value = os.environ.get(_ITEM_ENV_PREFIX + item, "").strip().lower()
    return value not in _FALSE


def install_all() -> list[str]:
    """Install every enabled compat hook. Returns the installed item names.

    Safe to call more than once per process; the second call is a no-op that
    returns the previously installed set. Hooks registered here are applied
    later by ``HookRegistry.apply_hooks()`` (called at the end of sglang's
    ``load_plugins()``), except the arch frozenset rebind, which takes effect
    immediately.
    """
    if _INSTALLED:
        return sorted(_INSTALLED)

    installed: list[str] = []
    from sglang_xing4_0_plugin.compat import arg_groups, mhc_pre, model_config, serving

    for mod in (arg_groups, model_config, mhc_pre, serving):
        installed.extend(mod.install())
    _INSTALLED.update(installed)

    if installed:
        logger.info(
            "sglang-xing4_0-plugin: compat hooks installed: %s "
            "(vendored from sglang PR#39793; remove as it merges)",
            ", ".join(sorted(installed)),
        )
    else:
        logger.info(
            "sglang-xing4_0-plugin: no compat hooks installed "
            "(all disabled or already provided in tree)"
        )
    return sorted(_INSTALLED)
