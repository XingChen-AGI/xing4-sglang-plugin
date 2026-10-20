# SPDX-License-Identifier: Apache-2.0
"""Vendored ``mhc_pre`` with splitk support for hc_hidden_size=14336.

Xing4_0 (hidden_size=3584, hc_mult=4 -> hc_hidden_size=14336) needs the
TileLang splitk path in ``mhc_pre``; pristine sglang only accepts
hc_hidden_size in {16384, 28672}. The function below is vendored verbatim
from sglang commit ``aa737beda5`` (branch ``support_xing4_0``, upstream PR
https://github.com/sgl-project/sglang/pull/39793); the only delta vs
pristine is the ``hc_hidden_size == 14336`` branch. Delete this module once
PR#39793 merges.

Installed as a HookRegistry REPLACE on
``sglang.kernels.ops.layernorm.mhc.mhc_pre``. When the installed sglang
already carries the 14336 branch (in-tree build), installation is skipped so
an evolved in-tree implementation is never clobbered by the vendored copy.
"""

from __future__ import annotations

import inspect
import logging

import torch
from sglang.kernels.ops.layernorm.mhc import (
    _compute_num_split_for_mhc_pre,
    _mhc_pre_gemm_sqrsum_dispatch,
    _use_deep_gemm_hc_prenorm,
    mhc_pre_big_fuse_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang,
    mhc_pre_gemm_sqrsum_splitk_kernel,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_MHC_MODULE = "sglang.kernels.ops.layernorm.mhc"
_TARGET = f"{_MHC_MODULE}.mhc_pre"


def install() -> list[str]:
    """Register the vendored mhc_pre REPLACE; returns installed item names."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from sglang_xing4_0_plugin.compat import compat_enabled

    if not compat_enabled("MHC_PRE"):
        return []

    import sglang.kernels.ops.layernorm.mhc as mhc_mod

    current = getattr(mhc_mod, "mhc_pre", None)
    if current is None:
        logger.warning(
            "sglang-xing4_0-plugin: %s not found; skipping MHC_PRE compat "
            "(renamed upstream?)",
            _TARGET,
        )
        return []
    if _already_installed(current):
        return []

    HookRegistry.register(_TARGET, mhc_pre, HookType.REPLACE)
    return ["MHC_PRE"]


def _already_installed(current) -> bool:
    """True when the live mhc_pre already handles hc_hidden_size=14336."""
    # Our own replacement (REPLACE wrapper delegates to the marked hook).
    for candidate in (current, getattr(current, "__wrapped__", None)):
        if getattr(candidate, "_xing4_0_plugin_vendored", False):
            return True
    # An in-tree build that already merged the 14336 branch.
    try:
        return "14336" in inspect.getsource(current)
    except (OSError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Vendored from sglang aa737beda5: python/sglang/kernels/ops/layernorm/mhc.py
# (delta vs pristine: the hc_hidden_size == 14336 splitk branch).
# ---------------------------------------------------------------------------


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    # layer_input is the post-norm activation fed into the MoE. Allocate it in
    # the symmetric memory pool so the downstream all-reduce uses the low-latency
    # NCCL symmetric path: the Triton inplace MoE runner writes the expert
    # output back into this buffer, so a symmetric input yields a symmetric
    # all-reduce input.
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        layer_input = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        )

    if _use_deep_gemm_hc_prenorm():
        n_splits = _compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)

        gemm_out_mul = torch.empty(
            n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )

        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

        tf32_hc_prenorm_gemm(
            residual_flat.view(num_tokens, hc_hidden_size),
            fn_flat,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
        gemm_last_dim = hc_mult3
        big_fuse_n_splits = n_splits
    else:
        if num_tokens <= 2048:
            assert n_splits == 1
            if hc_hidden_size == 16384:
                hidden_block = 256
            elif hc_hidden_size == 28672:
                hidden_block = 128
            elif hc_hidden_size == 14336:
                # 14336 = 4 * 3584 (xing4_0). hidden_block=256 -> 14336/256=56.
                # Caller must pass n_splits_pre dividing 56 so that
                # (14336//n_splits_pre) % 256 == 0 (e.g. n_splits_pre=8 -> 1792).
                hidden_block = 256
            else:
                raise NotImplementedError(
                    f"mhc_pre splitk kernel only supports hc_hidden_size in {{16384, 28672, 14336}}, "
                    f"got {hc_hidden_size}"
                )
            kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(
                hc_mult3,
                hc_hidden_size,
                split_k=n_splits_pre,
                token_block=32,
                hidden_block=hidden_block,
            )
            partial_out = torch.empty(
                n_splits_pre,
                num_tokens,
                32,
                dtype=torch.float32,
                device=residual.device,
            )
            partial_sqrsum = torch.empty(
                n_splits_pre, num_tokens, dtype=torch.float32, device=residual.device
            )
            kernel_0(
                residual_flat.view(num_tokens, hc_hidden_size),
                fn_flat,
                partial_out,
                partial_sqrsum,
            )
            # Stage_1 reduction is folded into big_fuse below; skip launching it.
            gemm_out_mul = partial_out
            gemm_out_sqrsum = partial_sqrsum
            gemm_last_dim = 32
            big_fuse_n_splits = n_splits_pre
        else:
            gemm_out_mul = torch.empty(
                n_splits,
                num_tokens,
                hc_mult3,
                dtype=torch.float32,
                device=residual.device,
            )
            gemm_out_sqrsum = torch.empty(
                n_splits, num_tokens, dtype=torch.float32, device=residual.device
            )
            assert n_splits == 1, (
                "The simple TileLang version gemm_sqrsum doesn't support split-k"
            )
            _mhc_pre_gemm_sqrsum_dispatch()(
                residual_flat.view(num_tokens, hc_mult * hidden_size),
                fn_flat,
                gemm_out_mul.squeeze(0),
                gemm_out_sqrsum.squeeze(0),
                hc_mult3,
                hc_mult * hidden_size,
            )
            gemm_last_dim = hc_mult3
            big_fuse_n_splits = n_splits

    if norm_weight is not None:
        assert norm_eps is not None, "norm_eps required when norm_weight is provided"
        assert norm_weight.shape == (hidden_size,), (
            f"norm_weight shape {tuple(norm_weight.shape)} != (hidden_size={hidden_size},)"
        )
        norm_weight_bf = (
            norm_weight.bfloat16()
            if norm_weight.dtype != torch.bfloat16
            else norm_weight
        )
        if not norm_weight_bf.is_contiguous():
            norm_weight_bf = norm_weight_bf.contiguous()
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight_bf,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )
    else:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


# Marker consulted by _already_installed for idempotent re-runs.
mhc_pre._xing4_0_plugin_vendored = True  # type: ignore[attr-defined]
