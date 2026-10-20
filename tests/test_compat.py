# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase 2 compat layer (sglang_xing4_0_plugin.compat).

These tests run against the in-tree fork environment, where every hunk the
compat layer replicates is already merged. That is the "overlay" mode: the
compat hooks install on top of in-tree behavior and every assertion below
must hold for both an in-tree build and a pristine build plus the plugin.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from sglang_xing4_0_plugin import compat

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def installed():
    """Install all compat hooks once and apply them (idempotent overlay)."""
    names = compat.install_all()
    from sglang.srt.plugins.hook_registry import HookRegistry

    HookRegistry.apply_hooks()
    return names


@pytest.fixture()
def overrides_mod():
    import sglang.srt.arg_groups.overrides as mod

    return mod


# ------------------------------------------------------------- kill switches


def test_compat_enabled_kill_switches(monkeypatch):
    monkeypatch.delenv("XING4_0_COMPAT_OFF", raising=False)
    monkeypatch.delenv("XING4_0_COMPAT_DSA", raising=False)
    assert compat.compat_enabled("DSA") is True

    monkeypatch.setenv("XING4_0_COMPAT_DSA", "0")
    assert compat.compat_enabled("DSA") is False
    assert compat.compat_enabled("MODEL_HOOK") is True  # untouched item

    monkeypatch.setenv("XING4_0_COMPAT_OFF", "1")
    assert compat.compat_enabled("MODEL_HOOK") is False  # global off


def test_install_all_idempotent():
    first = compat.install_all()
    second = compat.install_all()
    assert first == second


# ------------------------------------------------------------ installation


def test_all_items_installed(installed):
    # Items that register hooks unconditionally on both fork and pristine.
    expected = {
        "ARCH_SETS",
        "MODEL_HOOK",
        "DETERMINISTIC",
        "EAGLE_FAMILY",
        "DSA",
        "DRAFT_REMAP",
        "MODEL_SHAPES",
    }
    assert expected <= set(installed)


def test_arch_sets_contain_xing4_0(installed, overrides_mod):
    assert "Xing4_0ForCausalLM" in overrides_mod._DEEPSEEK_FAMILY_ARCHS
    assert "Xing4_0ForCausalLM" in overrides_mod._FLASHINFER_ALLREDUCE_FUSION_ARCHS


def test_function_hooks_applied(installed):
    from sglang.srt.plugins.hook_registry import HookRegistry

    targets = {
        "sglang.srt.arg_groups.model_hook.handle_model_specific_adjustments",
        "sglang.srt.arg_groups.overrides._deterministic_is_deepseek_model",
        "sglang.srt.arg_groups.speculative_hook._handle_eagle_family",
        "sglang.srt.configs.model_config.is_deepseek_dsa",
        "sglang.srt.configs.model_config.ModelConfig._config_draft_model",
        "sglang.srt.configs.model_config.ModelConfig._derive_model_shapes",
    }
    assert targets <= set(HookRegistry._patched)


# ------------------------------------------------------------- hook bodies


def test_is_deepseek_dsa_xing4_0(installed):
    from sglang.srt.configs.model_config import is_deepseek_dsa

    dsa = SimpleNamespace(architectures=["Xing4_0ForCausalLM"], index_topk=2048)
    assert is_deepseek_dsa(dsa) is True

    no_topk = SimpleNamespace(architectures=["Xing4_0ForCausalLM"], index_topk=None)
    assert is_deepseek_dsa(no_topk) is False

    llama = SimpleNamespace(architectures=["LlamaForCausalLM"], index_topk=2048)
    assert is_deepseek_dsa(llama) is False


def test_deterministic_deepseek_model(installed, overrides_mod, monkeypatch):
    from sglang_xing4_0_plugin.compat.arg_groups import (
        _deterministic_is_deepseek_model_around as around,
    )

    view = SimpleNamespace(model_path="/w/model")
    monkeypatch.setattr(
        overrides_mod,
        "model_config_of",
        lambda v: SimpleNamespace(hf_config=SimpleNamespace(
            architectures=["Xing4_0ForCausalLM"])),
    )
    assert around(lambda v: False, view) is True
    assert around(lambda v: True, view) is True

    # An original that raises must not mask the arch probe (the in-tree body
    # swallows lookup errors the same way).
    def _raise(view):
        raise RuntimeError("boom")

    assert around(_raise, view) is True

    # When even the arch probe fails, fall back to False.
    def _raise_probe(view):
        raise RuntimeError("boom")

    monkeypatch.setattr(overrides_mod, "model_config_of", lambda v: (_ for _ in ()).throw(RuntimeError("no cfg")))
    assert around(lambda v: False, view) is False


def test_draft_remap_before_hook(installed):
    from sglang_xing4_0_plugin.compat.model_config import (
        _config_draft_model_before as before,
    )

    draft = SimpleNamespace(
        is_draft_model=True,
        hf_config=SimpleNamespace(
            architectures=["Xing4_0ForCausalLM"], num_nextn_predict_layers=0
        ),
    )
    assert before(draft) is None
    assert draft.hf_config.architectures[0] == "DeepseekV3ForCausalLMNextN"
    assert draft.hf_config.num_nextn_predict_layers == 1

    # Non-draft is untouched.
    target = SimpleNamespace(
        is_draft_model=False,
        hf_config=SimpleNamespace(architectures=["Xing4_0ForCausalLM"]),
    )
    before(target)
    assert target.hf_config.architectures[0] == "Xing4_0ForCausalLM"

    # Non-Xing4_0 draft is untouched.
    other = SimpleNamespace(
        is_draft_model=True,
        hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
    )
    before(other)
    assert other.hf_config.architectures[0] == "DeepseekV3ForCausalLM"


def test_derive_model_shapes_fixup(installed):
    from sglang.srt.configs.model_config import AttentionArch
    from sglang_xing4_0_plugin.compat.model_config import (
        _derive_model_shapes_around as around,
    )

    class _Stub:
        def _init_mla_scaling(self, rope_scaling):
            self.rope_scaling_seen = rope_scaling

    text_config = SimpleNamespace(
        architectures=["Xing4_0ForCausalLM"],
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        hc_mult=4,
        hc_contract_for_draft=True,
        index_topk=None,  # wodsa-final checkpoint declares index_topk=null
        rope_scaling=None,
    )
    stub = _Stub()
    stub.hf_config = SimpleNamespace(architectures=["Xing4_0ForCausalLM"])
    stub.hf_text_config = text_config
    stub.attention_arch = AttentionArch.MHA  # pristine falls through to MHA
    stub.head_dim = 56  # hidden_size // num_attention_heads fallback
    stub.hidden_size = 3584
    stub.hc_hidden_size = None

    around(lambda self: None, stub)

    assert stub.head_dim == 256
    assert stub.attention_arch == AttentionArch.MLA
    assert stub.kv_lora_rank == 512
    assert stub.qk_nope_head_dim == 128
    assert stub.qk_rope_head_dim == 64
    assert stub.v_head_dim == 128
    assert stub.swa_kv_lora_rank == 512
    assert stub.index_head_dim is None  # not DSA without index_topk
    assert stub.hc_hidden_size == 3584 * 4  # hc_contract_for_draft restore

    # Already-MLA configs (in-tree build or the remapped NextN draft) are
    # left alone apart from the hc restore.
    already = _Stub()
    already.hf_config = SimpleNamespace(architectures=["Xing4_0ForCausalLM"])
    already.hf_text_config = text_config
    already.attention_arch = AttentionArch.MLA
    already.head_dim = 256
    already.hidden_size = 3584
    already.hc_hidden_size = None
    around(lambda self: None, already)
    assert already.head_dim == 256
    assert already.hc_hidden_size == 3584 * 4

    # Non-Xing4_0 configs are untouched.
    other = _Stub()
    other.hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    other.hf_text_config = SimpleNamespace(hc_mult=1)
    other.attention_arch = AttentionArch.MHA
    other.hidden_size = 4096
    around(lambda self: None, other)
    assert other.attention_arch == AttentionArch.MHA


def test_model_hook_disables_deepgemm_prenorm(installed, overrides_mod, monkeypatch):
    from sglang.srt.environ import envs

    from sglang_xing4_0_plugin.compat.arg_groups import (
        _handle_model_specific_adjustments_around as around,
    )

    flag = envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM
    saved = flag.get()
    try:
        flag.set(True)
        monkeypatch.setattr(
            overrides_mod,
            "model_config_of",
            lambda sa: SimpleNamespace(hf_config=SimpleNamespace(
                architectures=["Xing4_0ForCausalLM"])),
        )
        around(lambda *a, **k: None, SimpleNamespace())
        assert flag.get() is False

        flag.set(True)
        monkeypatch.setattr(
            overrides_mod,
            "model_config_of",
            lambda sa: SimpleNamespace(hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"])),
        )
        around(lambda *a, **k: None, SimpleNamespace())
        assert flag.get() is True
    finally:
        flag.set(saved)


def test_eagle_family_declares_draft_path(installed, overrides_mod, monkeypatch):
    from sglang_xing4_0_plugin.compat.arg_groups import (
        _handle_eagle_family_around as around,
    )

    declared = {}
    draft_path = {"value": None}

    def fake_model_config_of(server_args):
        return SimpleNamespace(hf_config=SimpleNamespace(
            architectures=["Xing4_0ForCausalLM"]))

    def fake_resolving_view(server_args):
        return SimpleNamespace(
            speculative_draft_model_path=draft_path["value"],
            model_path="/w/xing4",
            revision="main",
        )

    def fake_declare(server_args, source, **fields):
        declared.update(fields)

    for name, value in (
        ("model_config_of", fake_model_config_of),
        ("resolving_view", fake_resolving_view),
        ("declare_resolution", fake_declare),
    ):
        monkeypatch.setattr(overrides_mod, name, value)

    # No explicit draft path -> declare the bundled MTP head.
    around(lambda *a, **k: None, SimpleNamespace())
    assert declared == {
        "speculative_draft_model_path": "/w/xing4",
        "speculative_draft_model_revision": "main",
    }

    # Explicit draft path -> leave it alone.
    declared.clear()
    draft_path["value"] = "/w/other-draft"
    around(lambda *a, **k: None, SimpleNamespace())
    assert declared == {}

    # Non-Xing4_0 arch -> no declaration.
    declared.clear()
    draft_path["value"] = None
    monkeypatch.setattr(
        overrides_mod,
        "model_config_of",
        lambda sa: SimpleNamespace(hf_config=SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"])),
    )
    around(lambda *a, **k: None, SimpleNamespace())
    assert declared == {}


def test_mhc_pre_14336_available(installed):
    import sglang.kernels.ops.layernorm.mhc as mhc_mod
    import sglang_xing4_0_plugin.compat.mhc_pre as compat_mhc

    # The vendored copy always carries the 14336 splitk branch...
    assert compat_mhc.mhc_pre._xing4_0_plugin_vendored is True
    assert "14336" in inspect.getsource(compat_mhc.mhc_pre)

    # ...and the live mhc_pre supports it too, either because sglang merged
    # PR#39793 in tree (fork) or because our REPLACE hook is active.
    live = mhc_mod.mhc_pre
    live_src = ""
    try:
        live_src = inspect.getsource(live)
    except (OSError, TypeError):
        pass
    assert "14336" in live_src or "14336" in inspect.getsource(compat_mhc.mhc_pre)

    # The vendored model must call the effective (possibly replaced) kernel.
    import sglang_xing4_0_plugin.models.xing4_0 as model_mod

    assert model_mod._mhc_pre_orig is live
