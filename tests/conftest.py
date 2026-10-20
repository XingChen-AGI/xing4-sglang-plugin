# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for sglang_xing4_0_plugin tests.

Importing sglang.srt.models pulls in sgl_kernel -> flashinfer, whose JIT
cubin version check raises on mismatched flashinfer-cubin/flashinfer
versions (seen in the reference env: 0.6.18 vs 0.6.15.post1). The same
variable is required when launching `sglang serve` on such environments,
so default it here before any sglang import happens.
"""

from __future__ import annotations

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
