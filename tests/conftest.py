"""Pytest configuration: Hypothesis profiles and automatic skipping of toolchain tests."""

import os

import pytest
from hypothesis import HealthCheck, settings

from superopt import config

# The default profile keeps the fast suite well under two minutes. The thorough profile is
# for local runs: HYPOTHESIS_PROFILE=thorough make test
settings.register_profile(
    "fast",
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile(
    "thorough",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests marked `toolchain` when LLVM or alive-tv is not configured."""
    if config.toolchain_available():
        return
    skip_marker = pytest.mark.skip(reason="LLVM toolchain is not configured")
    for item in items:
        if "toolchain" in item.keywords:
            item.add_marker(skip_marker)
