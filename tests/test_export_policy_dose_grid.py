"""NN-0 / Policy-1 dose-grid case construction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_slip_dataset import (  # noqa: E402
    POLICY_SKIP_BASES,
    POLICY_TEST_CASES,
    _base_case_name,
    build_cases,
    build_policy_cases,
)


def test_policy_enrichment_names_not_collapsed_to_div2():
    """mass×μ crosses must keep distinct base names (train-eligible)."""
    names = {_base_case_name(c.name) for c in build_policy_cases(include_variants=False)}
    assert "mass_x2_friction_div2" in names
    assert "friction_s060" in names
    assert "friction_div2" in names  # still present for OOD test
    assert names & POLICY_TEST_CASES == {"friction_div2"}


def test_policy_cases_skip_fail_heavy_bases():
    bases = {_base_case_name(c.name) for c in build_policy_cases(include_variants=True)}
    assert bases.isdisjoint(POLICY_SKIP_BASES)


def test_nn0_build_cases_includes_enrichment_by_default():
    bases = {_base_case_name(c.name) for c in build_cases(include_variants=False)}
    assert "friction_s055" in bases
    assert "mass_x4_friction_s060" in bases
    legacy = {_base_case_name(c.name) for c in build_cases(
        include_variants=False, include_policy_enrichment=False
    )}
    assert "friction_s055" not in legacy
