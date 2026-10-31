"""Skipped placeholders for the core acceptance tests in spec section 4.4."""

import pytest


@pytest.mark.skip(reason="not implemented")
def test_boost_hop_cost_equivalence():
    raise NotImplementedError


@pytest.mark.skip(reason="not implemented")
def test_toy_chain_and_grid_costs():
    raise NotImplementedError


@pytest.mark.skip(reason="not implemented")
def test_lcb_closed_form_matches_beta_quantile():
    raise NotImplementedError


@pytest.mark.skip(reason="not implemented")
def test_artifact_hash_is_deterministic():
    raise NotImplementedError


@pytest.mark.skip(reason="not implemented")
def test_numpy_126_and_2_matrix():
    raise NotImplementedError
