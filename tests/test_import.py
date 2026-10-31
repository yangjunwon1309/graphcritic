"""Smoke test that the graphcritic core package imports."""


def test_graphcritic_imports():
    import graphcritic

    assert graphcritic is not None
