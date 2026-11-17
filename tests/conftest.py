"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def isolated_data_file(tmp_path, monkeypatch):
    """No test may read or write the real data.json: the app saves to a throwaway file instead."""
    monkeypatch.setenv("PAWPAL_DATA_FILE", str(tmp_path / "pawpal_test_data.json"))
