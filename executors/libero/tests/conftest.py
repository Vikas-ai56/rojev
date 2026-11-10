"""Test setup: the fake `libero` package (`fake_libero`) replaces LIBERO before any test imports LeRobot's LIBERO env,
so the plug-in runs on LeRobot's real env and processor code on any machine; every test gets its own LIBERO data root
holding the fake init states."""
import pytest

import fake_libero

fake_libero.install()


@pytest.fixture(autouse=True)
def libero_root(tmp_path, monkeypatch):
    fake_libero.write_init_states(tmp_path)
    monkeypatch.setattr(fake_libero, "ROOT", tmp_path)
    return tmp_path
