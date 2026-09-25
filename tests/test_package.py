"""The package installs, the Rust engine loads, and the CLI starts."""

import pytest

import hazres
from hazres.cli import main


def test_rust_engine_loads():
    assert hazres.engine_version() == hazres.__version__


def test_cli_help_runs(capsys):
    assert main([]) == 0
    assert "hazres" in capsys.readouterr().out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"engine {hazres.__version__}" in capsys.readouterr().out
