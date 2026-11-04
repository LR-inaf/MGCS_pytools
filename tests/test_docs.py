"""Unit tests for offline documentation locator, viewer, and builder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mgcs_pytools
from mgcs_pytools.docs import build_docs, get_docs_path, open_docs, open_docs_cli


def test_get_docs_path_exists():
    """Verify that get_docs_path locates the compiled HTML documentation index."""
    doc_path = get_docs_path()
    assert doc_path is not None
    assert isinstance(doc_path, Path)
    assert doc_path.name == "index.html"
    assert doc_path.is_file()


def test_package_level_export():
    """Verify that open_docs and get_docs_path are accessible directly from mgcs_pytools."""
    assert callable(mgcs_pytools.open_docs)
    assert callable(mgcs_pytools.get_docs_path)
    assert mgcs_pytools.get_docs_path() == get_docs_path()


def test_open_docs_success():
    """Verify open_docs opens the file URI using webbrowser without raising exceptions."""
    with patch("webbrowser.open", return_value=True) as mock_open:
        success = open_docs()
        assert success is True
        assert mock_open.called
        call_arg = mock_open.call_args[0][0]
        assert call_arg.startswith("file://")
        assert "index.html" in call_arg


def test_open_docs_with_custom_browser():
    """Verify open_docs handles a specific browser name."""
    mock_controller = MagicMock()
    with patch("webbrowser.get", return_value=mock_controller) as mock_get:
        success = open_docs(browser="firefox")
        assert success is True
        assert mock_get.called
        assert mock_controller.open.called


def test_open_docs_not_found():
    """Verify open_docs fails gracefully when documentation is not found."""
    with patch("mgcs_pytools.docs.get_docs_path", return_value=None):
        with patch("sys.stderr.write") as mock_stderr:
            success = open_docs()
            assert success is False
            assert mock_stderr.called


def test_cli_path_flag(capsys):
    """Verify that the CLI with --path flag outputs the path to index.html and exits 0."""
    with patch("sys.argv", ["mgcs-docs", "--path"]):
        with pytest.raises(SystemExit) as exc_info:
            open_docs_cli()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "index.html" in captured.out


def test_build_docs_mocked():
    """Verify build_docs runs subprocess with expected arguments."""
    mock_res = MagicMock()
    mock_res.returncode = 0
    with patch("subprocess.run", return_value=mock_res) as mock_run:
        success = build_docs()
        assert success is True
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "sphinx" in cmd


def test_cli_build_flag():
    """Verify that the CLI with --build flag invokes build_docs and exits 0."""
    with patch("sys.argv", ["mgcs-docs", "--build"]):
        with patch("mgcs_pytools.docs.build_docs", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                open_docs_cli()
            assert exc_info.value.code == 0


def test_cli_open_browser():
    """Verify that the CLI without arguments attempts to launch browser and exits 0."""
    with patch("sys.argv", ["mgcs-docs"]):
        with patch("mgcs_pytools.docs.open_docs", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                open_docs_cli()
            assert exc_info.value.code == 0
