"""Offline documentation viewer, locator, and builder for MGCS_pytools."""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from importlib import resources
from pathlib import Path


def get_docs_path() -> Path | None:
    """Locate the root index.html of the offline HTML documentation.

    The function searches:
    1. Package data directory: ``mgcs_pytools/_docs_html/index.html``.
    2. Local source development tree: ``docs/build/html/index.html`` or ``docs/_build/html/index.html``.

    Returns
    -------
    pathlib.Path or None
        Absolute path to ``index.html`` if found, None otherwise.
    """
    # 1. Check packaged docs via importlib.resources
    try:
        traversable = resources.files("mgcs_pytools") / "_docs_html" / "index.html"
        pkg_path = Path(str(traversable))
        if pkg_path.is_file():
            return pkg_path.resolve()
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    # Direct relative check to package directory
    direct_pkg_path = Path(__file__).resolve().parent / "_docs_html" / "index.html"
    if direct_pkg_path.is_file():
        return direct_pkg_path

    # 2. Check local repository development builds
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "src" / "mgcs_pytools" / "_docs_html" / "index.html",
        repo_root / "docs" / "build" / "html" / "index.html",
        repo_root / "docs" / "_build" / "html" / "index.html",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    return None


def build_docs() -> bool:
    """Build the offline HTML documentation using Sphinx.

    Outputs compiled HTML to ``src/mgcs_pytools/_docs_html``.

    Returns
    -------
    bool
        True if the build succeeded, False otherwise.
    """
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "docs" / "source"
    output_dir = Path(__file__).resolve().parent / "_docs_html"
    doctrees_dir = repo_root / "docs" / "build" / "doctrees"

    if not source_dir.exists():
        sys.stderr.write(f"Sphinx documentation source directory not found at: {source_dir}\n")
        return False

    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        "-d",
        str(doctrees_dir),
        str(source_dir),
        str(output_dir),
    ]
    print(f"Building Sphinx HTML documentation into: {output_dir}")
    res = subprocess.run(cmd)
    return res.returncode == 0


def open_docs(browser: str | None = None) -> bool:
    """Open the offline ReadTheDocs HTML documentation in a web browser.

    Parameters
    ----------
    browser : str or None, optional
        Browser name to open (e.g. ``'firefox'``, ``'chrome'``). If None, the system
        default web browser is used. Default is None.

    Returns
    -------
    bool
        True if the documentation was located and opened, False otherwise.
    """
    doc_path = get_docs_path()
    if doc_path is None or not doc_path.exists():
        sys.stderr.write(
            "Offline documentation not found in package data.\n"
            "To build the documentation locally, run:\n"
            "    python -m mgcs_pytools.docs --build\n"
        )
        return False

    uri = doc_path.as_uri()
    print(f"Opening offline documentation: {uri}")
    try:
        controller = webbrowser.get(browser) if browser else webbrowser
        controller.open(uri)
        return True
    except Exception as exc:
        sys.stderr.write(f"Failed to launch web browser ({exc}). You can open:\n{uri}\n")
        return False


def open_docs_cli() -> None:
    """Entry point for the ``mgcs-docs`` console script and ``python -m mgcs_pytools.docs``."""
    parser = argparse.ArgumentParser(
        prog="mgcs-docs",
        description="Open the offline ReadTheDocs HTML documentation for MGCS_pytools.",
    )
    parser.add_argument(
        "--path",
        action="store_true",
        help="Print the file path to index.html instead of launching a browser.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the offline Sphinx HTML documentation.",
    )
    parser.add_argument(
        "--browser",
        type=str,
        default=None,
        help="Specify a browser name (default: system default browser).",
    )
    args = parser.parse_args()

    if args.build:
        success = build_docs()
        sys.exit(0 if success else 1)

    if args.path:
        p = get_docs_path()
        if p and p.exists():
            print(str(p))
            sys.exit(0)
        else:
            sys.stderr.write("Offline documentation not found.\n")
            sys.exit(1)

    success = open_docs(browser=args.browser)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    open_docs_cli()
