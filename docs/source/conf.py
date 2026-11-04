"""Sphinx configuration for MGCS_pytools documentation."""

import os
import sys
from datetime import datetime

# Ensure the package is importable
sys.path.insert(0, os.path.abspath("../../src"))

# Project information
project = "MGCS_pytools"
copyright = f"{datetime.now().year}, Luca Rosignoli"
author = "Luca Rosignoli"
release = "0.3.0"
version = "0.3.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
]

# Source suffix configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon configuration (NumPy style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# HTML output configuration (ReadTheDocs theme, offline-friendly)
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
}

html_static_path = ["_static"]
html_show_sourcelink = True
html_copy_source = True
