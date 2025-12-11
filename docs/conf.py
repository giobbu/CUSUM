# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # So autodoc can find your modules

# -- Project information -----------------------------------------------------
project = 'CUSUM'
author = 'giovanni buroni'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True  # Automatically generate summary tables

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # RTD theme
html_static_path = ['_static']

