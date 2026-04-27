# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # So autodoc can find your modules

# -- Project information -----------------------------------------------------
project = 'CUSUM'
author = 'giovanni buroni'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

autosummary_generate = True  # Automatically generate summary tables
autoclass_content = "both"  # Include both class docstring and __init__ docstring

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon"
]
autosummary_generate = True  # Automatically generate summary tables


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"



html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/giobbu/CUSUM",
    "source_branch": "main",  # or "master"
    "source_directory": "docs/",  # path to your docs folder
}
html_static_path = ['_static']
html_css_files = ["custom.css"]


