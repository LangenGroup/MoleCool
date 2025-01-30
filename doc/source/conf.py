# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'MoleCool'
copyright = '2025, Felix Kogel'
author = 'Felix Kogel'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [ 'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'] 

# Add all modules that you use in your code and are not part of the standard
# python installation to the autodoc_mock_imports
# This prevents the import of those modules by Sphinx and thereby reduces
# the overhead and makes sure that the documentation can be generated on 
# machines where those modules are not present. 
autodoc_mock_imports=['numpy','matplotlib','scipy','sympy','numba','pandas','tqdm','h5py']

autoclass_content = "both"

# This value selects if automatically documented members are sorted alphabetical
# (value 'alphabetical'), by member type (value 'groupwise') or by source order
# (value 'bysource'). The default is alphabetical.
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' # 'sphinxdoc' 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- Examples.rst ------------------------------------------------------------

# Manually generating Examples.rst file for showing the source codes of all
# .py files located in the mymodules/Examples folder.

pathtoEx = r'../../MoleCool/Examples/'
files = os.listdir(pathtoEx)
with open('Examples.rst', 'w') as rstfile:
    for i,filename in enumerate(files):
        if filename[-3:] == '.py':
            rstfile.write(filename + '\n' + len(filename)*'=' + '\n\n' + '.. literalinclude:: ' + pathtoEx + filename + '\n\n')