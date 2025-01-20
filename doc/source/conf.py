# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from pathlib import Path

# -- Project information -----------------------------------------------------

project = 'neuraloperator'
from datetime import datetime
year = datetime.now().year
copyright = f'{year}, Jean Kossaifi, David Pitt, Nikola Kovachki, Zongyi Li and Anima Anandkumar'
author = 'Jean Kossaifi, David Pitt, Nikola Kovachki, Zongyi Li and Anima Anandkumar'

# The full version, including alpha/beta/rc tags
import neuralop
release = neuralop.__version__



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax', #'sphinx.ext.imgmath',
    'numpydoc.numpydoc',
    'sphinx_gallery.gen_gallery',
]

# -- sphinx_gallery configuration
from sphinx_gallery.gen_gallery import ExplicitOrder

sg_gallery_dir = 'auto_examples'
sg_examples_dir = '../../examples'

def subsection_title_order(src_dir: str):
    """
    src_dir : str
        string name of source path
    """

    sub_folders = [p for p in Path(src_dir).iterdir() if p.is_dir()]
    final_order = []
    for sub_folder in sub_folders:
        if sub_folder.name == "release_highlights":
            final_order.insert(0, sub_folder)
        else:
            final_order.append(sub_folder)
    return ExplicitOrder(final_order)
    
sphinx_gallery_conf = {
     'examples_dirs': [sg_examples_dir],   # path to your example scripts
     'gallery_dirs': [sg_gallery_dir],  # path to where to save gallery generated output
      #'subsection_order' : subsection_title_order(sg_examples_dir), # TODO@DAVID: turns out unneeded
     'nested_sections': True, # ensure we can nest multiple levels in the gallery toctree
}

# -- Template configuration 

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- NumpyDoc configuration ---------------------------------------
numpydoc_class_members_toctree = False
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False

# generate autosummary even if no references
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_flags = ['members']

# Napoleon
napoleon_google_docstring = False
napoleon_use_rtype = False

# imgmath/mathjax
imgmath_image_format = 'svg'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'tensorly_sphinx_theme'
html_logo = '_static/logos/neuraloperator_logo.png'
html_show_sphinx = False

html_theme_options = {
    'github_url': 'https://github.com/neuraloperator/neuraloperator',
    # 'google_analytics' : 'G-QSPLEF75VT',
    'nav_links' : [('Install', 'install'),
                   ('User Guide', 'user_guide/index'),
                   ('API', 'modules/api'),
                   ('Examples', 'auto_examples/index')
                  ],
    # 'external_nav_links' : [('TensorLy', 'http://tensorly.org/dev')]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Remove the permalinks ("¶" symbols)
html_permalinks_icon = ""

