# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os 
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Quantum Bayesian Networks'
copyright = '2023, Oleksandr Hoviadin, Mohamed Rayan Amara, Siwar Sraieb'
author = 'Oleksandr Hoviadin, Mohamed Rayan Amara, Siwar Sraieb'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'rst2pdf.pdfbuilder', 'sphinx.ext.inheritance_diagram', 'sphinx.ext.graphviz']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

pdf_documents = [('index', u'qbn', u'Quantum Bayesian Networks', u'Oleksandr Hoviadin, Mohamed Rayan Amara, Siwar Sraieb'),]

