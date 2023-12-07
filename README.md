# Riallto documentation

This branch contains source and the build for the Riallto documentation.

Docs are built offline on a Ryzen AI laptop. Generated html docs can be found in the ./docs folder on this branch. 

To rebuild, pull the main branch, copy the source dir, and make.bat to the docs folder in the main main repo.

In the Riallto venv run:

```
.\make.bat html
```

Copy contents of build/html to he docs folder

## Prerequisites for the venv

Install Sphinx on Windows, then in the venv:

```
pip install recommonmark
pip install sphinx_markdown_tables
pip install nbsphinx
pip install nbsphinx-link
pip install pandoc
pip install sphinx_copybutton
```

## Steps to reproduce webpages from scratch:

* In the docs directory; run: sphinx-quickstart to setup the Sphinx config (version number can be set here)
* Copy the conf.py settings from this directory (edit as required)
* Run sphinx-apidoc -o source/ ./npu to auto generate the templates for the source code
* Edit the index.rst to change the homepage and to include/exclude notebook/rst files as required








