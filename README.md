# Formozov Project

[![CI](https://github.com/ssciwr/formozov-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ssciwr/formozov-project/actions/workflows/ci.yml)

Private repository containing the prototype GUI for the [Formozov project](https://github.com/ssciwr/formozov-project-notes).

## Installation

If you have ssh authentication to GitHub set up you can install directly from this repository with pip:

```bash
pip install git+ssh://git@github.com/ssciwr/formozov-project
```

Alternatively you can clone the repository and then install it:

```bash
git clone git@github.com:ssciwr/formozov-project.git
cd formozov-project
pip install .
```

## Use

After installing, you should be able to run the following command to launch the GUI:

```bash
formozov-project
```

## Developer Installation

To do an editable installation including test dependencies and pre-commit:

```bash
git clone git@github.com:ssciwr/formozov-project.git
cd formozov-project
pip install -e .[tests]
pip install pre-commit
pre-commit install
```

The `-e` (editable install) flag used during installation means that any changes to the source code will be reflected in the installed package without needing to reinstall.

## Developer Use

The GUI is launched the same way as above:

```bash
formozov-project
```

To run the test suite:

```bash
pytest
```
