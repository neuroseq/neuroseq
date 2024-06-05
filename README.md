# Formozov Project

[![CI](https://github.com/ssciwr/formozov-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ssciwr/formozov-project/actions/workflows/ci.yml)

Private repository for the Formozov project.

## Installation

If you have ssh authentication to github set up you can install directly with pip:

```bash
`pip install git+ssh://git@github.com/ssciwr/formozov-project`
```

Alternatively you can clone the repository and then install it:

```bash
git clone git@github.com:ssciwr/formozov-project.git
cd formozov-project
pip install .
```

## Use

After installing you should be able to run the following command to launch the GUI:

```bash
formozov-project
```

## Developer Installation

To do an editable install including test dependencies and pre-commit:

```bash
git clone git@github.com:ssciwr/formozov-project.git
cd formozov-project
pip install -e .[tests]
pip install pre-commit
pre-commit install
```
