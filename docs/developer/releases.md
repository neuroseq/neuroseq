# Making a release

`.github/workflows/release.yml` publishes to [PyPI](https://pypi.org/project/neuroseq/)
when a version tag is pushed.

1. In a pull request, bump `__version__` in `src/neuroseq/__init__.py` and merge it into `main`.
2. Once CI on `main` has passed, tag the merge commit with the same version (no `v` prefix):

   ```bash
   git switch main
   git pull
   git tag 0.30.0
   git push origin 0.30.0
   ```

3. The workflow checks that the commit is on `main` and the tag matches
   `__version__`, builds the sdist and wheel, tests the installed wheel (including
   the bundled docs), uploads to PyPI and creates a GitHub release with generated
   notes.

Only `*.*.*` tags on `neuroseq/neuroseq` publish. The tag push does not run the
test suite. Pull requests and pushes to `main` only build and test the package.

## Publishing setup

Uploads use PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/)
(no stored token). This needs a trusted publisher on PyPI for `neuroseq/neuroseq`,
workflow `release.yml`, environment `pypi`, and a `pypi` environment in the
GitHub repository settings (add required reviewers to gate uploads).
