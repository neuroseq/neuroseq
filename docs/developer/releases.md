# Making a release

NeuroSeq is published to [PyPI](https://pypi.org/project/neuroseq/) by the
`release` GitHub Actions workflow (`.github/workflows/release.yml`) when a version
tag is pushed.

## Steps

1. In a pull request, bump `__version__` in `src/neuroseq/__init__.py`
   (e.g. `0.29.1` → `0.30.0`) and merge it into `main`.
2. Tag the merge commit on `main` with the same version, without a `v` prefix,
   and push the tag:

   ```bash
   git switch main
   git pull
   git tag 0.30.0
   git push origin 0.30.0
   ```

3. The `release` workflow checks that the tagged commit is on `main` and that the
   tag matches the package version, then builds the sdist and wheel and uploads
   both to PyPI.
4. Once the PyPI upload succeeds, it creates a GitHub release for the tag, with
   release notes generated from the merged pull requests and the sdist and wheel
   attached.

Only tags of the form `*.*.*` pushed to `neuroseq/neuroseq` are published; a tag on
a commit that is not on `main` fails the check, so push `main` before the tag. The
tag push itself does not run the test suite, so only tag a commit whose CI on
`main` has passed. On pull requests and pushes to `main` the workflow only builds
the package, so packaging problems show up before a release.

## Publishing setup

The upload uses PyPI
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is
stored in GitHub. This needs:

- a trusted publisher on PyPI for owner `neuroseq`, repository `neuroseq`,
  workflow `release.yml` and environment `pypi`;
- a `pypi` environment in the GitHub repository settings. Adding required
  reviewers to it makes each upload wait for approval.
