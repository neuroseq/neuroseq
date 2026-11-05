# Documentation

This project uses [Zensical](https://zensical.org/docs/), configured in
`zensical.toml` at the repository root. The published documentation is at
<https://neuroseq.github.io/>.

## To run docs locally

For the uv workflow, install the development dependencies, which include Zensical:

```bash
uv sync --locked
```

Start the local documentation server from the repository root:

```bash
uv run zensical serve
```

Then open the local URL shown in the terminal.

If NeuroSeq is installed directly into Conda, activate that environment,
install the documentation packages with `python -m pip install --group docs`,
and run `zensical serve` instead. See
[environment selection](developer_installation.md#using-uv-from-a-conda-shell).

## Optional: build static docs

To generate static files for deployment:

```bash
uv run zensical build --clean --strict
```

The generated site is written to the `site/` folder.

## Deployment

The `docs` GitHub Actions workflow runs the same clean, strict build on all pull
requests and pushes to the `main` branch. Only pushes to `main` publish the generated site
to the `main` branch of [neuroseq/neuroseq.github.io](https://github.com/neuroseq/neuroseq.github.io).
It can also be run manually on `main` using **Run workflow**.

### Why not publish directly?

The reason to push to and publish from [neuroseq/neuroseq.github.io](https://github.com/neuroseq/neuroseq.github.io)
is so that the published site is located at <https://neuroseq.github.io/>, i.e. `org-name.github.io`.

If it was published directly from this repo, the url would be https://neuroseq.github.io/neuroseq,
i.e. `org-name.github.io/repo-name` - but if in the future you get a custom domain name for neuroseq then the extra
repo, deploy key etc would no longer be needed and the neuroseq.github.io repo could be removed.
