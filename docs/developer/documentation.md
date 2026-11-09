# Documentation

The docs are Markdown in `docs/`, built with [Zensical](https://zensical.org/docs/)
and published at <https://neuroseq.github.io/>. Navigation is in `zensical.toml`.

## Layout

| Folder                  | Content                                                |
| ----------------------- | ------------------------------------------------------ |
| `docs/getting_started/` | Installation, quickstart, GUI overview, chat assistant |
| `docs/networks/`        | Simulators, external backends, hardware substrates     |
| `docs/input_design/`    | Static and dynamic stimulus design, export             |
| `docs/readout/`         | Experiments, imported data, analysis                   |
| `docs/developer/`       | This guide                                             |
| `docs/images/`          | Images                                                 |

A new page must be added to the `nav` in `zensical.toml`.

## Docs feed the chat assistant

The wheel bundles `docs/` as `neuroseq/docs` (`[tool.hatch.build.targets.wheel.force-include]`
in `pyproject.toml`). `neuroseq.llm.system_prompt` concatenates every `.md` file
into the assistant's system prompt, from the installed package or, in a checkout,
from the repository's `docs/`. Keep pages short and accurate: every word costs
prompt tokens and stale text produces wrong answers.

## Build and preview

```bash
uv run zensical serve                  # live preview at the URL shown
uv run zensical build --clean --strict # what CI runs; output in site/
```

With Conda, install the docs group (`python -m pip install --group docs`) and
drop `uv run`. `prek run -a` formats Markdown with prettier.

## Deployment

`.github/workflows/docs.yml` runs the strict build on pull requests and pushes
to `main`, and can be started manually. Pushes to `main` publish `site/` to
[neuroseq/neuroseq.github.io](https://github.com/neuroseq/neuroseq.github.io),
so the site lives at `neuroseq.github.io` rather than `neuroseq.github.io/neuroseq`.
With a custom domain, that extra repository and its deploy key would not be needed.
