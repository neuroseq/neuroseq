# In-App LLM Assistant (Experimental)

!!! warning "Experimental feature"
The in-app chat assistant is experimental and under active development.
Behaviour, UI, supported providers, and the project context it receives
may change without notice. It is not enabled in the default install.

NeuroSeq ships with an optional chat panel that lets you talk to an LLM about
the currently loaded project. The assistant is given a snapshot of the
project state (model, stimuli, experiments) as context and replies in a side
panel next to the main window.

## Installation

The assistant lives behind the `chat` extra, which pulls in the provider
SDKs and `keyring` for credential storage.

From the repository directly (requires SSH access to GitHub):

```bash
pip install "neuroseq[chat] @ git+ssh://git@github.com/ssciwr/neuroseq.git"
```

From a local checkout:

```bash
pip install ".[chat]"
```

Without the `chat` extra the panel is hidden and no LLM code paths run.

## Configuring a provider

Open **Settings → LLM assistant** to choose a provider and model. Three
backends are bundled:

| Provider  | Notes                                                                 |
| --------- | --------------------------------------------------------------------- |
| Ollama    | Local models served by a running [Ollama](https://ollama.com) daemon. |
| OpenAI    | OpenAI API, or any OpenAI-compatible endpoint via the base URL field. |
| Anthropic | Anthropic Claude API.                                                 |

For OpenAI and Anthropic you can paste an API key into the settings dialog.
Keys are stored in the OS keyring via the `keyring` package; if no keyring
backend is available NeuroSeq falls back to the `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` environment variables.

For Ollama you only need the base URL of the daemon (default
`http://localhost:11434`) and the name of a pulled model.

## Using the assistant

Toggle the chat panel from the **Chat** action in the toolbar (robot icon).
Type a message and press **Send**; replies stream in token by token. Each
new message includes a fresh snapshot of the current project so the model
can answer questions about the loaded simulation and data.

The assistant can request read-only details from the loaded project: model
equations, parameters, initial conditions, synapses, monitors, runtime state,
stimuli, experiments, and imaging metadata. Large arrays are read in pages of
flattened values, with shape and units supplied where available. It can also
list and read Python files within the installed `neuroseq` source package,
including example projects. Files elsewhere on your computer are not accessible.

Requested project data and source text are sent to your selected provider.
With a hosted provider, this includes the requested experimental data. Reads
reflect the project at the time of each request. Stop cancels further requests;
files already saved remain on disk.

To create an analysis plugin, ask the assistant, for example: "Create an
analysis plugin that plots the mean trace across selected neurons." It can save
Python files directly in `src/neuroseq/resources/analysis_plugins` (or the
equivalent installed package folder). Updates to existing files require an
explicit overwrite request. Python syntax is checked before saving; this does
not verify scientific correctness or runtime behavior. Reopen the right-click
analysis menu to discover the new plugin. Discovery imports the Python code.
The package folder must be writable by your operating-system account.

## Limitations

- Project and source access is read-only except for Python files in the analysis
  plugin folder. The assistant cannot directly execute Python or run simulations.
- Inspection uses a JSON request protocol supported by all three backends;
  reliability depends on the model following that protocol. Each turn allows
  up to 12 read/write requests; ask a follow-up to continue. Data is read on demand rather
  than sending the entire project and source tree in every prompt.
- Conversations are not persisted across sessions.
- Provider availability, model lists, and rate limits depend entirely on
  the chosen backend.
