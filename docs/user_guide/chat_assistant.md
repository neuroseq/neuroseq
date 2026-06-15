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

## Limitations

- No tool use: the assistant can read project context but cannot edit
  the project or run actions on your behalf.
- Conversations are not persisted across sessions.
- Provider availability, model lists, and rate limits depend entirely on
  the chosen backend.
