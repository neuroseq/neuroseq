# In-App LLM Assistant (Experimental)

!!! warning "Experimental feature"
The in-app chat assistant is experimental and under active development.
Behaviour, UI, supported providers, and the project context it receives
may change without notice.

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

The panel itself is available in the default install. API providers require
their optional dependencies. Codex uses a separately installed Codex CLI
and does not require the `chat` extra.

## Configuring a provider

Open **Settings → LLM assistant** to choose a provider and model. Four
backends are bundled:

| Provider  | Notes                                                                       |
| --------- | --------------------------------------------------------------------------- |
| Ollama    | Local models served by a running [Ollama](https://ollama.com) daemon.       |
| OpenAI    | OpenAI API, or any OpenAI-compatible endpoint via the base URL field.       |
| Anthropic | Anthropic Claude API.                                                       |
| Codex     | Local Codex app-server using your existing Codex sign-in and configuration. |

For OpenAI and Anthropic you can paste an API key into the settings dialog.
Keys are stored in the OS keyring via the `keyring` package; if no keyring
backend is available NeuroSeq falls back to the `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` environment variables.

For Ollama you only need the base URL of the daemon (default
`http://localhost:11434`) and the name of a pulled model.

### Codex

1. Install the [Codex CLI](https://learn.chatgpt.com/docs/codex/cli), then run
   `codex login` from a terminal under the same operating-system account as NeuroSeq.
2. In **Settings → LLM Assistant**, choose **Codex (local sign-in)**.
3. Leave **Codex executable** empty to find `codex` on PATH, or enter the full
   executable path. This can also point to the Codex binary bundled with your
   VS Code extension. Restart NeuroSeq after changing your system PATH.
4. Leave **Model** set to `default` to use your Codex configuration, or use
   **Refresh** to choose a model. Model access depends on your account and provider.
5. Use **Test connection** to check the local app-server and sign-in without
   sending an inference request. Open **Chat** from the toolbar and ask a question.

For example: “Where is the stimulation editor implemented?” or “Explain the
equations and parameters in my current model.” The panel supplies NeuroSeq's
documentation, a fresh project summary, and the conversation so far. It can
request project details and source pages through the same inspection protocol
as the other providers.

The integration uses the [Codex app-server protocol](https://learn.chatgpt.com/docs/app-server),
which also supports custom IDE clients. It is a NeuroSeq help panel with streamed
answers and Stop support. It does not embed the VS Code extension or share its
open conversation. Each request uses an ephemeral thread with a read-only Codex
sandbox and no native tool approvals. Your local Codex provider configuration
and configured tools still apply; the native filesystem sandbox does not confine
reads to the NeuroSeq source package. Native tool approval requests and interactive
tool prompts are rejected. This panel does not provide a code-editing review workflow.

## Using the assistant

Toggle the chat panel from the **Chat** action in the toolbar (robot icon).
Type a message and press **Enter** or click **Send**; use **Shift+Enter** for a
new line. Replies stream in token by token. Each
new message includes a fresh snapshot of the current project so the model
can answer questions about the loaded simulation and data.

The assistant can request read-only details from the loaded project: model
equations, parameters, initial conditions, synapses, monitors, runtime state,
stimuli, experiments, and imaging metadata. Large arrays are read in pages of
flattened values, with shape and units supplied where available. It can also
list and read Python files within the installed `neuroseq` source package,
including example projects. This inspection protocol does not expose files
elsewhere on your computer; see the Codex-specific tool behavior above.

Requested project data and source text are sent to your selected provider.
With a hosted provider, this includes the requested experimental data. Reads
reflect the project at the time of each request. Stop cancels further requests;
files already saved remain on disk.

To create an analysis plugin, ask the assistant, for example: "Create an
analysis plugin that plots the mean trace across selected neurons." Before
anything is saved, NeuroSeq shows a confirmation dialog with the target path and
the complete code (or, for an update, a diff against the current file). The file
is written only if you click **Save plugin**; if you cancel, the assistant is told
that you declined. Review the code before accepting: tool results can contain
text from shared project files, so a malicious project could try to steer the
assistant into proposing harmful code.

Chat-authored plugins are saved only in your per-user plugin folder (for example
`~/.local/share/neuroseq/analysis_plugins` on Linux; override with the
`NEUROSEQ_USER_PLUGIN_DIR` environment variable), never in the NeuroSeq package.
Names starting with `_` and names of built-in plugins are rejected, and only
plugins already in your folder can be overwritten. Python syntax is checked
before saving; this does not verify scientific correctness or runtime behavior.
Reopen the right-click analysis menu to discover the new plugin. Discovery
imports the Python code.

## Limitations

- Project and source access is read-only except for Python files in the analysis
  plugin folder. NeuroSeq's inspection protocol cannot directly execute Python
  or run simulations. Codex's configured native tools have their own sandbox policy.
- Inspection uses a JSON request protocol supported by all four backends;
  reliability depends on the model following that protocol. Each turn allows
  up to 12 read/write requests; ask a follow-up to continue. Data is read on demand rather
  than sending the entire project and source tree in every prompt.
- Conversations are not persisted across sessions.
- Provider availability, model lists, and rate limits depend entirely on
  the chosen backend.
