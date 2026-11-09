# LLM Assistant (Experimental)

An optional chat panel for questions about the loaded project and NeuroSeq.
It receives the documentation, a fresh project summary with each message, and
read-only access to project details and the NeuroSeq source. Behaviour and
supported providers may change.

## Setup

Open **Settings → LLM Assistant**, choose a provider and model, then click
**Test connection** (checks sign-in without sending a prompt).

| Provider                    | Requirements                                                                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Codex (local sign-in)       | [Codex CLI](https://learn.chatgpt.com/docs/codex/cli) and `codex login`. Recommended.                                                             |
| Claude Code (local sign-in) | [Claude Code](https://code.claude.com/docs/en/setup) and `claude auth login` (subscription, Console account or `ANTHROPIC_API_KEY`). Recommended. |
| Ollama (local) (default)    | running [Ollama](https://ollama.com) daemon (default `http://localhost:11434`) and a pulled model (default `qwen2.5:7b`)                          |
| OpenAI                      | API key; any OpenAI-compatible endpoint via the base URL field                                                                                    |
| Anthropic                   | API key                                                                                                                                           |

- Ollama, OpenAI and Anthropic need the `chat` extra:
  `pip install "neuroseq[chat]"` (installs `openai`, `anthropic`, `httpx`,
  `keyring`).
- API keys are stored in the OS keyring. Without a keyring backend, NeuroSeq
  reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.
- Codex and Claude Code: sign in under the same OS account as NeuroSeq. Leave
  the executable field empty to use `PATH` (restart NeuroSeq after changing
  `PATH`); Codex may also point to the binary bundled with its VS Code
  extension. Model `default` uses your CLI configuration; Claude Code also
  offers `opus`, `sonnet`, `haiku`, Codex lists models via **Refresh**.

## Use

Toggle **View → Show Chat** (robot icon). **Enter** sends, **Shift+Enter** adds
a line, **Stop** cancels. Replies stream in.

The assistant can read model equations, parameters, initial conditions,
synapses, monitors, runtime state, stimuli, experiments, imaging metadata
(large arrays in pages) and Python files inside the installed `neuroseq`
package. It cannot read other files, run code or run simulations. Anything it
reads is sent to the provider, including experimental data with a hosted
provider.

### Writing analysis plugins

Ask e.g. "Create an analysis plugin that plots the mean trace across selected
neurons." NeuroSeq shows the target path and full code (or a diff) and saves
only after you click **Save plugin**. Review the code: a shared project could
contain text that steers the assistant towards harmful code.

- Saved only to the per-user plugin folder (Linux:
  `~/.local/share/neuroseq/analysis_plugins`, override with
  `NEUROSEQ_USER_PLUGIN_DIR`), never into the package.
- Names starting with `_` or matching built-in plugins are rejected; only
  plugins already in your folder can be overwritten.
- Syntax is checked, correctness is not. Reopen the right-click analysis menu
  to load the plugin (this imports its code).

## Limitations

- Up to 12 read/write requests per turn; ask a follow-up to continue.
  Reliability depends on the model following NeuroSeq's JSON request protocol.
- Conversations are not saved between sessions.
- Codex runs each request in an ephemeral read-only sandbox, but your configured
  Codex tools still apply and its filesystem sandbox is not limited to the
  NeuroSeq package. Claude Code runs `claude --print` in an empty temporary
  directory with all native tools, MCP servers and settings disabled.
