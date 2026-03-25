# Council Engine

A structured deliberation protocol across diverse AI models.

Single-model outputs are fluent, fast, and often wrong in the same direction. Council runs a bounded deliberation across different models, processes critique explicitly, and returns a decision you can inspect. The point is not more text. It is better judgment.

**Website:** [councilengine.dev](https://councilengine.dev)

## How it works

Council orchestrates a bounded 3-stage protocol:

1. **Proposals** — Multiple models respond to the same task independently. Each produces a clear position.
2. **Critique** — Models review each other through a constrained vocabulary: *challenge*, *alternative*, *refinement*, and *question*. Silence is treated as approval.
3. **Resolution** — A lead model classifies the disagreement and produces one of four adaptive outcomes:
   - **Recommendation** — Council agrees; single clear guidance.
   - **Alternatives** — Real tradeoffs remain; presented with decision rules.
   - **Question** — A missing fact is decisive; outputs a clarifying question.
   - **Investigate** — Evidence is needed; outputs an investigation plan.

## Why heterogeneous models

Research shows that multi-agent debate improves reasoning when the agents are actually different. Same-model personas tend to converge on the same errors. Council uses genuinely different foundation models — not different prompts to the same model — because diverse architectures surface conflicting assumptions and different failure modes.

## Supported providers

| Provider | Auth | Web search |
|----------|------|------------|
| **Gemini** (Google) | API key (AI Studio or Vertex AI) or ADC | Yes (Google Search) |
| **ChatGPT** (OpenAI) | API key | Yes (web_search_preview) |
| **Claude** (Anthropic) | API key | No |
| **Ollama** (local) | None — runs locally | No |

## Install

```bash
pip install council-engine
```

All providers are included. No optional extras needed.

## Setup

Run the guided setup:

```bash
council init
```

This creates `~/.council/config.toml` with your API keys and validates each connection. You need at least one provider configured; for multi-model deliberation, configure two or more.

### Manual configuration

Edit `~/.council/config.toml` directly:

```toml
[providers.gemini]
api_key = "your-google-api-key"       # from https://aistudio.google.com/apikey

[providers.chatgpt]
api_key = "your-openai-api-key"       # from https://platform.openai.com/api-keys

[providers.claude]
api_key = "your-anthropic-api-key"    # from https://console.anthropic.com/settings/keys

[providers.ollama]
host = "http://localhost:11434"       # ollama serve && ollama pull <model>
model = "llama3.1:8b"                 # browse models: https://ollama.com/library
```

Environment variables override the config file: `COUNCIL_GEMINI_API_KEY`, `COUNCIL_OPENAI_API_KEY`, `COUNCIL_CLAUDE_API_KEY`, `COUNCIL_OLLAMA_HOST`.

## Usage

Council has two modes: an **interactive REPL** for exploratory sessions, and **CLI commands** for one-shot queries and scripting.

### Interactive REPL

```bash
council
```

Launches a persistent session with tab completion, slash commands, and multi-model deliberation. This is the recommended way to use Council for exploratory work — you can run discussions, follow up with the lead model, reconvene, attach files, and manage conversations all in one session.

- `message` — convene the full council
- `@gemini <msg>` — send to a single model directly
- `/discuss <prompt>` — reconvene the council after follow-up
- `/attach <file>` — attach files for the next message
- `/save` — save the last response to a file
- `/new` / `/resume <name>` — manage conversations
- `/help` — full command list

### CLI commands

CLI commands are fire-and-forget — they run once and exit. Use them for scripting, pipelines, and quick queries.

```bash
# Multi-model deliberation (proposals → critique → resolution)
council discuss --models gemini,claude "Should we use event sourcing or CRUD?"

# Single-model query
council ask --model chatgpt "What are the tradeoffs of gRPC vs REST?"

# Multi-model chat in a named conversation
council chat --models gemini,chatgpt my-project "Your message here"

# Promote a one-shot ask into a named conversation
council promote my-project
```

To follow up on a `council discuss` result interactively, launch the REPL with `council` and use `/resume`.

## Architecture

Council is built with explicit boundaries and auditability:

- **Event bus** — orchestration decoupled from rendering; same protocol drives terminal UI, plain text, or custom consumers
- **Persistent audit trail** — every proposal, critique, and resolution stored in SQLite with full metadata
- **Chair subsystem** — sandboxed file operations with mediated reads (workspace-wide) and writes (council-output/ only)
- **Provider abstraction** — pluggable model backends with parallel execution via asyncio
- **Two integration surfaces** — interactive REPL for humans, CLI commands for automation and agent tooling

See the full [architecture documentation](https://councilengine.dev/architecture).

## Research

Council's design is informed by research on multi-model deliberation:

1. Du, Y. et al. 2023. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.*
2. *Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity.* 2026.
3. *If Multi-Agent Debate is the Answer, What is the Question?* 2025.
4. *Dipper: Diversity in Prompts for Producing Large Language Model Ensembles.* 2024.
5. Zheng, et al. 2023. *Persona prompting and factual task performance.*

## License

MIT
