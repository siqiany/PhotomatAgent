# PhotomatAgent

PhotomatAgent is a Python scientific-agent runtime for materials research, with a focus on infrared optoelectronic materials and detectors. It provides an explicit model/tool loop, structured scientific state, local workspace tools, optional scientific capability packs, and integrations for MCP and SCNet-based computation.

It is designed to run as a CLI application. The runtime owns context construction, model streaming, tool authorization, tool execution, observation redaction, and event logging.

## Features

- Explicit asynchronous agent loop with streaming model responses and tool calls.
- Provider-neutral model protocol with OpenAI Responses API, Anthropic Messages API, and offline fake-provider adapters.
- Tool registry with direct, deferred, and hidden exposure modes.
- Progressive discovery for deferred tools through `tool_search`, `tool_describe`, and `tool_call`.
- Workspace-scoped file and shell tools with approval policies and sensitive-path blocking.
- Durable conversation history plus a bounded working context with tool-result pruning and optional structured compaction.
- Structured `ScientificState` for claims, evidence, calculations, tasks, questions, and contradictions.
- Capability packs for materials databases, literature retrieval, crystal structures, electronic structure, infrared constraints, quantum dots, detector metrics, and more.
- Optional VASP, Hefei-NAMD, MAGUS, Slurm/SCNet, and MCP integrations.
- JSONL execution traces, replay, session analysis, and deterministic experiment runs.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended)

The core package uses Pydantic, Typer, Rich, prompt-toolkit, python-dotenv, OpenAI, Anthropic, pymatgen, and MCP. Scientific integrations are installed as optional extras.

## Installation

```bash
git clone <repository-url>
cd PhomatAgent
uv sync
```

Install only the extras you need:

```bash
# Materials Project
uv sync --extra materials

# Electronic-structure analysis
uv sync --extra electronic

# Local literature indexing and retrieval
uv sync --extra literature

# All optional scientific capability groups
uv sync --extra science
```

## Configuration

Create a workspace-local `.env` file from the supplied template:

```bash
cp .env.example .env
```

For OpenAI, configure at least:

```dotenv
PHOTOMATAGENT_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

For Anthropic:

```dotenv
PHOTOMATAGENT_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
```

Never commit `.env`. It can contain model credentials, Materials Project credentials, and SCNet/HPC settings.

Run the local diagnostic after configuration:

```bash
uv run photomatagent doctor
```

`doctor` checks the Python/runtime configuration, configured provider, available skills directory, writable session directory, and a local fake-provider loop smoke test.

## Quick start

Start an interactive chat:

```bash
uv run photomatagent chat
```

Run a single goal:

```bash
uv run photomatagent chat \
  --goal "Find useful capabilities for infrared detector material screening"
```

Run without a paid model, using the built-in fake provider:

```bash
uv run photomatagent chat \
  --provider fake \
  --approval auto \
  --goal "Inspect this workspace"
```

By default, writes, edits, shell commands, and newly registered scientific tools require approval. `--approval auto` bypasses confirmation and should only be used in a trusted workspace.

## Common commands

```bash
# Inspect registered tools and their exposure mode
uv run photomatagent tools list
uv run photomatagent tools surface
uv run photomatagent tools search "band structure"

# Inspect skills and capability availability
uv run photomatagent skills list
uv run photomatagent scientific status

# Inspect configured MCP servers
uv run photomatagent mcp list
uv run photomatagent mcp status

# Inspect recorded sessions
uv run photomatagent sessions list
uv run photomatagent sessions stats latest
uv run photomatagent sessions replay latest

# Run deterministic experiment definitions
uv run photomatagent experiments run experiments/offline-smoke.json
```

During interactive chat, `/help` lists slash commands. The router includes tool, skill, scientific-capability, MCP, session, experiment, configuration, and approval commands.

## How it works

```text
CLI / experiment runner
  → AgentRuntime
  → ContextEngine + ToolSurfacePlanner
  → ModelProvider
  → tool calls
  → safety and permission checks
  → ToolRegistry + Tool.execute()
  → observation redaction and truncation
  → conversation, scientific state, and JSONL events
  → next iteration or completion
```

The main control loop is implemented in `src/photomatagent/runtime/loop.py`. It does not delegate tool execution to a model SDK.

### Tool exposure

Every tool is registered with one of three exposure modes:

- `DIRECT`: its full schema is supplied to the model each turn.
- `DEFERRED`: it is discoverable through a compact catalog; the model must use `tool_search` → `tool_describe` → `tool_call`.
- `HIDDEN`: it is not model-visible and cannot be executed.

Deferred execution still follows the same sensitive-path checks, permission policy, argument validation, event logging, and observation policy as direct tools.

### Context and state

`ConversationState` retains provider-neutral conversation messages. `ContextEngine` builds a separate model-visible working copy, replaces old large tool outputs with provenance placeholders when needed, and can request a schema-validated compaction summary only when the conversation is safe to compact.

`ScientificState` is maintained independently and can receive claims, evidence, calculation records, and tasks through tool result `state_updates`.

## Scientific capabilities

Capability packs are registered from `src/photomatagent/scientific/capabilities/registry.py`. Optional dependencies are checked at runtime; unavailable packs should report their status rather than prevent the base runtime from starting.

| Capability | Main use |
| --- | --- |
| `materials` | Materials Project search, summaries, and structures. |
| `literature` | arXiv/local-paper search, indexing, passage retrieval, and evidence extraction. |
| `structure` | Structure summary, symmetry, density, neighbours, and format conversion. |
| `electronic` | Band/DOS summaries, plotting, and effective-mass workflows. |
| `ir` | Infrared target and physical-constraint compilation. |
| `quantum_dot` / `alloy` | Brus-model analysis, size/composition scans, and band-gap bowing. |
| `photodetector` | EQE/responsivity conversion and target checks. |
| `defects`, `transport`, `device`, `optics`, `interface`, `kp` | Optional domain-specific calculations and diagnostics. |
| `generation` | Formula retrieval/validation and optional generative workflows. |
| `vasp`, `namd`, `magus` | Prepare, submit, monitor, collect, and inspect high-cost workflows. |

Check the capabilities actually available in the current environment:

```bash
uv run photomatagent scientific status
```

## MCP and remote compute

MCP server declarations are loaded from `.photomatagent/mcp.json` or `.photomatagent/mcp.yaml`. Supported transports are stdio and streamable HTTP. MCP tools are registered through the same tool registry and can use deferred exposure.

The VASP, Hefei-NAMD, and MAGUS integrations use a narrow application layer over the SCNet backend. Remote submission is gated by configuration:

```dotenv
PHOTOMATAGENT_ALLOW_HPC_SUBMIT=0
```

Keep this disabled until the target SCNet environment, scheduler configuration, resource policy, and external executables have been verified.

## Sessions and experiments

When event logging is enabled, runtime events are written under:

```text
.photomatagent/sessions/<session-id>/events.jsonl
```

The session tools can list, summarize, inspect working-context metrics, and replay those traces without rerunning a model or tool.

Experiment JSON files define independent tasks and deterministic expectations. The runner creates a fresh runtime and trace for each task, then stores experiment results under `.photomatagent/experiments/`.

## Development

Run the test suite:

```bash
uv run pytest
```

Run selected tests while working on a subsystem:

```bash
uv run pytest tests/test_loop.py
uv run pytest tests/test_tool_surface.py
uv run pytest tests/test_context_lifecycle.py
```

## Project layout

```text
src/photomatagent/
├── cli/            # CLI commands, chat UI, rendering, approvals
├── runtime/        # agent loop, context, safety, permissions, events
├── models/         # OpenAI, Anthropic, and fake-provider adapters
├── tools/          # tool protocol, registry, workspace tools, discovery
├── scientific/     # scientific state, capability packs, applications, HPC
├── mcp/            # external MCP gateway
├── mcp_servers/    # bundled SCNet MCP server
├── skills/         # skill discovery and loading
├── logging/        # JSONL event persistence
├── observability/  # trace analysis and replay
└── experiments/    # experiment runner and evaluation

skills/              # domain SOP content
tests/               # pytest suite
experiments/         # runnable experiment specifications and vertical slices
docs/                # supplementary design and domain documentation
```

## Safety notes

- `bash` runs commands in the local environment after approval; it is not an OS sandbox.
- Sensitive file paths are blocked by policy, but this is a guardrail rather than a complete filesystem isolation boundary.
- Scientific capabilities may be unavailable, unconfigured, approximate, or dependent on external data and solvers. Inspect their status and returned evidence before relying on results.
- Scheduler completion does not, by itself, establish scientific validity; collect and validate result artifacts.

## License

No repository-level license file is currently included. Confirm the intended license before redistributing the project or its bundled assets.
