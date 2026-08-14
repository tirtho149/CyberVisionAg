# Claude CLI Headless Mode (`claude -p`) — Quick Reference

## Core Concept

`claude -p` runs Claude Code in **headless/pipe mode** — no interactive UI, output goes to stdout. Designed for scripting, automation, and programmatic use.

---

## Invocation

```bash
claude -p "Your prompt here"
```

### Key Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-p` | Enable headless mode (required) | `claude -p "prompt"` |
| `--output-format` | Output structure | `stream-json`, `json`, `text` |
| `--model` | Select model | `haiku`, `sonnet`, `opus` |
| `--allowedTools` | Auto-approve tools (no user confirmation) | `"Read"`, `"Read,Edit,Bash"` |
| `--verbose` | Include detailed event stream | Flag only |
| `--system-prompt` | Replace entire system prompt | `"You are a security reviewer"` |
| `--append-system-prompt` | Add to default system prompt | `"Focus on performance"` |
| `--include-partial-messages` | Stream partial token updates | Flag only |
| `--continue` | Continue a previous conversation | Flag only |
| `--resume` | Resume a specific session | Session ID |

---

## Output Formats

### `text` (default)
Plain text — just the final response.

### `json`
Single JSON object with metadata.

### `stream-json` (recommended for traces)
Newline-delimited JSON (NDJSON). Each line is an event:

```json
{"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}, {"type": "tool_use", "name": "Read", "input": {...}}]}}
{"type": "result", "session_id": "...", "duration_ms": 12893, "num_turns": 9, "total_cost_usd": 0.0298, "result": "..."}
```

**Event types in `stream-json`:**
- `"type": "assistant"` — Contains `message.content[]` array with `text` and `tool_use` blocks
- `"type": "result"` — Final event. Contains `session_id`, `duration_ms`, `num_turns`, `total_cost_usd`, and `result` (final text response)

---

## Extracting Reasoning Traces (Python)

```python
import json, subprocess

result = subprocess.run(
    ["claude", "-p", prompt,
     "--allowedTools", "Read",
     "--output-format", "stream-json",
     "--verbose",
     "--model", "haiku"],
    capture_output=True, text=True,
    cwd=str(PROJECT_DIR),
    timeout=180
)

trace = []
final_result = {}

for line in result.stdout.strip().split('\n'):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue

    if obj.get("type") == "assistant":
        for c in obj.get("message", {}).get("content", []):
            if c.get("type") == "text" and c.get("text", "").strip():
                trace.append({"role": "assistant", "type": "text", "content": c["text"].strip()})
            elif c.get("type") == "tool_use":
                trace.append({"role": "assistant", "type": "tool_use", "tool": c["name"], "input": c.get("input", {})})

    elif obj.get("type") == "result":
        final_result = obj
```

**What you get from `final_result`:**
```python
final_result.get("session_id")       # UUID for the session
final_result.get("duration_ms")      # Wall-clock time
final_result.get("num_turns")        # Number of agent turns
final_result.get("total_cost_usd")   # API cost
final_result.get("result")           # Final text response
```

---

## Streaming to Terminal with `jq`

```bash
claude -p "Write a poem" \
  --output-format stream-json \
  --verbose \
  --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
```

---

## Tool Permissions

Tools are blocked by default in `-p` mode. Use `--allowedTools` to auto-approve:

```bash
--allowedTools "Read"                    # Single tool
--allowedTools "Read,Edit,Bash"          # Multiple tools
--allowedTools "Bash(git diff *)"        # Scoped permission (trailing * = prefix match)
```

**Note:** Space before `*` matters for prefix matching.

---

## Error Handling Pattern

```python
try:
    result = subprocess.run([...], capture_output=True, text=True, timeout=180)

    if result.returncode != 0:
        # Check result.stderr for error details
        handle_error(result.stderr)

except subprocess.TimeoutExpired:
    handle_timeout()
except Exception as e:
    handle_generic(e)
```

---

## Environment Setup

- **Working directory matters** — Claude Code uses `cwd` as its project context. Pass `cwd=` in `subprocess.run()`.
- **Env vars** — Load `.env` before invoking if needed (API keys, config).
- **No interactive prompts** — `-p` mode never asks for user input; blocked tools silently fail unless `--allowedTools` grants them.

---

## Typical Trace Structure in Logs

```json
{
  "trace": [
    {"role": "assistant", "type": "text", "content": "I'll analyze this..."},
    {"role": "assistant", "type": "tool_use", "tool": "Read", "input": {"file_path": "/path/to/file"}},
    {"role": "assistant", "type": "text", "content": "Based on my analysis..."}
  ],
  "duration_ms": 12893,
  "num_turns": 9,
  "cost_usd": 0.0298,
  "session_id": "8c434073-0c94-4c16-bb1b-4aaa7fd92678"
}
```

---

## Things Not Covered Here (not found in repo)

- **`--json-schema`** for structured output validation — documented in Claude Code but not used in this repo
- **MCP server configuration** — no custom MCPs found
- **Multi-turn conversations** via `--continue`/`--resume` — not used
- **`--system-prompt`** as a separate flag — prompts are embedded in the user message instead
