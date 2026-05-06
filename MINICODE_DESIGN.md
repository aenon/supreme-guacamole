# Minicode Design — Minimal AI Coding Agent

> A conversational TUI coding agent in 3 files: one Python file, one markdown (system prompts), one .env (config).

---

## Table of Contents

1. [Philosophy & Constraints](#1-philosophy--constraints)
2. [File Structure](#2-file-structure)
3. [TUI Layout](#3-tui-layout)
4. [Agent Loop](#4-agent-loop)
5. [Tool System](#5-tool-system)
6. [System Prompt Loading](#6-system-prompt-loading)
7. [Context Window Management](#7-context-window-management)
8. [LLM Client & Model Scanning](#8-llm-client--model-scanning)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Implementation Plan (Roadmap)](#10-implementation-plan)

---

## 1. Philosophy & Constraints

**Core idea:** A coding agent you can drop into any project and start talking to immediately. The entire agent lives in 3 files. No npm install. No rust toolchain. `pip install textual httpx tiktoken` and you're done.

| File | Purpose | Lines (target) |
|---|---|---|
| `minicode.py` | Entire agent: TUI, agent loop, tools, LLM client | ~800 |
| `minicode.md` | System prompt definitions (one or more) | ~200 |
| `.env` | Configuration: API endpoint, model, keys | ~5 |

**Design constraints:**
- Zero external tool coupling — tools call `subprocess`, not SDKs
- Textual for TUI (Python's best terminal framework — reactive, async, accessible)
- httpx for API calls (no openai SDK dependency, full control over streaming)
- tiktoken for token counting (only heuristic dependency)
- System prompts are NEVER compacted during context management

---

## 2. File Structure

```
project/
├── minicode.py          # The agent
├── minicode.md          # System prompt(s) — loaded at startup
└── .env                 # Configuration

~/.minicode/
└── instructions.md      # User-level system prompt (loaded if exists)

.project/.minicode/      # (optional, in project root)
└── instructions.md      # Workspace-level system prompt (loaded if exists)
```

### 2.1 `.env` format

```env
# Required
MINICODE_API_BASE=https://api.openai.com/v1
MINICODE_API_KEY=sk-...

# Optional
MINICODE_MODEL=gpt-4o              # Auto-selects first if omitted
MINICODE_MAX_TOKENS=4096            # Output token budget
MINICODE_CONTEXT_RESERVE=8192       # Tokens reserved for output
MINICODE_AUTO_COMPACT=true          # Auto-compact on context pressure
MINICODE_WORKSPACE_PROMPT=.minicode/instructions.md  # Relative to project root
```

### 2.2 `minicode.md` format

A markdown file with one or more system prompts separated by `---` dividers and optional frontmatter:

```markdown
---
title: Default
id: default
---

You are minicode, a coding agent.

You have access to the following tools:
...
```

Additional prompts use a second `---` divider:

```markdown
---

---
title: Code Review
id: code-review
---

You are a code reviewer. Focus on security, correctness, and style.
```

The agent loads `minicode.md` at startup. The first prompt (no frontmatter or `id: default`) is the default. Users can switch prompts via `/prompt <id>` at runtime.

### 2.3 `minicode.py` module map

```
minicode.py
├── ── imports ──
│    Textual (App, Widgets, CSS)
│    httpx (async HTTP)
│    tiktoken (token counting)
│    subprocess, shlex (tool execution)
│    pathlib, os, json, re (stdlib)
├── ── constants ──
│    VERSION, APP_NAME, CRITICAL_THRESHOLD=0.80
├── ── Tool classes ──
│    Tool (ABC), ReadTool, WriteTool, EditTool,
│    GrepTool, GlobTool, BashTool, DiffTool
├── ── LLM Client ──
│    scan_models(), stream_completion(), count_tokens()
├── ── Prompt Loader ──
│    load_system_prompts(), load_user_prompt(), load_workspace_prompt()
├── ── Context Manager ──
│    estimate_usage(), needs_compact(), compact()
├── ── TUI Widgets ──
│    ChatMessage (rich.Text), MessageList (ScrollView),
│    ToolCallWidget (collapsible), StatusBar (Footer),
│    InputBar (TextArea)
├── ── App ──
│    MinicodeApp( textual.App )
│    ├── compose()   — layout assembly
│    ├── on_mount()  — startup: scan models, load prompts, greet
│    └── chat()      — main agent loop
└── ── main ──
     argparse (--prompt, --model, --debug)
     asyncio.run(MinicodeApp().run_async())
```

---

## 3. TUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ╔══════════════════════════════════════════════════════════╗ │
│  ║  Message History (ScrollView)                           ║ │
│  ║                                                         ║ │
│  ┌─────────────────────────────────────────────────────┐   ║ │
│  │  User: implement a fibonacci function                │   ║ │
│  └─────────────────────────────────────────────────────┘   ║ │
│  ┌─────────────────────────────────────────────────────┐   ║ │
│  │  Assistant: I'll create that for you.                │   ║ │
│  │                                                      │   ║ │
│  │  ┌─ Bash: cat fibonacci.py ──────────────────────┐  │   ║ │
│  │  │ def fib(n):                                    │  │   ║ │
│  │  │     if n <= 1: return n                       │  │   ║ │
│  │  │     return fib(n-1) + fib(n-2)                │  │   ║ │
│  │  └───────────────────────────────────────────────┘  │   ║ │
│  └─────────────────────────────────────────────────────┘   ║ │
│  ╚══════════════════════════════════════════════════════════╝ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  > /model gpt-4o                                        ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │  gpt-4o tokens: 12,450 / 128,000 (10%) │ cwd: ~/project ││
│  │  ● Connected     tools: 8/8     /help for commands      ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Layout composition (Textual `compose()`)

```python
def compose(self) -> ComposeResult:
    yield Header(show_clock=False)       # "minicode — gpt-4o"
    yield MessageList()                   # ScrollView with chat history
    yield InputBar()                      # TextArea for prompt input
    yield StatusBar()                     # Footer widget, reactive
```

### 3.2 StatusBar design

A footer widget that only renders the bottom strip. Reactive fields:

| Field | Content | Updates when |
|---|---|---|
| Model name | `gpt-4o` | Model switches |
| Tokens | `12,450 / 128,000 (10%)` | After each turn; color-coded |
| CWD | `~/project` | On `/cd` |
| Connection | `● Connected` / `○ Offline` | After health check |
| Tools | `tools: 8/8` | Tool registry changes |
| Context status | `[COMPACTING]` / `[WARN 85%]` | When approaching limit |

Color coding for token percentage:
- `< 60%`: green
- `60-80%`: yellow (warning)
- `> 80%`: red (critical, trigger compaction)

### 3.3 Slash commands (in InputBar)

| Command | Action |
|---|---|
| `/model <name>` | Switch model (re-scan available list) |
| `/prompt <id>` | Switch system prompt |
| `/compact` | Manual trigger context compaction |
| `/clear` | Clear conversation (keep system prompt) |
| `/cd <path>` | Change working directory |
| `/env` | Show current env/config |
| `/help` | Show command list |
| `/quit` | Exit |

### 3.4 Tool call rendering

Each tool call gets a bordered collapsible block:

```
┌─ Read: src/main.py ─────────────────────────────┐
│  │ 1│ import os                                  │
│  ...                                              │
│  │24│ def main():                                │
└──────────────────────────────────────────────────┘
```

Non-mutating tools (Read, Grep, Glob): short, auto-expanded after 3s
Mutating tools (Write, Edit, Bash, Diff): shown with a progress indicator, collapsed to one line when done

### 3.5 Stream rendering

Assistant responses stream character-by-character into the message list. The active message is highlighted with a cursor indicator. When a tool call is encountered mid-stream, the streaming pauses, the tool call widget appears with a spinner, and streaming resumes when the tool result arrives.

---

## 4. Agent Loop

```
┌──────────────────────────────────────────────┐
│                  Agent Loop                    │
│                                                │
│  User sends message                            │
│       │                                        │
│       ▼                                        │
│  Check context usage                           │
│       │                                        │
│       ├── > 80% → auto-compact history         │
│       │         (skip system prompts!)         │
│       │                                        │
│       ▼                                        │
│  Build message array:                          │
│   [sys_prompts...] + [history...] + [user_msg] │
│       │                                        │
│       ▼                                        │
│  Stream completion from API                    │
│       │                                        │
│       ▼                                        │
│  Parse stream for:                             │
│   ├── text content → render directly           │
│   ├── tool_use start → create ToolCallWidget   │
│   │   └── execute tool → render result         │
│   │   └── feed result back to model loop       │
│   └── tool_use end → collapse widget           │
│       │                                        │
│       ▼                                        │
│  Append assistant message to history           │
│       │                                        │
│       ▼                                        │
│  Update StatusBar (token count, context %)     │
│       │                                        │
│       └── Back to waiting for user input       │
└──────────────────────────────────────────────┘
```

### 4.1 Streaming protocol (SSE)

The agent uses OpenAI-compatible streaming (`stream=True`). Each SSE event is one of:

```
event: data
data: {"id":"...","object":"chat.completion.chunk", ...}

event: data
data: {"choices":[{"delta":{"content":"Hello"}}]}

event: data
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_...","function":{"name":"read","arguments":"{\"path\":\"src/main.py\"}"}}]}}]}
```

Two key details:
- Tool call accumulation: arguments arrive as deltas across multiple chunks. The agent appends arguments until the tool call is complete, then executes.
- Tool call streaming: native tool_use semantics (model stops generating text, issues tool call block).

### 4.2 Tool loop (automatic)

After the model issues tool calls, the agent:
1. Executes all parallel tool calls simultaneously (independent tools)
2. Collects results
3. Sends results back to the model as `tool_result` messages
4. The model may respond with more text, more tool calls, or finish

This loop repeats until the model responds with text only (no tool calls). Maximum 10 consecutive tool rounds to prevent infinite loops.

---

## 5. Tool System

### 5.1 Tool interface

```python
@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None

class Tool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult: ...
    
    def max_output_chars(self) -> int:
        return 4000  # Truncation per tool
    
    def to_tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
```

### 5.2 Tool roster

| Tool | Name | Parameters | Description |
|---|---|---|---|
| **Read** | `read` | `path: str`, `offset?: int`, `limit?: int` | Read file contents with line numbers |
| **Write** | `write` | `path: str`, `content: str` | Create/overwrite a file |
| **Edit** | `edit` | `path: str`, `old_string: str`, `new_string: str` | Find-and-replace in file (fuzzy match) |
| **Grep** | `grep` | `pattern: str`, `path?: str`, `file_glob?: str` | Search file contents with regex |
| **Glob** | `glob` | `pattern: str`, `path?: str` | Find files by name pattern |
| **Bash** | `bash` | `command: str`, `timeout?: int` | Execute shell command |
| **Diff** | `diff` | `path?: str` | Show git diff for files |

### 5.3 Tool-specific behaviors

**Read:**
- Output is clipped at `max_output_chars` (4000)
- Shows line numbers in the output
- Directory listing if path is a directory

**Write:**
- Creates parent directories automatically
- No overwrite confirmation (the model asked for it)

**Edit:**
- Uses difflib for fuzzy matching (like Hermes Agent's patch tool)
- Returns unified diff of the change
- Falls back to Write with a warning if exact match isn't found

**Grep:**
- Calls `rg` (ripgrep) directly via subprocess
- Lies about line numbers — shows them in the output
- Falls back to `grep -rn` if ripgrep not available

**Glob:**
- Uses Python's `glob.glob` or `pathlib.Path.rglob`
- Results sorted by modification time (newest first)
- Limited to 50 results

**Bash:**
- Uses `subprocess.run` with `shlex` for safe argument handling
- Running in project root directory (user's cwd)
- 60-second default timeout, configurable
- Output truncated at 8000 chars
- No pseudo-terminal (PTY-less for simplicity) — fine for most commands

**Diff:**
- Runs `git diff` (with optional path filter) — uses `--no-color` for clean output
- Shows staged and unstaged changes

### 5.4 Permission system

By default, all tools execute immediately (no approval prompts). This is a deliberate design choice for the minimal version — the user can see every tool call in the TUI and interrupt with Ctrl+C if something looks wrong. A future `--safe` mode could add tool approval.

---

## 6. System Prompt Loading

### 6.1 Load order (precedence)

```
1. Built-in defaults from minicode.md (title="Default" or first prompt)
2. User-level: ~/.minicode/instructions.md        (APPENDED after default)
3. Workspace-level: .minicode/instructions.md      (APPENDED after user-level)
4. AGENTS.md / CLAUDE.md (from project root)       (APPENDED after workspace)
5. /prompt <id> runtime switch                     (REPLACES #1, keeps #2-#4)
```

All layers are combined into the final system prompt. Only the base prompt (from `minicode.md`) can be swapped at runtime — the user and workspace layers stay.

### 6.2 Prompt loading code sketch

```python
def load_all_prompts(cwd: str) -> list[dict]:
    """Returns list of system prompt message dicts."""
    prompts = []
    
    # 1. Built-in prompts from minicode.md
    prompts.extend(parse_minicode_md())
    
    # 2. User-level
    user_prompt_path = Path.home() / ".minicode" / "instructions.md"
    if user_prompt_path.exists():
        prompts.append({
            "role": "system",
            "content": user_prompt_path.read_text()
        })
    
    # 3. Workspace-level
    ws_prompt_path = Path(cwd) / ".minicode" / "instructions.md"
    if ws_prompt_path.exists():
        prompts.append({
            "role": "system",
            "content": ws_prompt_path.read_text()
        })
    
    # 4. AGENTS.md / CLAUDE.md (hierarchical discovery)
    for root in walk_up(cwd):
        for name in ("AGENTS.md", "CLAUDE.md"):
            p = root / name
            if p.exists():
                prompts.append({
                    "role": "system",
                    "content": p.read_text()
                })
    
    return prompts
```

### 6.3 Markdown parsing (`parse_minicode_md`)

```python
def parse_minicode_md(path="minicode.md") -> list[dict]:
    """
    Parses minicode.md into a list of system prompt dicts.
    
    Format: prompts separated by \n---\n dividers.
    Optional YAML frontmatter between --- delimiters at start of each prompt.
    """
```

Multiple prompts in one file allow the user to define personas (e.g., "coder", "reviewer", "architect") and switch between them at runtime.

### 6.4 `AGENTS.md` / `CLAUDE.md` hierarchical search

```python
def walk_up(start: Path) -> Iterator[Path]:
    """Yield directories from start to filesystem root."""
    for parent in [start, *start.parents]:
        yield parent
        if parent == parent.parent:  # reached root
            break
```

This matches the convention used by OpenCode, Claude Code, and Codex CLI — the most specific (closest to cwd) wins.

---

## 7. Context Window Management

### 7.1 Strategy overview

System prompt messages are NEVER compacted. Always preserved verbatim.

Conversation history (user + assistant messages) is selectively compacted when the context window fills up.

### 7.2 Token estimation

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # Works for most OpenAI-compatible models

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def estimate_message_tokens(msg: dict) -> int:
    """Approximate tokens for a message dict."""
    base = 3  # per-message overhead
    content = msg.get("content", "") or ""
    if isinstance(content, str):
        base += count_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if block.get("type") == "text":
                base += count_tokens(block["text"])
            elif block.get("type") == "image_url":
                base += 1000  # rough image estimate
    # Tool call overhead
    if "tool_calls" in msg:
        base += sum(count_tokens(tc["function"]["arguments"]) 
                    for tc in msg["tool_calls"])
    return base
```

### 7.3 Context pressure detection

After each assistant turn, the agent estimates total tokens across the full message array:

```python
class ContextManager:
    def __init__(self):
        self.context_window: int = 128_000  # Detected from model info
        self.reserve: int = 8_192           # Space for output
        self.system_token_count: int = 0    # Cached after load
        
    def usable_window(self) -> int:
        return self.context_window - self.reserve
    
    def pressure(self, messages: list[dict]) -> float:
        total = self.system_token_count
        for msg in messages:
            if msg["role"] == "system":
                continue  # Already counted
            total += estimate_message_tokens(msg)
        return total / self.usable_window()
```

| Pressure | Action |
|---|---|
| `< 0.60` | Normal — no action |
| `0.60 - 0.80` | Show yellow warning in status bar |
| `0.80 - 0.95` | Show red warning, auto-compact after current turn |
| `> 0.95` | Block new input until compaction completes |

### 7.4 Compaction strategy

When compaction triggers:

1. **Identify system prompts** (skip entirely)
2. **Identify "protected" messages**: last 3 user-assistant exchanges (keep intact)
3. **For older messages**: summarize each exchange into a single sentence using a lightweight summary call
4. **For tool results in old exchanges**: replace full output with `[Tool result truncated: N chars]`
5. **Replace old exchanges** with compacted equivalents in the history

```python
async def compact(self, messages: list[dict]) -> list[dict]:
    # Find system prompt boundaries
    sys_msgs = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]
    
    # Protect last 3 exchanges (6 messages + their tool results)
    protected = history[-12:] if len(history) >= 12 else history[:]
    to_compact = history[:-len(protected)] if len(history) > 12 else []
    
    if not to_compact:
        return messages
    
    # Summarize old exchanges
    summary_text = await self.summarize_exchanges(to_compact)
    
    # Inject compacted summary
    compacted = [{
        "role": "system",
        "content": f"[Compacted context from previous turns: {summary_text}]"
    }]
    
    return sys_msgs + compacted + protected
```

### 7.5 Summary API call

The compaction summary uses the same model (cheap, fast):

```python
async def summarize_exchanges(self, exchanges: list[dict]) -> str:
    summary_prompt = (
        "Summarize the following conversation exchanges concisely. "
        "Capture the key decisions, file changes, and findings. "
        "This summary replaces the full history for future turns.\n\n"
    )
    summary_messages = [
        {"role": "system", "content": summary_prompt},
        *exchanges
    ]
    
    result = await self.client.complete(
        messages=summary_messages,
        max_tokens=512,
        stream=False
    )
    return result.choices[0].message.content
```

### 7.6 Manual compaction

User triggers via `/compact` command. Same logic as auto-compact but always runs regardless of pressure.

### 7.7 Token count display

Status bar shows: `tokens: 12,450 / 128,000 (10%)`

- Numerator: estimated tokens of ALL messages including system prompts
- Denominator: full context window (not usable_window)
- Percentage: pressure against usable_window

---

## 8. LLM Client & Model Scanning

### 8.1 Model scanning (startup)

On startup, the agent calls `GET {base_url}/models` to discover available models:

```python
async def scan_models() -> list[str]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data["data"]]
```

The agent filters for likely chat models (contains `gpt`, `claude`, `gemini`, `llama`, `qwen`, `mistral`, `deepseek`, etc.) and presents them in a picker if `MINICODE_MODEL` is not set.

The context window size is detected from:
1. A hardcoded mapping of known model prefixes → context windows
2. If unknown, defaults to 128K and reports `[estimated]` in the status bar

```python
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "claude-sonnet-4": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-opus-4": 200_000,
    "gemini": 1_000_000,  # approximate
    "deepseek": 64_000,
    "llama-3": 8_192,     # conservative for local models
    "qwen": 32_000,
}
```

### 8.2 Streaming completion

```python
async def stream_completion(
    messages: list[dict],
    tools: list[dict],
    model: str,
    max_tokens: int = 4096,
):
    body = {
        "model": model,
        "messages": messages,
        "tools": tools if tools else None,
        "stream": True,
        "max_tokens": max_tokens,
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        ) as response:
            response.raise_for_status()
            
            current_tool_call = None
            
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                    
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                    
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"]
                
                # Text content
                if "content" in delta and delta["content"]:
                    yield {"type": "text", "content": delta["content"]}
                
                # Tool calls
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        if tc.get("id"):
                            current_tool_call = {"id": tc["id"], "name": "", "args": ""}
                        
                        if "function" in tc:
                            if tc["function"].get("name"):
                                current_tool_call["name"] = tc["function"]["name"]
                            if tc["function"].get("arguments"):
                                current_tool_call["args"] += tc["function"]["arguments"]
                        
                        # Check for finish reason on this tool call
                        finish = chunk["choices"][0].get("finish_reason")
                        if finish == "tool_calls" and current_tool_call:
                            yield {
                                "type": "tool_use",
                                "id": current_tool_call["id"],
                                "name": current_tool_call["name"],
                                "arguments": current_tool_call["args"],
                            }
                            current_tool_call = None
```

### 8.3 Tool result injection

After executing a tool, the result is injected back into the conversation:

```python
tool_result_message = {
    "role": "tool",
    "tool_call_id": tool_call_id,
    "content": truncated_output  # clipped at max_output_chars
}
```

The agent loop sends this back to the model in a non-streaming call (or streaming call with only the tool result message + history). This continues until the model produces a text response.

---

## 9. Data Flow Diagrams

### 9.1 Startup flow

```
                     minicode.py startup
                            │
                            ▼
                   Load .env configuration
                            │
                            ▼
                  Load minicode.md prompts
                            │
                            ▼
                  Scan {base_url}/models
                            │
                            ▼
                  Resolve model (env / picker / first)
                            │
                            ▼
                  Load user prompt (~/.minicode/instructions.md)
                            │
                            ▼
                  Load workspace prompt (.minicode/instructions.md)
                            │
                            ▼
                  Discover AGENTS.md / CLAUDE.md
                            │
                            ▼
                  Initialize ContextManager
                            │
                            ▼
                  Mount TUI → display welcome message
                            │
                            ▼
                  ┌── Waiting for user input ──┐
```

### 9.2 Turn flow

```
User types message in InputBar + Enter
            │
            ▼
    Add user message to history
            │
            ▼
    ContextManager.pressure() ──> >80%? ──> auto-compact
            │
            ▼
    Build full message array:
    [system prompts] + [history] + [user msg]
            │
            ▼
    stream_completion(messages, tools)
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Process SSE stream                    │
    │                                        │
    │  text delta ──> render in TUI          │
    │  tool_use   ──> render widget          │
    │               ──> execute tool         │
    │               ──> render result        │
    │               ──> inject result        │
    │               ──> continue loop        │
    │                                        │
    │  finish_reason:stop ──> done           │
    └───────────────────────────────────────┘
            │
            ▼
    Append assistant message to history
            │
            ▼
    Update StatusBar (token %, tool status)
            │
            ▼
    ┌── Waiting for user input ──┐
```

### 9.3 Context window state machine

```
                     ┌──────────────┐
                     │   Normal     │  < 60%
                     │  (green)     │
                     └──────┬───────┘
                            │ pressure > 60%
                            ▼
                     ┌──────────────┐
                     │   Warning    │  60-80%
                     │  (yellow)    │
                     └──────┬───────┘
                            │ pressure > 80%
                            ▼
                     ┌──────────────┐
                     │   Critical   │  > 80%
                     │  (red)       │
                     ├──────────────┤
                     │ After turn:  │
                     │ auto-compact │ ──> Normal (if compact successful)
                     │              │     Warning (if partial)
                     └──────────────┘
                            │ pressure > 95%
                            ▼
                     ┌──────────────┐
                     │   Blocked    │  > 95%
                     │  [COMPACT]   │  Input disabled
                     │  (blinking)  │  Force compact → Normal
                     └──────────────┘
```

---

## 10. Implementation Plan (Roadmap)

### Phase 1: Core Infrastructure (~400 lines)

- [ ] `.env` loading with `python-dotenv` or manual parsing
- [ ] `minicode.md` prompt parser (frontmatter + markdown body)
- [ ] LLM client: model scanning, streaming, non-streaming
- [ ] Token estimation with tiktoken
- [ ] ContextManager with pressure detection

### Phase 2: TUI Shell (~300 lines)

- [ ] Textual `App` with compose layout
- [ ] `MessageList` widget (ScrollView with rich.Text messages)
- [ ] `InputBar` widget (TextArea with command detection)
- [ ] `StatusBar` widget (reactive footer)
- [ ] Streaming message rendering (append text as it arrives)
- [ ] Tool call widget (collapsible bordered block)
- [ ] Slash commands: `/model`, `/prompt`, `/compact`, `/clear`, `/help`, `/quit`

### Phase 3: Tool System (~200 lines)

- [ ] `Tool` ABC with JSON Schema definition
- [ ] `ReadTool` — file reading with line numbers
- [ ] `WriteTool` — file creation/overwrite
- [ ] `EditTool` — fuzzy find-and-replace with diff output
- [ ] `GrepTool` — ripgrep subprocess wrapper
- [ ] `GlobTool` — file discovery by pattern
- [ ] `BashTool` — shell command execution
- [ ] `DiffTool` — git diff display
- [ ] Tool result truncation (per-tool max chars)

### Phase 4: Agent Loop (~100 lines)

- [ ] Main chat loop (user input → stream → tool execution → continue)
- [ ] Automatic tool loop (max 10 rounds)
- [ ] Tool result injection back to model
- [ ] Compaction: summarization call + history replacement
- [ ] Error handling: API errors, tool failures, streaming interruptions

### Phase 5: Polish & DX (~100 lines)

- [ ] Startup model picker (if no model configured)
- [ ] Welcome message with key bindings
- [ ] `/cd` command for directory switching
- [ ] AGENTS.md / CLAUDE.md hierarchical discovery
- [ ] User prompt (`~/.minicode/instructions.md`)
- [ ] Workspace prompt (`.minicode/instructions.md`)
- [ ] Terminal resize handling
- [ ] Interrupt handling (Ctrl+C during tool execution)

### Phase 6: Nice-to-Haves

- [ ] Syntax highlighting in tool output (Rich syntax)
- [ ] Multi-line input support (Shift+Enter for newlines)
- [ ] Command history in InputBar (Up/Down arrows)
- [ ] Session persistence (save/restore conversation to JSON)
- [ ] MCP server support
- [ ] `--safe` mode with tool approval dialogs

---

## Appendix A: Dependency Analysis

| Dependency | Version | Size | Purpose | Alternative |
|---|---|---|---|---|
| `textual` | ≥0.52 | ~2MB | TUI framework | prompt_toolkit (no reactive widgets) |
| `httpx` | ≥0.27 | ~1MB | Async HTTP client | aiohttp (more complex API) |
| `tiktoken` | ≥0.7 | ~500KB | Token counting | rough char/4 estimation (less accurate) |
| `rich` | ≥13.0 | ~1MB | Text rendering | Already a textual dependency |

**Zero-dependency path** (if textual is too heavy):
- Replace textual with `prompt_toolkit` + `rich`
- Would lose reactive widgets but gain smaller install
- ~500 lines instead of ~300 for TUI

## Appendix B: Key Design Decisions

| Decision | Rationale |
|---|---|
| **No openai SDK** | httpx gives full control over streaming, no SDK version issues, works with any OpenAI-compatible endpoint |
| **Textual for TUI** | Best Python TUI framework. Reactive, async-native, accessible. Same mental model as React/SolidJS |
| **System prompts never compacted** | User-defined behavior should never be lost. Only the conversation history gets summarized |
| **tiktoken for counting** | cl100k_base covers most modern models. Better than char/word estimation |
| **No approval prompts** | Keeps flow fast. User sees every tool call in TUI and can Ctrl+C to stop |
| **Tool calls accumulated by index** | Models may send tool call arguments in multiple chunks; index-based accumulation is the standard approach |
| **ripgrep for search** | Much faster than `grep`, available on most systems, handles large codebases |
| **Markdown-first prompts** | Users already write in markdown. No XML, no YAML blobs. Frontmatter for metadata |
| **AGENTS.md / CLAUDE.md convention** | Follows the existing ecosystem. Users who have these files get them for free |
