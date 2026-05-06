# Minicode — Implementation Plan

## Repository Structure

```
minicode/
├── minicode.py          # ~800 lines — the agent
├── minicode.md          # ~200 lines — system prompt definitions
├── .env.example         # ~10 lines — config template
└── pyproject.toml       # Dependency declaration
```

Dependencies: `textual>=0.52`, `httpx>=0.27`, `tiktoken>=0.7`, `python-dotenv>=1.0`

---

## Phase 1: Skeleton & Imports (~60 lines)

Write the file structure, imports, constants, and main entry point. This is the "compiles and shows a blank TUI" phase.

### 1.1 Imports section

```
stdlib: asyncio, json, os, re, subprocess, shlex, textwrap
         from dataclasses import dataclass, field
         from pathlib import Path
         from collections.abc import AsyncIterator
         from typing import Any
external: httpx, tiktoken, dotenv
textual: App, ComposeResult, events, reactive
         from textual.app import App
         from textual.widgets import Header, Footer, Input, Static, RichLog
         from textual.containers import Vertical, Horizontal
         from textual.reactive import reactive
         from textual.screen import Screen
```

### 1.2 Constants

```python
VERSION = "0.1.0"
APP_NAME = "minicode"
CRITICAL_PRESSURE = 0.80
BLOCK_PRESSURE = 0.95
MAX_TOOL_ROUNDS = 10
TOOL_OUTPUT_MAX = 4000
BASH_OUTPUT_MAX = 8000
```

### 1.3 Main entry point

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description="minicode — minimal AI coding agent")
    parser.add_argument("--model", help="Model override (e.g. gpt-4o)")
    parser.add_argument("--prompt", default="default", help="System prompt ID")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(MinicodeApp(config=args).run_async())

if __name__ == "__main__":
    main()
```

**Verification checkpoint:** `python minicode.py` opens a blank Textual app with Header/Footer that doesn't crash.

---

## Phase 2: Config & Prompt Loader (~80 lines)

### 2.1 Config loader

```python
@dataclass
class Config:
    api_base: str
    api_key: str
    model: str
    max_tokens: int = 4096
    context_reserve: int = 8192
    auto_compact: bool = True
    workspace_prompt: str = ".minicode/instructions.md"
    debug: bool = False

def load_config(args) -> Config:
    """Load .env then override with CLI args."""
```

Search order: `.env` in cwd → parent dirs → `~/.minicode/.env`.

### 2.2 Prompt parser

```python
@dataclass
class SystemPrompt:
    id: str
    title: str
    content: str

def parse_minicode_md(path: str | Path = "minicode.md") -> list[SystemPrompt]:
    """
    Parse minicode.md into system prompts.
    
    Format: documents separated by \n---\n
    Each document may have YAML frontmatter between --- lines.
    
    Returns list of SystemPrompt objects. First one is the default.
    If file doesn't exist, returns a single built-in default prompt.
    """
```

### 2.3 Prompt hierarchy loader

```python
def collect_prompts(cwd: str, selected_id: str) -> list[dict]:
    """
    Returns list of system message dicts in final order.
    
    1. Selected prompt from minicode.md (matched by id)
    2. ~/.minicode/instructions.md (if exists)
    3. cwd/.minicode/instructions.md (if exists)
    4. AGENTS.md / CLAUDE.md from hierarchical walk (if exists)
    """
```

**Verification checkpoint:** Agent prints the prompt count and selected prompt ID on startup.

---

## Phase 3: LLM Client (~70 lines)

### 3.1 Model scanning

```python
async def scan_models(config: Config) -> list[str]:
    """GET /v1/models, filter for likely chat models, return sorted list."""
    
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000, "gpt-4o-mini": 128_000,
    "gpt-4": 8_192, "gpt-3.5-turbo": 16_385,
    "claude-sonnet-4": 200_000, "claude-3.5-sonnet": 200_000,
    "claude-3-haiku": 200_000, "claude-opus-4": 200_000,
    "gemini": 1_000_000, "deepseek": 64_000,
    "llama": 8_192, "qwen": 32_000, "mistral": 32_000,
}

def resolve_context_window(model: str) -> int:
    """Match model against known prefixes, default 128K."""
```

### 3.2 Token counting

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def estimate_messages_tokens(messages: list[dict]) -> int:
    """Approximate total tokens across a message list."""
```

### 3.3 Streaming completion

```python
async def stream_completion(
    config: Config,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> AsyncIterator[dict]:
    """
    POST /v1/chat/completions with stream=True
    
    Yields dicts:
      {"type": "text", "content": str}
      {"type": "tool_use", "id": str, "name": str, "arguments": str}
      {"type": "done", "finish_reason": str, "usage": dict | None}
    """
```

Key implementation detail: tool call arguments arrive as streaming deltas indexed by `tc["index"]`. Accumulate them per-index, yield full tool_use when `finish_reason == "tool_calls"`.

### 3.4 Non-streaming completion

```python
async def complete(
    config: Config,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 512,
) -> dict:
    """Non-streaming completion for compaction summaries."""
```

**Verification checkpoint:** Run a test call with `curl`-equivalent behavior — send a simple message, get a response back, verify streaming works.

---

## Phase 4: Tool System (~150 lines)

### 4.1 Base class

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
    
    def to_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": self.parameters, "required": list(self.parameters.keys())}
            }
        }
    
    def truncate(self, text: str) -> str:
        max_chars = BASH_OUTPUT_MAX if self.name == "bash" else TOOL_OUTPUT_MAX
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return text
```

### 4.2 Tool implementations

Each tool in its own method on a `ToolRegistry` class:

```
read(path, offset=1, limit=500)    → Path.read_text() with line slicing
write(path, content)               → Path.write_text(), create parents
edit(path, old_string, new_string) → difflib.SequenceMatcher fuzzy match, return diff
grep(pattern, path=".", file_glob=None) → subprocess.run(["rg", ...]), fallback to grep -rn    
glob(pattern, path=".")            → Path(path).rglob(pattern), sorted by mtime, limit 50
bash(command, timeout=60)          → subprocess.run with shlex.split, timeout
diff(path=None)                    → subprocess.run(["git", "diff", ...]), --no-color
```

### 4.3 Tool registry

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self.register(ReadTool())
        self.register(WriteTool())
        self.register(EditTool())
        self.register(GrepTool())
        self.register(GlobTool())
        self.register(BashTool())
        self.register(DiffTool())
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def definitions(self) -> list[dict]:
        return [t.to_definition() for t in self._tools.values()]
    
    async def execute(self, name: str, args: dict) -> ToolResult:
        if name not in self._tools:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")
        return await self._tools[name].execute(**args)
```

**Verification checkpoint:** Write a test script that imports ToolRegistry and runs each tool with reasonable arguments. All should return valid results.

---

## Phase 5: Context Manager (~50 lines)

### 5.1 Implementation

```python
class ContextManager:
    def __init__(self, context_window: int, reserve: int = 8192):
        self.context_window = context_window
        self.reserve = reserve
    
    @property
    def usable(self) -> int:
        return self.context_window - self.reserve
    
    def pressure(self, system_tokens: int, messages: list[dict]) -> float:
        total = system_tokens
        for msg in messages:
            if msg["role"] == "system":
                continue
            total += estimate_message_tokens(msg)
        return total / self.usable
    
    def needs_compact(self, system_tokens: int, messages: list[dict]) -> bool:
        return self.pressure(system_tokens, messages) >= CRITICAL_PRESSURE
    
    def compact(self, messages: list[dict], system_tokens: int) -> list[dict]:
        """
        Compact non-system messages.
        - Keep last 3 exchanges (1 exchange = user + assistant + tool_results)
        - Summarize everything older via non-streaming LLM call
        - Return new message list with compacted summary injected
        """
```

The compact method is the trickiest piece. Implementation steps:
1. Split messages into system + non-system
2. Count system tokens (to verify nothing gets compacted)
3. Find protected range (last 12 non-system messages = ~3 exchanges)
4. Extract older messages for summarization
5. Build summary prompt, call `complete()`
6. Return: system_msgs + [compacted content] + protected_msgs

### 5.2 Summary prompt template (inlined)

```python
SUMMARY_PROMPT = """You are a conversation summarizer. Condense the following exchanges into 2-3 sentences. Focus on:
- What the user asked for
- What code/files were discussed or changed
- Key decisions made
- Any errors or blockers encountered

Exchanges to summarize:
"""
```

**Verification checkpoint:** Feed a few mock exchanges into compact(), verify the output structure preserves system prompts and keeps last 3 exchanges intact.

---

## Phase 6: TUI Widgets (~200 lines)

This is the most text-heavy phase. Layout:

```
┌─────────────────────────────────────────────┐
│ Header: "minicode — gpt-4o"                 │
├─────────────────────────────────────────────┤
│ MessageList (VerticalScroll)                │
│                                             │
│  ┌──── User message ──────────────────────┐ │
│  │ implement fibonacci                    │ │
│  └────────────────────────────────────────┘ │
│  ┌──── Assistant ─────────────────────────┐ │
│  │ I'll write that.                        │ │
│  │ ┌─ Bash: cat fib.py ────────────────┐  │ │
│  │ │ def fib(n): ...                   │  │ │
│  │ └───────────────────────────────────┘  │ │
│  └────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│ InputBar                                    │
├─────────────────────────────────────────────┤
│ StatusBar: model | tokens% | cwd | status   │
└─────────────────────────────────────────────┘
```

### 6.1 MessageList widget

```python
class MessageList(VerticalScroll):
    """Scrollable chat history. Auto-scrolls to bottom on new messages."""
    
    def add_user_message(self, content: str):
        """Add a user message bubble."""
        
    def start_assistant_message(self) -> int:
        """Create a container for a new assistant message. Returns index."""
        
    def append_content(self, index: int, text: str):
        """Append streaming text to assistant message at index."""
        
    def add_tool_call(self, index: int, name: str, args: str):
        """Add a collapsible tool call widget to assistant message."""
        
    def update_tool_result(self, tool_call_widget, result: ToolResult):
        """Update tool call widget with execution result."""
```

Each message is a `Static` widget with Rich markup rendering (or a `RichLog` for assistant messages to support streaming).

### 6.2 InputBar widget

```python
class InputBar(Input):
    """Text input with slash command detection."""
    
    BINDINGS = [
        ("enter", "submit"),
        ("up", "history_back"),
        ("down", "history_forward"),
    ]
    
    @property
    def is_command(self) -> bool:
        return self.value.startswith("/")
    
    def action_submit(self):
        value = self.value.strip()
        if not value:
            return
        if value.startswith("/"):
            self.post_message(CommandSubmitted(value))
        else:
            self.post_message(MessageSubmitted(value))
        self.clear()
```

### 6.3 StatusBar widget

```python
class StatusBar(Footer):
    """Bottom status bar with reactive token counter."""
    
    model_name = reactive("")
    token_count = reactive(0)
    context_window = reactive(128_000)
    pressure = reactive(0.0)
    cwd = reactive("")
    connected = reactive(True)
    tool_count = reactive(0)
    compacting = reactive(False)
    
    def render(self) -> str | Text:
        """Build the status bar text with color-coded token counter."""
        color = "green" if self.pressure < 0.6 else "yellow" if self.pressure < 0.8 else "red"
        tokens_str = f"[{color}]{self.token_count:,} / {self.context_window:,} ({self.pressure:.0%})[/]"
        parts = [
            f"[bold]{self.model_name}[/]",
            f"tokens: {tokens_str}",
            f"cwd: {self.cwd}",
            f"{'●' if self.connected else '○'} {'Connected' if self.connected else 'Disconnected'}",
            f"tools: {self.tool_count}",
        ]
        if self.compacting:
            parts.insert(1, "[bold yellow][COMPACTING][/]")
        return Text(" │ ".join(parts))
```

### 6.4 App compose

```python
class MinicodeApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    MessageList {
        height: 1fr;
        border: none;
    }
    InputBar {
        dock: bottom;
        height: 3;
        margin: 0 1;
    }
    StatusBar {
        dock: bottom;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield MessageList()
        yield InputBar(placeholder="Ask minicode to do something...")
        yield StatusBar()
```

### 6.5 Event wiring (in App)

```python
def on_mount(self):
    """Startup: scan models, load prompts, show welcome."""
    
def on_message_submitted(self, event: MessageSubmitted):
    """Handle user message: add to chat, start agent loop."""
    
def on_command_submitted(self, event: CommandSubmitted):
    """Handle /commands: /model, /prompt, /compact, /clear, /cd, /help, /quit."""
```

**Verification checkpoint:** Full TUI launches, accepts text input, shows rendered messages, slash commands work, status bar updates.

---

## Phase 7: Agent Loop (~100 lines)

### 7.1 Main loop

```python
async def run_agent_loop(self, user_message: str):
    """Run one user message through the full agent loop."""
    
    # 1. Add user message to history
    self.message_list.add_user_message(user_message)
    self.history.append({"role": "user", "content": user_message})
    
    # 2. Check context pressure
    sys_tokens = sum(count_tokens(s["content"]) for s in self.system_msgs)
    if self.context_manager.needs_compact(sys_tokens, self.history):
        self.status_bar.compacting = True
        self.history = await self.context_manager.compact(self.history, sys_tokens)
        self.status_bar.compacting = False
    
    # 3. Build messages array
    messages = self.system_msgs + self.history
    
    # 4. Agent loop (tool rounds)
    tool_round = 0
    while tool_round < MAX_TOOL_ROUNDS:
        # Stream completion
        msg_index = self.message_list.start_assistant_message()
        assistant_msg = {"role": "assistant", "content": "", "tool_calls": []}
        
        async for event in stream_completion(self.config, messages, self.tools.definitions()):
            if event["type"] == "text":
                self.message_list.append_content(msg_index, event["content"])
                assistant_msg["content"] += event["content"]
            
            elif event["type"] == "tool_use":
                tool_widget = self.message_list.add_tool_call(msg_index, event["name"], event["arguments"])
                result = await self.tools.execute(event["name"], json.loads(event["arguments"]))
                self.message_list.update_tool_result(tool_widget, result)
                assistant_msg["tool_calls"].append({
                    "id": event["id"],
                    "type": "function",
                    "function": {"name": event["name"], "arguments": event["arguments"]}
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": event["id"],
                    "content": result.output if result.success else result.error
                })
        
        self.history.append(assistant_msg)
        
        # Detect finish
        if not assistant_msg["tool_calls"]:
            break  # Text-only response → done
        tool_round += 1
    
    # 5. Update status bar
    self.status_bar.token_count = sum(estimate_message_tokens(m) for m in self.history) + sys_tokens
    self.status_bar.pressure = self.context_manager.pressure(sys_tokens, self.history)
```

### 7.2 Parallel tool execution

Before the tool loop round, gather all tool calls from the assistant message and execute independent ones concurrently:

```python
async def execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
    """Execute parallel-safe tools concurrently."""
    tasks = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        tasks.append(self.tools.execute(name, args))
    
    results = await asyncio.gather(*tasks)
    return [
        {"role": "tool", "tool_call_id": tc["id"], "content": r.output if r.success else r.error}
        for tc, r in zip(tool_calls, results)
    ]
```

**Verification checkpoint:** Full end-to-end: user types a question → agent streams response → tools execute → results appear → status bar updates.

---

## Phase 8: minicode.md System Prompt (~200 lines)

The default system prompt. This is the "persona" that defines the coding agent's behavior.

```markdown
---
title: Default Coding Agent
id: default
---

You are minicode, a minimal AI coding agent. You help the user write, read, and
modify code in their project.

## Available Tools

You have access to the following tools:

1. **read(path, offset?, limit?)** — Read a file. Shows line numbers. Use this
   to understand existing code.

2. **write(path, content)** — Create or overwrite a file. Creates parent
   directories. Use this to create new files or replace entire files.

3. **edit(path, old_string, new_string)** — Find-and-replace edit. Use this for
   targeted changes to existing files. Prefer this over write() when only
   changing a few lines.

4. **grep(pattern, path?, file_glob?)** — Search file contents with regex.
   Use this to find where things are defined or referenced.

5. **glob(pattern, path?)** — Find files by name pattern (e.g., "*.py").
   Use this to discover project structure.

6. **bash(command, timeout?)** — Execute any shell command. Use this to run
   tests, builds, linters, git commands, or any terminal tool.

7. **diff(path?)** — Show git diff for files. Use this to review pending changes.

## Guidelines

- Prefer edit() over write() for small changes — it preserves context.
- Always read a file before editing it unless you just created it.
- Use grep() to find relevant code before making changes.
- Run tests after making changes to verify correctness.
- Show the user what you're doing — explain your reasoning.
- If a bash command fails, try to diagnose and fix the issue.
```

---

## Phase 9: .env.example (~10 lines)

```
# Required
MINICODE_API_BASE=https://api.openai.com/v1
MINICODE_API_KEY=sk-your-api-key-here

# Optional
MINICODE_MODEL=
MINICODE_MAX_TOKENS=4096
MINICODE_CONTEXT_RESERVE=8192
MINICODE_AUTO_COMPACT=true
```

---

## Phase 10: Edge Cases & Polish

### Error handling matrix

| Scenario | Behavior |
|---|---|
| No .env file | Show clear error, guide user to create one |
| Invalid API key | Stream fails → show error in chat + status bar turns red |
| Network timeout | httpx timeout → retry once → show error in chat |
| Model doesn't support tools | Fall back to text-only mode, warn user |
| Tool execution fails (file not found, etc.) | Return error as tool result, model handles it |
| Context compaction fails | Log error, keep history as-is, don't crash |
| Terminal resize | Textual handles natively |
| Ctrl+C during tool execution | Cancel current tool, return partial result |
| Very long tool output | Truncate at max chars with [...] marker |
| Empty user input | Ignore, don't send to API |
| Streaming disconnects mid-turn | Reconnect on next user message |

### Key edge case: Tool call within a tool call

Not supported in v1. If the model tries this, the agent ignores nested tool_use.

### Key edge case: Multiple parallel tool calls

Supported via `asyncio.gather`. Tools that modify state (write, edit, bash, diff) are marked as mutating. Non-mutating tools (read, grep, glob) execute in parallel. Mutating tools execute sequentially to avoid races. For v1, all tools execute in parallel (simpler, and git/res filesystem handles conflicts reasonably).

---

## Implementation Order Summary

```
Phase 1: Skeleton (~60 lines)         — 10 minutes
Phase 2: Config & Prompt Loader       — 15 minutes
Phase 3: LLM Client                   — 15 minutes
Phase 4: Tool System                  — 30 minutes
Phase 5: Context Manager              — 10 minutes
Phase 6: TUI Widgets                  — 45 minutes
Phase 7: Agent Loop                   — 20 minutes
Phase 8: minicode.md                  — 10 minutes
Phase 9: .env.example                  — 2 minutes
Phase 10: Edge Cases & Polish         — 20 minutes
────────────────────────────────────────────────
Total: ~3 hours coding
```

Each phase produces a runnable artifact. After Phase 3 you can test API connectivity. After Phase 4 you can test tools. After Phase 6 you have the full shell. Phase 7 wires it all together.
