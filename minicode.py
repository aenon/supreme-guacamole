#!/usr/bin/env python3
"""minicode — a minimal AI coding agent with conversational TUI."""

from __future__ import annotations

import asyncio, difflib, json, os, re, shlex, subprocess
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx, tiktoken
from dotenv import load_dotenv
from textual import on
from textual.app import App, ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

# ── Constants ────────────────────────────────────────────────────────────────

VERSION = "0.1.0"
APP_NAME = "minicode"
PRESSURE_WARN = 0.60
PRESSURE_CRITICAL = 0.80
PRESSURE_BLOCK = 0.95
MAX_TOOL_ROUNDS = 10
TOOL_OUTPUT_MAX = 4000
BASH_OUTPUT_MAX = 8000

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 4096
    context_reserve: int = 8192
    auto_compact: bool = True
    debug: bool = False

@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None

@dataclass
class SystemPrompt:
    id: str
    title: str
    content: str

# ── Config Loader ────────────────────────────────────────────────────────────

def load_config(cli_args: Namespace | None = None) -> Config:
    load_dotenv(Path.cwd() / ".env")
    config = Config(
        api_base=os.getenv("MINICODE_API_BASE", ""),
        api_key=os.getenv("MINICODE_API_KEY", ""),
        model=os.getenv("MINICODE_MODEL", ""),
        max_tokens=int(os.getenv("MINICODE_MAX_TOKENS", "4096")),
        context_reserve=int(os.getenv("MINICODE_CONTEXT_RESERVE", "8192")),
        auto_compact=os.getenv("MINICODE_AUTO_COMPACT", "true").lower() != "false",
    )
    if cli_args and cli_args.model:
        config.model = cli_args.model
    if cli_args and cli_args.debug:
        config.debug = True
    return config

# ── Prompt Loader ─────────────────────────────────────────────────────────────

def walk_up(start: Path) -> Iterator[Path]:
    for parent in [start, *start.parents]:
        yield parent
        if parent == parent.parent:
            break

def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Returns (metadata_dict, body_text) from markdown with YAML frontmatter."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is not None:
        meta = {k.strip(): v.strip()
                for line in lines[1:end_idx]
                if ":" in line
                for k, _, v in [line.partition(":")]}
        return meta, "\n".join(lines[end_idx + 1:]).strip()
    # Closing --- consumed by split — all remaining lines with ':' are frontmatter
    remaining = [l for l in lines[1:] if l.strip()]
    if remaining and all(":" in l for l in remaining):
        meta = {k.strip(): v.strip()
                for line in lines[1:] if ":" in line
                for k, _, v in [line.partition(":")]}
        return meta, ""
    return {}, text

def _split_documents(text: str) -> list[str]:
    """Split by --- dividers, distinguishing frontmatter from true dividers."""
    lines = text.splitlines()
    documents, current = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "---":
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            fwd_kv = j < len(lines) and ":" in lines[j]
            k = i - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            prev_kv = k >= 0 and ":" in lines[k]
            if prev_kv or fwd_kv:
                current.append(line); i += 1; continue
            if current and any(l.strip() for l in current):
                documents.append("\n".join(current))
            current = []; i += 1; continue
        current.append(line); i += 1
    if current and any(l.strip() for l in current):
        documents.append("\n".join(current))
    return documents

def parse_minicode_md(path: str | Path = "minicode.md") -> list[SystemPrompt]:
    _DF = [SystemPrompt("default", "Default Coding Agent", "You are minicode, a coding agent.")]
    p = Path(path)
    if not p.exists():
        return _DF
    docs = _split_documents(p.read_text())
    prompts: list[SystemPrompt] = []
    for doc in docs:
        doc = doc.strip()
        if not doc: continue
        meta, body = parse_frontmatter(doc)
        if not body: continue
        pid = meta.get("id", "default" if not prompts else f"p-{len(prompts)}")
        prompts.append(SystemPrompt(pid, meta.get("title", pid.title()), body))
    return prompts or _DF

def collect_prompts(cwd: Path, selected_id: str) -> list[dict[str, str]]:
    """Build system messages: built-in + user + workspace + AGENTS.md/CLAUDE.md."""
    messages: list[dict[str, str]] = []
    builtins = parse_minicode_md("minicode.md")
    sel = next((p for p in builtins if p.id == selected_id), builtins[0] if builtins else None)
    if sel: messages.append({"role": "system", "content": sel.content})
    for p in [Path.home() / ".minicode" / "instructions.md",
              cwd / ".minicode" / "instructions.md"]:
        if p.exists(): messages.append({"role": "system", "content": p.read_text().strip()})
    for root in walk_up(cwd):
        for name in ("AGENTS.md", "CLAUDE.md"):
            f = root / name
            if f.exists(): messages.append({"role": "system", "content": f.read_text().strip()})
    return messages

# ── LLM Client ────────────────────────────────────────────────────────────────

_TOK = tiktoken.get_encoding("cl100k_base")

MODEL_CTX: dict[str, int] = {
    "gpt-4o": 128_000, "gpt-4o-mini": 128_000, "gpt-4.5": 128_000,
    "gpt-4-turbo": 128_000, "gpt-4": 8_192, "gpt-3.5": 16_385,
    "claude-sonnet-4": 200_000, "claude-3.5": 200_000, "claude-3": 200_000,
    "claude-opus-4": 200_000, "gemini": 1_000_000, "deepseek": 64_000,
    "llama": 8_192, "qwen": 32_000, "mistral": 32_000, "o": 200_000,
}

def count_tokens(text: str) -> int:
    return len(_TOK.encode(text))

def estimate_message_tokens(msg: dict) -> int:
    n = 4
    c = msg.get("content", "")
    if isinstance(c, str): n += count_tokens(c)
    elif isinstance(c, list):
        for b in c:
            if b.get("type") == "text": n += count_tokens(str(b.get("text", "")))
    for tc in msg.get("tool_calls", []):
        n += count_tokens(json.dumps(tc.get("function", {}).get("arguments", ""))) + 8
    return n

def estimate_messages_tokens(msgs: list[dict]) -> int:
    return sum(estimate_message_tokens(m) for m in msgs)

def resolve_context_window(model: str) -> int:
    for pfx, sz in MODEL_CTX.items():
        if model.lower().startswith(pfx): return sz
    return 128_000

def _hdr(config: Config) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"}

async def scan_models(config: Config) -> list[str]:
    if not config.api_base or not config.api_key: return []
    try:
        async with httpx.AsyncClient(timeout=10) as cl:
            r = await cl.get(f"{config.api_base.rstrip('/')}/models", headers=_hdr(config))
            r.raise_for_status()
            models = r.json().get("data", [])
        kw = ("gpt","claude","gemini","llama","qwen","mistral","deepseek",
              "o1","o3","o4","codestral","dolphin","command","yi-","phi")
        return sorted(m["id"] for m in models
                      if any(k in m.get("id","").lower() for k in kw))
    except Exception: return []

async def stream_completion(
    config: Config, messages: list[dict], tools: list[dict] | None = None
) -> AsyncIterator[dict]:
    """Yields {"type":"text","content":str} | {"type":"tool_use","id","name","arguments"} | {"type":"done","finish_reason"}"""
    body: dict[str, Any] = {"model": config.model, "messages": messages,
                             "stream": True, "max_tokens": config.max_tokens}
    if tools: body["tools"] = tools
    url = f"{config.api_base.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=120) as cl:
        async with cl.stream("POST", url, headers=_hdr(config), json=body) as resp:
            resp.raise_for_status()
            tc_acc: dict[int, dict] = {}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "): continue
                p = line[6:].strip()
                if p == "[DONE]": break
                try: chunk = json.loads(p)
                except json.JSONDecodeError: continue
                choices = chunk.get("choices", [])
                if not choices: continue
                c = choices[0]; d = c.get("delta", {}); f = c.get("finish_reason")
                if d.get("content"): yield {"type": "text", "content": d["content"]}
                for tc in d.get("tool_calls", []):
                    e = tc_acc.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"): e["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"): e["name"] += fn["name"]
                    if fn.get("arguments"): e["arguments"] += fn["arguments"]
                if f == "tool_calls" and tc_acc:
                    for e in tc_acc.values():
                        if e["name"] and e["arguments"]:
                            yield {"type": "tool_use", "id": e["id"],
                                   "name": e["name"], "arguments": e["arguments"]}
                    tc_acc.clear()
                if f and f != "tool_calls": yield {"type": "done", "finish_reason": f}

async def complete(config: Config, messages: list[dict],
                   tools: list[dict] | None = None, max_tokens: int = 512) -> dict:
    body: dict[str, Any] = {"model": config.model, "messages": messages,
                             "stream": False, "max_tokens": max_tokens}
    if tools: body["tools"] = tools
    url = f"{config.api_base.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=60) as cl:
        r = await cl.post(url, headers=_hdr(config), json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

# ── Tools ─────────────────────────────────────────────────────────────────────

class Tool(ABC):
    name: str = ""
    desc: str = ""
    params: dict[str, dict] = {}  # JSON Schema properties

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult: ...

    def definition(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.desc,
            "parameters": {"type": "object", "properties": self.params,
                           "required": list(self.params)}}}

    def _trunc(self, text: str, max_chars: int | None = None) -> str:
        limit = max_chars or (BASH_OUTPUT_MAX if self.name == "bash" else TOOL_OUTPUT_MAX)
        if len(text) > limit: return text[:limit] + f"\n... [truncated at {limit} chars]"
        return text

class ReadTool(Tool):
    name = "read"
    desc = "Read a file with line numbers. Returns directory listing if path is a dir."
    params = {"path": {"type": "string"}, "offset": {"type": "integer"},
              "limit": {"type": "integer"}}

    async def execute(self, path: str, offset: int = 1, limit: int = 500) -> ToolResult:
        p = Path(path)
        try:
            if not p.exists(): return ToolResult(False, "", f"not found: {path}")
            if p.is_dir():
                items = sorted(p.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:50]
                return ToolResult(True, "\n".join(str(x) for x in items))
            lines = p.read_text().splitlines()
            start = max(0, offset - 1)
            end = min(len(lines), start + limit)
            out = "\n".join(f"{i+1:4}|{l}" for i, l in enumerate(lines[start:end], start))
            return ToolResult(True, self._trunc(out))
        except Exception as e: return ToolResult(False, "", str(e))

class WriteTool(Tool):
    name = "write"
    desc = "Create or overwrite a file. Creates parent directories."
    params = {"path": {"type": "string"}, "content": {"type": "string"}}

    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return ToolResult(True, f"Wrote {p.stat().st_size} bytes to {path}")
        except Exception as e: return ToolResult(False, "", str(e))

class EditTool(Tool):
    name = "edit"
    desc = "Find-and-replace edit. old_string must be unique in the file."
    params = {"path": {"type": "string"}, "old_string": {"type": "string"},
              "new_string": {"type": "string"}}

    async def execute(self, path: str, old_string: str, new_string: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists(): return ToolResult(False, "", f"not found: {path}")
            text = p.read_text()
            if old_string not in text:
                # Fuzzy match fallback
                matcher = difflib.SequenceMatcher(None, old_string, text)
                blocks = matcher.get_matching_blocks()
                best = max(blocks[:-1], key=lambda b: b.size)
                if best.size < len(old_string) * 0.5:
                    return ToolResult(False, "", "old_string not found; write() the file instead")
                old = text[best.b:best.b + best.size]
                return ToolResult(False, "", f"fuzzy match found but differs:\n  expected: {old_string[:80]!r}\n  found:    {old[:80]!r}")
            new_text = text.replace(old_string, new_string, 1)
            p.write_text(new_text)
            diff = "\n".join(difflib.unified_diff(text.splitlines(), new_text.splitlines(),
                                                  fromfile=path, tofile=path, lineterm=""))
            return ToolResult(True, self._trunc(diff) if diff else "No changes")
        except Exception as e: return ToolResult(False, "", str(e))

class GrepTool(Tool):
    name = "grep"
    desc = "Search file contents with regex (uses ripgrep)."
    params = {"pattern": {"type": "string"}, "path": {"type": "string"},
              "file_glob": {"type": "string"}}

    async def execute(self, pattern: str, path: str = ".", file_glob: str = "") -> ToolResult:
        try:
            cmd = ["rg", "--no-heading", "-n", "--color", "never"]
            if file_glob: cmd += ["-g", file_glob]
            cmd += [pattern, path]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 1: return ToolResult(True, "(no matches)")
            if r.returncode > 1:
                # rg not found, fallback to grep
                cmd2 = ["grep", "-rn", "--color=never"]
                if file_glob: cmd2 += ["--include", file_glob]
                cmd2 += [pattern, path]
                r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
                return ToolResult(True, self._trunc(r2.stdout or "(no matches)"))
            return ToolResult(True, self._trunc(r.stdout))
        except FileNotFoundError:
            r = subprocess.run(["grep", "-rn", "--color=never", pattern, path],
                               capture_output=True, text=True, timeout=30)
            return ToolResult(True, self._trunc(r.stdout or "(no matches)"))
        except Exception as e: return ToolResult(False, "", str(e))

class GlobTool(Tool):
    name = "glob"
    desc = "Find files by name pattern (e.g., '*.py', '**/*.ts')."
    params = {"pattern": {"type": "string"}, "path": {"type": "string"}}

    async def execute(self, pattern: str, path: str = ".") -> ToolResult:
        try:
            results = sorted(Path(path).rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
            return ToolResult(True, "\n".join(str(r) for r in results) or "(no matches)")
        except Exception as e: return ToolResult(False, "", str(e))

class BashTool(Tool):
    name = "bash"
    desc = "Execute a shell command. Default timeout 60s."
    params = {"command": {"type": "string"}, "timeout": {"type": "integer"}}

    async def execute(self, command: str, timeout: int = 60) -> ToolResult:
        try:
            r = await asyncio.to_thread(
                subprocess.run, command, shell=True, capture_output=True,
                text=True, timeout=timeout
            )
            out = r.stdout
            if r.stderr: out += "\n" + r.stderr
            success = r.returncode == 0
            return ToolResult(success, self._trunc(out.strip() or "(no output)", BASH_OUTPUT_MAX),
                              None if success else f"exit code: {r.returncode}")
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"timed out after {timeout}s")
        except Exception as e: return ToolResult(False, "", str(e))

class DiffTool(Tool):
    name = "diff"
    desc = "Show git diff. Optional path filter."
    params = {"path": {"type": "string"}}

    async def execute(self, path: str = "") -> ToolResult:
        try:
            cmd = ["git", "diff", "--no-color"]
            if path: cmd += ["--", path]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return ToolResult(True, self._trunc(r.stdout or "(no changes)"))
        except FileNotFoundError: return ToolResult(False, "", "git not found")
        except Exception as e: return ToolResult(False, "", str(e))

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        for t in [ReadTool(), WriteTool(), EditTool(), GrepTool(), GlobTool(), BashTool(), DiffTool()]:
            self._tools[t.name] = t

    def definitions(self) -> list[dict]:
        return [t.definition() for t in self._tools.values()]

    async def execute(self, name: str, args: dict) -> ToolResult:
        t = self._tools.get(name)
        if not t: return ToolResult(False, "", f"unknown tool: {name}")
        try: return await t.execute(**args)
        except Exception as e: return ToolResult(False, "", str(e))

# ── Context Manager ──────────────────────────────────────────────────────────

_SUMMARY_PROMPT = (
    "Summarize these conversation exchanges in 2-3 sentences. "
    "Focus on: user requests, files changed, decisions made, errors encountered.\n\n"
    "Exchanges:\n"
)

class ContextManager:
    def __init__(self, context_window: int, reserve: int = 8192):
        self.window = context_window
        self.reserve = reserve

    @property
    def usable(self) -> int:
        return self.window - self.reserve

    def pressure(self, system_tokens: int, messages: list[dict]) -> float:
        n = system_tokens + sum(estimate_message_tokens(m) for m in messages
                                if m.get("role") != "system")
        return n / self.usable

    def needs_compact(self, system_tokens: int, messages: list[dict]) -> bool:
        return self.pressure(system_tokens, messages) >= PRESSURE_CRITICAL

    def blocked(self, system_tokens: int, messages: list[dict]) -> bool:
        return self.pressure(system_tokens, messages) >= PRESSURE_BLOCK

    async def compact(self, messages: list[dict], config: Config) -> list[dict]:
        """Summarize old history, keep system prompts + last 3 exchanges intact."""
        sys_msgs = [m for m in messages if m["role"] == "system"]
        hist = [m for m in messages if m["role"] != "system"]
        if len(hist) <= 12: return messages  # < 3 exchanges, nothing to compact

        protected = hist[-12:]  # last 3 exchanges
        old = hist[:-12]

        summary_text = ""
        try:
            resp = await complete(config,
                [{"role": "system", "content": _SUMMARY_PROMPT + json.dumps(old)}],
                max_tokens=512)
            summary_text = resp.get("content", "(summary unavailable)")
        except Exception:
            summary_text = f"(compaction failed, {len(old)} messages dropped)"

        compacted = [{"role": "system",
                      "content": f"[Compacted context: {summary_text}]"}]
        return sys_msgs + compacted + protected


# ── TUI Messages ──────────────────────────────────────────────────────────────

class MessageSubmitted(Message):
    """User submitted a chat message (non-command)."""
    def __init__(self, content: str) -> None:
        self.content = content
        super().__init__()

class CommandSubmitted(Message):
    """User submitted a /slash command."""
    def __init__(self, command: str, args: str) -> None:
        self.command = command
        self.args = args
        super().__init__()

HELP_TEXT = """\
[b]/model &lt;name&gt;[/]  Switch model
[b]/prompt &lt;id&gt;[/]   Switch system prompt
[b]/compact[/]       Manual context compaction
[b]/clear[/]         Clear conversation
[b]/cd &lt;path&gt;[/]     Change working directory
[b]/env[/]           Show config
[b]/help[/]          Show this help
[b]/quit[/]          Exit"""

# ── TUI App ──────────────────────────────────────────────────────────────────

class MinicodeApp(App):
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #messages { height: 1fr; border: none; }
    #status { dock: bottom; height: 1; padding: 0 1; background: $panel; }
    #input { dock: bottom; height: auto; margin: 0 1; }
    """

    config: Config = field(default_factory=Config)
    model_name: reactive[str] = reactive("\u2014")
    token_pct: reactive[float] = reactive(0.0)
    tool_count: reactive[int] = reactive(0)
    compacting: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, name=f"{APP_NAME} v{VERSION}")
        yield RichLog(id="messages", highlight=True, markup=True, wrap=True)
        yield Static("", id="status")
        yield Input(placeholder="Ask minicode to do something...  /help for commands",
                    id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = APP_NAME
        self.system_msgs = collect_prompts(Path.cwd(), self.cli_prompt_id)
        self.history: list[dict] = []
        self.tools = ToolRegistry()
        self.ctx = ContextManager(resolve_context_window(self.config.model),
                                  self.config.context_reserve)
        if self.config.debug:
            self.notify(f"Loaded {len(self.system_msgs)} system prompt(s), "
                        f"model={self.config.model or '(not set)'}")

    # ── Status bar ──────────────────────────────────────────────────────

    def _update_footer(self) -> None:
        """Render status line into the #status Static widget."""
        pct = min(self.token_pct, 0.999)
        color = "green" if pct < 0.60 else "yellow" if pct < 0.80 else "red"
        parts = [
            f"[bold]{self.model_name}[/]",
            f"ctx: [{color}]{pct:.0%}[/]",
            f"tools: {self.tool_count}",
            str(Path.cwd()),
        ]
        if self.compacting:
            parts.insert(1, "[bold yellow]⟳[/]")
        self.query_one("#status", Static).update(" │ ".join(parts))

    def watch_model_name(self) -> None:
        self._update_footer()

    def watch_token_pct(self) -> None:
        self._update_footer()

    def watch_tool_count(self) -> None:
        self._update_footer()

    def watch_compacting(self) -> None:
        self._update_footer()

    # ── Slash commands ───────────────────────────────────────────────────

    def _handle_command(self, cmd: str, args: str) -> None:
        """Dispatch a slash command."""
        chat = self.query_one("#messages", RichLog)
        if cmd == "help":
            chat.write(HELP_TEXT)
        elif cmd == "quit":
            self.exit()
        elif cmd == "clear":
            chat.clear()
            self.history.clear()
            self._refresh_status()
        elif cmd == "model":
            if args:
                self.model_name = args
                self.config.model = args
                chat.write(f"[dim]Model switched to {args}[/]")
            else:
                chat.write(f"[dim]Current model: {self.model_name}[/]")
        elif cmd == "prompt":
            if args:
                self.cli_prompt_id = args
                self.system_msgs = collect_prompts(Path.cwd(), args)
                chat.write(f"[dim]Prompt switched to {args} ({len(self.system_msgs)} message(s))[/]")
            else:
                chat.write(f"[dim]Current prompt: {self.cli_prompt_id}[/]")
        elif cmd == "cd":
            if args:
                try:
                    p = Path(args).expanduser().resolve()
                except ValueError:
                    chat.write(f"[dim]Invalid path: {args}[/]"); return
                if p.is_dir():
                    os.chdir(p)
                    chat.write(f"[dim]cwd → {p}[/]")
                    self._update_footer()
                else:
                    chat.write(f"[dim]not a directory: {args}[/]")
            else:
                chat.write(f"[dim]cwd: {Path.cwd()}[/]")
        elif cmd == "env":
            c = self.config
            chat.write(f"[dim]model={c.model or '(not set)'}  "
                       f"api={c.api_base or '(not set)'}  "
                       f"max_tokens={c.max_tokens}  reserve={c.context_reserve}[/]")
        elif cmd == "compact":
            if self.history:
                self.compacting = True
                chat.write("[dim yellow]⟳ Compacting…[/]")
                asyncio.create_task(self._manual_compact())
            else:
                chat.write("[dim]Nothing to compact[/]")
        else:
            chat.write(f"[dim]Unknown command: /{cmd}.  /help for list[/]")

    # ── Input handling ───────────────────────────────────────────────────

    @on(Input.Submitted)
    def on_input(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        event.input.clear()
        if value.startswith("/"):
            parts = value[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            self._handle_command(cmd, args)
        else:
            self.post_message(MessageSubmitted(value))

    def on_message_submitted(self, event: MessageSubmitted) -> None:
        chat = self.query_one("#messages", RichLog)
        chat.write(f"\n[bold cyan]You:[/] {event.content}")
        self.query_one("#input", Input).disabled = True
        asyncio.create_task(self._run_agent_loop(event.content))

    # ── Agent Loop ────────────────────────────────────────────────────────

    async def _run_agent_loop(self, user_content: str) -> None:
        """Run one user message through the full agent loop."""
        inp = self.query_one("#input", Input)
        chat = self.query_one("#messages", RichLog)
        try:
            await self._agent_turn(chat, user_content)
        except Exception as e:
            chat.write(f"\n[red]Unexpected error: {e}[/]")
        finally:
            inp.disabled = False
            inp.focus()

    async def _agent_turn(self, chat: RichLog, user_content: str) -> None:
        if not self.config.api_key:
            chat.write("\n[red]No API key. Set MINICODE_API_KEY in .env[/]"); return
        if not self.config.model:
            chat.write("\n[red]No model. Set MINICODE_MODEL in .env or /model[/]"); return

        self.history.append({"role": "user", "content": user_content})

        # Context pressure check → auto-compact
        sys_tokens = estimate_messages_tokens(self.system_msgs)
        if self.config.auto_compact and self.ctx.needs_compact(sys_tokens, self.history):
            self.compacting = True
            chat.write("\n[dim yellow]⟳ Compacting context…[/]")
            try:
                compacted = await self.ctx.compact(self.system_msgs + self.history, self.config)
                self.history = [m for m in compacted if m["role"] != "system"]
            except Exception:
                pass
            self.compacting = False

        # Tool loop
        tools_def = self.tools.definitions()
        max_rounds = MAX_TOOL_ROUNDS
        for _ in range(max_rounds):
            messages = self.system_msgs + self.history
            assistant_text = ""
            tool_calls: list[dict] = []

            chat.write("\n[bold]Assistant:[/] ")
            try:
                async for event in stream_completion(self.config, messages, tools_def):
                    if event["type"] == "text":
                        assistant_text += event["content"]
                        chat.write(event["content"])
                    elif event["type"] == "tool_use":
                        tool_calls.append(event)
                    elif event["type"] == "done":
                        break
            except Exception as e:
                chat.write(f"\n[red]API error: {e}[/]")
                self.history.append({"role": "assistant", "content": assistant_text or f"(error: {e})"})
                break

            if not tool_calls:
                chat.write("\n")
                self.history.append({"role": "assistant", "content": assistant_text})
                break

            # Render tool calls
            chat.write("\n")
            tc_msgs: list[dict] = []
            for tc in tool_calls:
                tc_msgs.append({
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                })
                chat.write(f"  [dim]● {tc['name']}({tc['arguments'][:200]})[/]\n")

            self.history.append({
                "role": "assistant", "content": assistant_text or None,
                "tool_calls": tc_msgs
            })

            # Execute tools + feed results
            for tc in tool_calls:
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {}
                result = await self.tools.execute(tc["name"], args)
                status = "[green]✓[/]" if result.success else "[red]✗[/]"
                output_lines = result.output.split("\n")[:10]
                preview = "\n    ".join(output_lines)
                if len(result.output.split("\n")) > 10:
                    preview += "\n    …"
                chat.write(f"    {status} {preview}\n")
                if result.error:
                    chat.write(f"    [red]{result.error}[/]\n")
                self.history.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": result.output
                })
        else:
            # Exceeded max rounds
            chat.write("\n[dim yellow](max tool rounds reached)[/]")
            self.history.append({"role": "assistant", "content": "(max tool rounds)"})

        self._refresh_status()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        """Update reactive status bar fields from current state."""
        self.model_name = self.config.model or "\u2014"
        self.tool_count = len(self.tools._tools)
        sys_tokens = estimate_messages_tokens(self.system_msgs)
        self.token_pct = self.ctx.pressure(sys_tokens, self.history) if self.history else 0.0

    async def _manual_compact(self) -> None:
        """Manual compaction triggered by /compact."""
        chat = self.query_one("#messages", RichLog)
        try:
            compacted = await self.ctx.compact(self.system_msgs + self.history, self.config)
            self.history = [m for m in compacted if m["role"] != "system"]
            chat.write(" [dim]done[/]\n")
        except Exception as e:
            chat.write(f" [red]failed: {e}[/]\n")
        self.compacting = False
        self._refresh_status()

# ── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=f"{APP_NAME} — minimal AI coding agent")
    p.add_argument("--model", help="Model override")
    p.add_argument("--prompt", default="default", help="System prompt ID")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app = MinicodeApp()
    app.config = load_config(args)
    app.cli_prompt_id = args.prompt
    app.run()

if __name__ == "__main__":
    main()
