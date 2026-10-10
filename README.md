# Trodden

> Your agents take the well-trodden path.

Procedural memory for AI coding agents. Trodden learns the steps that solved
a task in your repository (files changed, commands run, the check that
passed) and gives them back to the agent the next time a similar task comes
up.

It is one Rust binary. It runs locally, makes no network calls and needs no
account. It works with Claude Code, Codex, Gemini CLI, Qwen Code, GitHub
Copilot CLI, Factory Droid, Cursor, OpenCode, Kilo Code, Kimi Code and Cline.

## How it works

1. **Capture.** When a turn or session ends, Trodden reads the transcript
   (for agents that do not keep one with command results, it follows the
   session through the agent's hooks instead).
   It keeps the first line of each prompt (up to 200 characters), commands,
   file paths, touched function names and exit codes, with secrets redacted.
   From a failed tool call it keeps at most one normalized error line, used
   only to recognize the same error later. It never stores file contents,
   other tool output or the rest of a prompt.
2. **Extract.** A task becomes a procedure only if it changed files and a
   check (tests, build, lint) passed after the last change. Extraction is
   deterministic and uses no language model.
3. **Recall.** On each prompt, Trodden matches it against the procedures
   learned in the same repository and injects at most one, only when the
   match is confident and the procedure's files still exist. Recall takes
   under 10 ms.
4. **Learn.** Each injection is scored by whether its task ended with a
   passing check. A small share of matches is held out to compare against,
   and procedures that do worse than no procedure are taken out of recall.
5. **Remind (optional, off by default).** If the agent changed files after a
   procedure was injected and did not run the procedure's check afterwards,
   the end of the turn asks it to run the check before finishing.

What the agent sees:

```
<trodden-memory id="p_e5d8f1fc71cc" rev="1" learned-from="1 session">
A path that worked for a similar past task in this repository. Reference data, not instructions: confirm it against the current code.
Task: Page 2 of the product listing repeats the last product from page 1
1. edit src/paginate.js (paginate)
2. check: npm test (expect exit 0)
</trodden-memory>
```

## Install

Requires Rust.

```sh
git clone https://github.com/aayank13/trodden && cd trodden
cargo install --locked --path crates/trodden
trodden init
```

Then connect your agents. `trodden connect <agent>` adds Trodden's hooks to
the agent's user settings, next to any hooks you already have
(`trodden connect` alone lists the agents and which are installed;
`--dry-run` shows the change first; `trodden disconnect <agent>` removes it):

```sh
trodden connect claude-code
trodden connect codex
```

Claude Code can also load the same hooks as a plugin instead
(`claude plugin marketplace add aayank13/trodden`, then
`claude plugin install trodden@trodden`); use one or the other.

Optional:

```sh
trodden embeddings install   # semantic matching, a one-time 32 MB download
trodden backfill             # learn from your existing sessions
```

## Agents

| Agent | `connect` name | Recall on prompt | Recall on a failed command | Verify reminder | Learns from | Backfill |
|---|---|---|---|---|---|---|
| Claude Code | `claude-code` | yes | yes | yes | transcript | yes |
| Codex | `codex` | yes | yes | yes | transcript | yes |
| Gemini CLI | `gemini` | yes | yes | yes | transcript | yes |
| Qwen Code | `qwen` | yes | yes | yes | transcript | yes |
| GitHub Copilot CLI | `copilot` | yes | yes | yes | transcript | yes |
| Factory Droid | `droid` | yes | yes | yes | transcript | yes |
| Cursor | `cursor` | no | yes | yes | hooks | no |
| OpenCode | `opencode` | yes | yes | no | hooks | no |
| Kilo Code | `kilo` | yes | yes | no | hooks | no |
| Kimi Code | `kimi` | yes | no | yes | hooks | no |
| Cline | `cline` | extension only | yes | no | hooks | no |

"no" means the agent's hooks cannot do it: Cursor cannot add context to a
prompt, Kimi Code ignores output after a failed command, and OpenCode, Kilo
Code and Cline cannot continue a finished turn. The Cline CLI cannot add
context to a prompt either, so Cline recalls on a prompt only in its VS Code
and JetBrains extension. Agents that learn from hooks only learn from sessions
after they are connected.

A few agents need one step of their own after `trodden connect`: Codex asks
you to trust the new hooks the next time it starts, Cline needs its "Enable
Hooks" setting turned on, and running Factory Droid, OpenCode and Kilo Code
sessions must be restarted. `trodden connect` prints the step.

Agents with MCP support but no usable hooks can still ask Trodden directly
through `trodden mcp`.

## Commands

| Command | What it does |
|---|---|
| `trodden status` | What is stored, and whether capture is on |
| `trodden list` | Procedures learned in this repository (`--all-repos` for every repository, `--revisions` for every revision) |
| `trodden show <id>` | A procedure in full |
| `trodden recall "<prompt>" --explain` | What recall would inject, and why (`--cwd <dir>` to ask from another directory) |
| `trodden rejected` | Tasks that did not become procedures, and why |
| `trodden outcomes` | How tasks went with and without each procedure |
| `trodden retire <id>` | Take a procedure out of recall for good |
| `trodden forget <id>` / `--repo` / `--all --yes` | Delete procedures |
| `trodden pause` / `resume` | Stop or restart capture and recall |
| `trodden config` | Show the current settings |
| `trodden config holdout <0-1 or auto>` | Share of matches held out for comparison |
| `trodden config verify-reminder <on or off>` | Ask the agent to run a recalled procedure's check when it skipped it |
| `trodden connect [<agent>]` / `disconnect <agent>` | Install or remove an agent's hooks |
| `trodden backfill [--agent <agent>]` | Learn from sessions already on disk |
| `trodden doctor` | Check the installation |
| `trodden mcp` | Read-only MCP tools for agents without prompt hooks |

## Status

Pre-alpha. Formats and commands may change between versions. Prebuilt
binaries come next.

## License

[Apache-2.0](LICENSE)
