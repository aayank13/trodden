# Trodden

> Your agents take the well-trodden path.

Procedural memory for AI coding agents. Trodden learns the steps that solved
a task in your repository (files changed, commands run, the check that
passed) and gives them back to the agent the next time a similar task comes
up.

It is one Rust binary. It runs locally, makes no network calls and needs no
account. It works with Claude Code today.

## How it works

1. **Capture.** When a turn or session ends, Trodden reads the transcript.
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

Requires Rust and Claude Code.

```sh
git clone https://github.com/aayank13/trodden && cd trodden
cargo install --locked --path crates/trodden
trodden init
claude plugin marketplace add aayank13/trodden
claude plugin install trodden@trodden
```

Optional:

```sh
trodden embeddings install   # semantic matching, a one-time 32 MB download
trodden backfill             # learn from your existing Claude Code sessions
```

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
| `trodden doctor` | Check the installation |
| `trodden mcp` | Read-only MCP tools for agents without prompt hooks |

## Status

Pre-alpha. Formats and commands may change between versions. Support for
Codex and other agents, and prebuilt binaries, come next.

## License

[Apache-2.0](LICENSE)
