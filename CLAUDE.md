# Isaac Sim — Claude Code Guide

The agent guide lives in [`AGENTS.md`](AGENTS.md). It is the single source of
truth for every agent platform (Cursor, Codex CLI, Claude Code, ...). `skills/`
is the canonical, agent-agnostic location for SKILL.md-format workflow guides.

At session start, read [`AGENTS.md`](AGENTS.md) — request loop, skill layout,
library pointers.

Read [`skills/SKILLS.md`](../../skills/SKILLS.md). Route to skills **on demand only** — read a skill's file when the user's request matches its domain (e.g., `skills/profile-isaac-sim/SKILL.md` for performance work). Do not bulk-load all skills upfront.

Load `.cursor/rules/*.mdc` files on demand when editing code that falls under
their domain (C++ / Python codestyle, docs style, extension structure, build
instructions, pip packaging, ...).

Skills are authored once under `skills/`. `.claude/skills/` contains symlinks
into `skills/` so Claude Code's native skill discovery works without
duplication. `.cursor/skills/` is retired.
