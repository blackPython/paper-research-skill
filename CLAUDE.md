# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a **literature review agent skill** for Claude Code, distributed as a standalone skill package. Users install it by copying or symlinking the skill folder into their `~/.claude/skills/` directory. The skill searches, retrieves, and synthesizes academic papers using two CLI tools.

## CLI Tools

### arxiv (arxivterminal)

Install: `pip install arxivterminal`

Papers are fetched into a local SQLite database, then searched locally.

```bash
arxiv fetch --num-days 7 --categories cs.AI,cs.CL   # populate DB from arXiv API
arxiv search "transformer efficiency" --limit 10      # pattern match against DB
arxiv search -e "transformer efficiency"              # LSA semantic search
arxiv search -e -f "transformer efficiency"           # force retrain LSA model
arxiv show --days-ago 3                               # interactive browse
arxiv stats                                           # DB summary
```

Key arXiv categories: `cs.AI`, `cs.CL`, `cs.CV`, `cs.LG`, `cs.NE`, `stat.ML`, `cs.IR`, `cs.RO`

**Important**: arxiv requires `fetch` before `search` — it only searches the local DB.

### hf papers (huggingface_hub CLI)

Install: `pip install -U "huggingface_hub[cli]"` (requires >= 1.12.0 for papers support)

Searches the Hugging Face daily papers index (curated, trending ML papers). No local DB — queries HF API directly.

```bash
hf papers search "vision language" --limit 10         # keyword search
hf papers search "diffusion" --format json            # structured JSON output
hf papers list --sort trending --limit 10             # trending papers
hf papers list --date 2026-04-28                      # papers from specific date
hf papers list --week 2026-W18                        # papers from ISO week
hf papers info 2601.15621                             # metadata for a paper (by arXiv ID)
hf papers read 2601.15621                             # full paper as markdown
```

Use `--format json` on `search` and `list` for machine-parseable output.

## Skill Structure

This repo is the distributable skill. The expected layout:

```
research_agent/
  SKILL.md              # Frontmatter (name, description, triggers) + skill content
  references/           # Deeper reference files linked from SKILL.md
  CLAUDE.md             # This file (dev guidance, not part of the skill itself)
```

To install: symlink or copy this directory into `~/.claude/skills/literature-review/`.

### SKILL.md Frontmatter

```yaml
---
name: literature-review
description: "Trigger description — Claude Code uses this to decide when to auto-invoke"
---
```

See `~/.claude/skills/strands/SKILL.md` or `~/.claude/skills/vllm/SKILL.md` for reference patterns.

## Architecture Decisions

- **Two-source strategy**: arxiv CLI covers broad arXiv categories with local DB + semantic search; HF papers covers curated/trending ML papers with direct API access.
- **HF papers `read` returns full markdown**: use this for deep paper analysis. arxiv only provides abstracts.
- **JSON output for programmatic use**: prefer `--format json` with HF papers commands when parsing results.
