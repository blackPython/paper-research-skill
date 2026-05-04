# CLI Reference

## unified-search (tools/unified_search.py) — Primary Interface

Consolidated search across all sources. Single SQLite index at `.cache/papers.db` (60K+ papers).

### Commands

**`update`** — Refresh source caches, rebuild index, enrich new papers

```bash
python tools/unified_search.py update                    # refresh all sources
python tools/unified_search.py update --source hf        # refresh only HF
python tools/unified_search.py update --source arxiv     # refresh only arXiv
python tools/unified_search.py update --source openreview  # refresh OpenReview
python tools/unified_search.py update --source acl       # refresh ACL Anthology
```

| Flag | Description |
|------|-------------|
| `--source` | Refresh only this source: `hf`, `arxiv`, `openreview`, `acl` (default: all) |

Tracks last-updated per source. arXiv fetches only days since last update. HF fetches only new days. OpenReview re-fetches 10 venue-years. ACL re-parses the anthology repo.

**`search`** — Search the unified index

```bash
python tools/unified_search.py search "transformer efficiency" --limit 50
python tools/unified_search.py search -e "multimodal reasoning" --limit 50
python tools/unified_search.py search -e "query" --source openreview --limit 50
python tools/unified_search.py search -e "query" --source acl --limit 50
python tools/unified_search.py search -e "query" --source hf --limit 50
python tools/unified_search.py search -e "query" --source arxiv --limit 50
python tools/unified_search.py search -e "query" --since 2025-04-01 --limit 50
python tools/unified_search.py search -e "query" --limit 50 --format json
```

| Flag | Description |
|------|-------------|
| `-e, --semantic` | LSA semantic search (default is keyword match) |
| `-l, --limit N` | Max results (default: 20) |
| `--source` | Filter by source: `hf`, `arxiv`, `openreview`, `acl` |
| `--since YYYY-MM-DD` | Only papers published after this date |
| `--format text\|json` | Output format (default: text) |

Keyword search ranks by citations + upvotes. Semantic search ranks by LSA cosine similarity.

**`enrich`** — Add citation counts from Semantic Scholar

```bash
python tools/unified_search.py enrich                        # enrich all papers missing citations
python tools/unified_search.py enrich --paper 2401.06209     # enrich specific paper(s)
python tools/unified_search.py enrich --paper 2401.06209,2310.06825
python tools/unified_search.py enrich --source hf            # enrich only HF-sourced papers
```

| Flag | Description |
|------|-------------|
| `--paper` | Specific arXiv ID(s) to enrich, comma-separated |
| `--source` | Only enrich papers from this source |

Uses `S2_API_KEY` env var if set (100 req/s). Falls back to unauthenticated (rate limited). Exponential backoff via tenacity.

**`stats`** — Show index statistics

```bash
python tools/unified_search.py stats                    # full overview
python tools/unified_search.py stats --source hf        # stats for one source
python tools/unified_search.py stats --source arxiv
```

| Flag | Description |
|------|-------------|
| `--source` | Show stats for only this source |

---

## Live Search Tools (not from cache)

These query APIs directly — use alongside cached search for freshness.

### hf papers (huggingface_hub CLI)

```bash
hf papers search "vision language" --limit 20 --format json    # keyword search (live)
hf papers list --sort trending --date today --format json       # trending today
hf papers list --sort trending --week 2026-W18 --format json    # trending this week
hf papers info 2601.15621                                       # paper metadata
hf papers read 2601.15621                                       # full paper as markdown
```

### arxiv search-api (tools/arxiv_search.py)

```bash
python tools/arxiv_search.py search-api "memory tool LLM agent" --limit 20 --format json
python tools/arxiv_search.py search-api "query" --categories cs.AI,cs.CL,cs.LG --limit 20
```

Queries arXiv API directly. No local DB needed. Searches across all of arXiv history.

---

## Source Fetchers (called internally by `unified_search.py update`)

### hf-papers-cache (tools/hf_papers_cache.py)

Local SQLite cache for HF daily papers at `~/.local/share/hf-papers/papers.db`.

```bash
python tools/hf_papers_cache.py update                    # fetch new days since last
python tools/hf_papers_cache.py fetch --start 2025-01-01  # full historical fetch
python tools/hf_papers_cache.py stats
```

### arxiv-search (tools/arxiv_search.py)

Local SQLite cache at `~/.local/share/arxiv-search/papers.db`.

```bash
python tools/arxiv_search.py fetch --num-days 7 --categories cs.AI,cs.CL,cs.CV,cs.LG,cs.NE,stat.ML,cs.IR
python tools/arxiv_search.py search -e "query" --limit 20    # LSA search on local DB
python tools/arxiv_search.py stats
```

### openreview-search (tools/openreview_search.py)

JSON cache at `~/.openreview/`. Requires `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` env vars.

```bash
python tools/openreview_search.py fetch --conference ICLR --year 2026
python tools/openreview_search.py search -e "query" --limit 20
python tools/openreview_search.py stats
```

Supported conferences: ICLR, ICML, NeurIPS, EMNLP, NAACL, ACL.

### acl-anthology-search (tools/acl_anthology_search.py)

Local SQLite cache at `~/.local/share/acl-anthology/papers.db`.

```bash
python tools/acl_anthology_search.py fetch                                    # all default collections
python tools/acl_anthology_search.py fetch --collections 2025.acl,2025.emnlp  # specific
python tools/acl_anthology_search.py search -e "query" --venue emnlp --limit 20
python tools/acl_anthology_search.py stats
```

Default collections: `2024.acl`, `2025.acl`, `2024.emnlp`, `2025.emnlp`, `2024.naacl`, `2025.naacl`, `2024.eacl`, `2024.findings`, `2025.findings`

### citation-counts (tools/citation_counts.py)

Semantic Scholar batch citation lookup. Uses `S2_API_KEY` env var if set.

```bash
python tools/citation_counts.py lookup 2401.06209 2310.06825
python tools/citation_counts.py lookup --format json 2401.06209
echo "2401.06209" | python tools/citation_counts.py lookup --stdin
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENREVIEW_USERNAME` | OpenReview API authentication |
| `OPENREVIEW_PASSWORD` | OpenReview API authentication |
| `S2_API_KEY` | Semantic Scholar API key (100 req/s vs ~1 req/s) |

## arXiv Categories

| Category | Area |
|----------|------|
| `cs.AI` | Artificial Intelligence |
| `cs.CL` | Computation and Language (NLP) |
| `cs.CV` | Computer Vision |
| `cs.LG` | Machine Learning |
| `cs.NE` | Neural and Evolutionary Computing |
| `stat.ML` | Machine Learning (Statistics) |
| `cs.IR` | Information Retrieval |
| `cs.RO` | Robotics |
