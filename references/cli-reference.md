# CLI Reference

## arxiv-search (tools/arxiv_search.py)

Custom arXiv CLI with local SQLite cache and LSA semantic search. Uses the `arxiv` Python package (v3+) to query the arXiv API, caches results locally, and provides both keyword and LSA-based search.

DB path: `~/.local/share/arxiv-search/papers.db`

### Commands

**`fetch`** — Populate local DB from arXiv API

```bash
python tools/arxiv_search.py fetch --num-days 7 --categories cs.AI,cs.CL,cs.CV,cs.LG,cs.NE,stat.ML,cs.IR
```

| Flag | Description |
|------|-------------|
| `--num-days N` | Days of papers to fetch (default: 7) |
| `--categories` | Comma-separated arXiv categories |
| `--max-results N` | Max papers to fetch per run (default: 2000) |

Fetching is idempotent — duplicates are skipped by arXiv ID.

**`search`** — Search the local DB

```bash
python tools/arxiv_search.py search "transformer efficiency" --limit 10         # keyword match
python tools/arxiv_search.py search -e "transformer efficiency" --limit 10      # LSA semantic search
python tools/arxiv_search.py search -e -f "transformer efficiency" --limit 10   # force retrain LSA model
python tools/arxiv_search.py search -e "query" --limit 10 --format json         # JSON output
```

| Flag | Description |
|------|-------------|
| `-e, --semantic` | LSA semantic search (better relevance than keyword match) |
| `-f, --force-retrain` | Force retrain LSA model (use with `-e` after DB refresh) |
| `-l, --limit N` | Max results to return (default: 10) |
| `--format text\|json` | Output format (default: text) |

The LSA model is trained on first semantic search and cached. It auto-retrains when the DB size changes. Use `-f` to force retrain.

**`search-api`** — Search arXiv API directly (no local DB needed)

```bash
python tools/arxiv_search.py search-api "memory tool LLM agent" --limit 20
python tools/arxiv_search.py search-api "memory tool LLM agent" --categories cs.AI,cs.CL,cs.LG --limit 20
python tools/arxiv_search.py search-api "memory tool LLM agent" --format json
python tools/arxiv_search.py search-api "memory tool LLM agent" --sort date --limit 10
```

| Flag | Description |
|------|-------------|
| `-l, --limit N` | Max results to return (default: 20) |
| `--sort relevance\|date` | Sort order (default: relevance) |
| `--categories` | Comma-separated arXiv categories to filter (optional) |
| `--format text\|json` | Output format (default: text) |

Queries the arXiv API directly — no fetching or local DB needed. Searches across all of arXiv history. Best for broad surveys or time ranges beyond what's practical to fetch locally.

**`stats`** — DB summary statistics

```bash
python tools/arxiv_search.py stats
```

### Relevant Categories

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

---

## openreview-search (tools/openreview_search.py)

Search peer-reviewed papers from top ML/AI conferences via OpenReview. Uses cached JSON files at `~/.openreview/`.

Supported conferences: **ICLR, ICML, NeurIPS, EMNLP, NAACL, AAAI** (2024-2025 cached).

### Commands

**`search`** — Search cached conference papers

```bash
python tools/openreview_search.py search "memory agent" --limit 10                          # keyword
python tools/openreview_search.py search -e "memory mechanisms for agents" --limit 10        # LSA semantic
python tools/openreview_search.py search -e "query" --conferences ICLR,ICML --years 2025     # filter
python tools/openreview_search.py search -e "query" --format json                            # JSON output
```

| Flag | Description |
|------|-------------|
| `-e, --semantic` | LSA semantic search |
| `-l, --limit N` | Max results (default: 10) |
| `--conferences` | Comma-separated conferences to search (default: all) |
| `--years` | Comma-separated years (default: 2024,2025) |
| `--format text\|json` | Output format (default: text) |

**`fetch`** — Fetch papers from OpenReview API into cache

```bash
python tools/openreview_search.py fetch --conference ICLR --year 2026
```

Requires `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` environment variables.

**`stats`** — Show cache statistics

```bash
python tools/openreview_search.py stats
```

---

## acl-anthology-search (tools/acl_anthology_search.py)

Local SQLite cache + LSA semantic search for ACL Anthology papers. Caches papers from ACL-family conferences (ACL, EMNLP, NAACL, EACL, Findings) into `~/.local/share/acl-anthology/papers.db`.

### Commands

**`fetch`** — Populate cache from ACL Anthology

```bash
python tools/acl_anthology_search.py fetch                                    # fetch all default collections (2024-2025)
python tools/acl_anthology_search.py fetch --collections 2024.emnlp,2025.acl   # specific collections
```

| Flag | Description |
|------|-------------|
| `--collections` | Comma-separated collection IDs (default: ACL/EMNLP/NAACL/EACL/Findings 2024-2025) |

Default collections: `2024.acl`, `2025.acl`, `2024.emnlp`, `2025.emnlp`, `2024.naacl`, `2025.naacl`, `2024.eacl`, `2024.findings`, `2025.findings`

First fetch clones the ACL Anthology git repo (~1 min) then parses XML data. Subsequent fetches reuse the local clone.

**`search`** — Search cached papers

```bash
python tools/acl_anthology_search.py search "vision language" --limit 10           # keyword match
python tools/acl_anthology_search.py search -e "multimodal reasoning" --limit 10   # LSA semantic search
python tools/acl_anthology_search.py search -e "query" --venue emnlp --limit 10    # filter by venue
python tools/acl_anthology_search.py search -e "query" --limit 10 --format json    # JSON output
```

| Flag | Description |
|------|-------------|
| `-e, --semantic` | LSA semantic search (better relevance than keyword match) |
| `-l, --limit N` | Max results to return (default: 10) |
| `--format text\|json` | Output format (default: text) |
| `--venue` | Filter by venue (e.g., emnlp, acl, naacl, findings) |

**`stats`** — Cache summary statistics

```bash
python tools/acl_anthology_search.py stats
```

---

## hf-papers-cache (tools/hf_papers_cache.py)

Local SQLite cache + LSA semantic search for Hugging Face daily papers. Fetches paper metadata from the HF API day-by-day and stores in `~/.local/share/hf-papers/papers.db`. Cache covers all daily papers from 2025-01-01 onward.

### Commands

**`fetch`** — Populate cache from HF API

```bash
python tools/hf_papers_cache.py fetch                         # fetch all missing days (2025-01-01 to today)
python tools/hf_papers_cache.py fetch --start 2025-03-01      # fetch from a specific start date
python tools/hf_papers_cache.py fetch --end 2025-04-01        # fetch up to a specific end date
```

| Flag | Description |
|------|-------------|
| `--start YYYY-MM-DD` | Start date (default: 2025-01-01) |
| `--end YYYY-MM-DD` | End date (default: today) |

Fetching is idempotent — already-fetched days are skipped. First full fetch takes ~3 minutes (489 days). Adds ~0.3s per day due to rate limiting.

**`update`** — Fetch only new days since last fetch

```bash
python tools/hf_papers_cache.py update
```

Convenience shortcut: finds the last fetched date and fetches from there to today. Run this at the start of every research session (Step 0).

**`search`** — Search cached papers

```bash
python tools/hf_papers_cache.py search "vision transformer" --limit 10         # keyword match
python tools/hf_papers_cache.py search -e "multimodal reasoning" --limit 10    # LSA semantic search
python tools/hf_papers_cache.py search -e "query" --limit 10 --format json     # JSON output
python tools/hf_papers_cache.py search -e "query" --since 2025-04-01           # date filter
```

| Flag | Description |
|------|-------------|
| `-e, --semantic` | LSA semantic search (better relevance than keyword match) |
| `-l, --limit N` | Max results to return (default: 10) |
| `--format text\|json` | Output format (default: text) |
| `--since YYYY-MM-DD` | Only search papers from this date onward |

Keyword search sorts by upvotes (descending), so trending/popular papers surface first. Semantic search ranks by LSA cosine similarity. The LSA model is auto-trained on first semantic search and cached; it retrains when the cache size changes.

**`stats`** — Cache summary statistics

```bash
python tools/hf_papers_cache.py stats
```

---

## hf papers (huggingface_hub CLI)

Install: `pip install -U "huggingface_hub[cli]"` (requires >= 1.12.0)

Used for browsing trending papers by time window and reading full paper content. For search, use the `hf-papers-cache` tool above instead.

### Commands

**`hf papers list`** — Browse by date or trending
```bash
hf papers list --sort trending --date today --format json
hf papers list --sort trending --week 2026-W18 --format json
hf papers list --sort trending --month 2026-04 --format json
```

| Flag | Description |
|------|-------------|
| `--sort trending` | Sort by trending |
| `--date YYYY-MM-DD` | Papers from specific date (or `today`) |
| `--week YYYY-Www` | Papers from ISO week |
| `--month YYYY-MM` | Papers from month |
| `--limit N` | Max results (default: 50, max: 100) |
| `--format json` | Machine-parseable JSON output |

**`hf papers info`** — Paper metadata (JSON)
```bash
hf papers info 2601.15621
```

**`hf papers read`** — Full paper as markdown
```bash
hf papers read 2601.15621
```

---

## citation-counts (tools/citation_counts.py)

Looks up citation counts from Semantic Scholar by arXiv ID. Supports batch lookup (up to 400 IDs per request). No authentication required.

### Commands

**`lookup`** — Get citation counts for arXiv IDs

```bash
python tools/citation_counts.py lookup 2401.06209 2310.06825 2403.09611
python tools/citation_counts.py lookup --format json 2401.06209 2310.06825
echo "2401.06209\n2310.06825" | python tools/citation_counts.py lookup --stdin
```

| Flag | Description |
|------|-------------|
| `--format text\|json` | Output format (default: text) |
| `--stdin` | Read arXiv IDs from stdin (one per line) |

Returns `citationCount` (total) and `influentialCitationCount` (citations where the citing paper substantially builds on this work) for each paper.

Interpret citation counts relative to paper age:
- **Very recent** (<3 months): 0-10 citations is normal
- **Moderate age** (3-12 months): 50+ is strong signal
- **Older** (1+ year): 100+ indicates established work
