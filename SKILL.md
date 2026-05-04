---
name: paper-research
description: "Search, retrieve, and synthesize academic ML/CS papers into a literature review. Use this skill whenever the user asks for a literature review, research survey, paper search, state-of-the-art overview, 'what's the latest research on X', 'find papers about Y', 'summarize recent work on Z', or any task involving discovering and synthesizing academic papers. Also trigger when the user mentions arxiv papers, trending ML papers, or wants to understand the research landscape around a topic — even if they don't explicitly say 'literature review'."
---

# Literature Review Agent

Search, retrieve, and synthesize academic ML/CS papers into a structured literature review using the `arxiv` and `hf papers` CLI tools.

## Prerequisites

These must be installed:

```bash
pip install arxiv                            # arxiv API client (v3+)
pip install openreview-py                    # OpenReview API client
pip install acl-anthology                    # ACL Anthology data access
pip install scikit-learn                     # for LSA semantic search
pip install pypdf                            # for extracting affiliations from PDFs
pip install -U "huggingface_hub[cli]"        # hf papers CLI (>= 1.12.0)
```

The skill includes four custom search tools:
- `tools/hf_papers_cache.py` — local SQLite cache + LSA semantic search for Hugging Face daily papers (cache at `~/.local/share/hf-papers/papers.db`)
- `tools/arxiv_search.py` — local SQLite cache + LSA semantic search for arXiv papers
- `tools/openreview_search.py` — search peer-reviewed papers from top ML conferences (ICLR, ICML, NeurIPS) via OpenReview cached JSON files at `~/.openreview/`
- `tools/acl_anthology_search.py` — local SQLite cache + LSA semantic search for ACL Anthology papers (ACL, EMNLP, NAACL, EACL, Findings 2024-2025)

## Workflow

### Step 0: Update caches

Before any search, ensure the HF papers cache is current. Run the update command — it only fetches days not already cached, so it's fast when the cache is recent:

```bash
python tools/hf_papers_cache.py update
```

This fetches any missing days from the last cached date to today. The cache stores all HF daily papers from 2025-01-01 onward in a local SQLite database at `~/.local/share/hf-papers/papers.db`.

If this is the very first run (empty cache), it will fetch all days from 2025-01-01 — this takes ~3 minutes. Subsequent runs only fetch new days and complete in seconds.

### Step 1: Understand the request

Before searching, clarify what the user needs:
- **Topic**: the core research question or area
- **Time range**: how far back to look (default: 7 days for recent work, 30+ days for broader surveys)
- **Depth**: quick scan (5-10 papers, summary-level) or deep review (10-20 papers, detailed analysis)
- **Focus**: any specific angles, methods, or applications they care about
- **Report structure**: ask if, after the research is done, the user would like to brainstorm the structure of the report before writing it. If yes, follow the collaborative structuring process described in Step 4b below.

If the user's request is clear enough, skip the clarification and proceed directly.

### Step 2: Search for papers

Use four complementary discovery tiers. Each tier has different strengths — use all for comprehensive coverage:

1. **HF Papers** — community-curated, trending, highest signal for what's new and hot
2. **OpenReview** — peer-reviewed papers from top ML conferences (ICLR, ICML, NeurIPS), confirmed quality
3. **ACL Anthology** — peer-reviewed NLP/CL papers (ACL, EMNLP, NAACL, EACL, Findings), confirmed quality
4. **arXiv** — bleeding edge preprints, broadest coverage, requires credibility filter

**A. HF Papers — trending and curated** (start here, highest signal-to-noise)

Hugging Face daily papers are community-curated — trending papers have been upvoted by ML practitioners, making this the best signal for what the community considers important. Always check HF papers, even if the user's request is keyword-specific.

All HF paper searches use the local cache (`tools/hf_papers_cache.py`). The cache was updated in Step 0.

```bash
# Semantic search (best relevance)
python tools/hf_papers_cache.py search -e "<query>" --limit 20 --format json

# Keyword search (fallback)
python tools/hf_papers_cache.py search "<query>" --limit 20 --format json

# Search only recent papers (e.g., last 30 days)
python tools/hf_papers_cache.py search -e "<query>" --since 2025-04-01 --limit 20 --format json

# Check cache coverage
python tools/hf_papers_cache.py stats
```

Results are sorted by upvotes (keyword) or semantic relevance (LSA), so trending/popular papers naturally surface first.

For browsing trending papers by time window (when the user wants "what's new today/this week"), you can still use the `hf papers` CLI directly:

```bash
hf papers list --sort trending --date today --format json
hf papers list --sort trending --week <YYYY-Www> --format json
```

Use `--format json` so results are machine-parseable.

**B. OpenReview — peer-reviewed conference papers** (highest credibility, no credibility filter needed)

Searches papers accepted at top ML conferences: **ICLR, ICML, NeurIPS**. These are peer-reviewed — no need for the arXiv credibility filter. The cache at `~/.openreview/` contains 26K+ papers from 2024-2026.

```bash
# Semantic search across all cached conferences
python tools/openreview_search.py search -e "<query>" --limit 20

# Filter by conference and year
python tools/openreview_search.py search -e "<query>" --conferences ICLR,ICML --years 2025 --limit 20

# Keyword search (fallback)
python tools/openreview_search.py search "<query>" --limit 20

# JSON output
python tools/openreview_search.py search -e "<query>" --limit 20 --format json

# Check what's cached
python tools/openreview_search.py stats

# Fetch fresh data (requires OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD env vars)
python tools/openreview_search.py fetch --conference ICLR --year 2026
```

OpenReview papers carry the strongest quality signal (peer review at top venues). Always search here for survey-type reviews, and prefer OpenReview/ACL Anthology results over arXiv when the same work appears in both.

**C. ACL Anthology — peer-reviewed NLP/CL papers** (highest credibility for NLP, no credibility filter needed)

Searches papers from ACL-family conferences: **ACL, EMNLP, NAACL, EACL, Findings** (2024-2025). The cache at `~/.local/share/acl-anthology/papers.db` contains 14K+ papers.

```bash
# Semantic search across all cached venues
python tools/acl_anthology_search.py search -e "<query>" --limit 20

# Filter by venue
python tools/acl_anthology_search.py search -e "<query>" --venue emnlp --limit 20

# Keyword search (fallback)
python tools/acl_anthology_search.py search "<query>" --limit 20

# JSON output
python tools/acl_anthology_search.py search -e "<query>" --limit 20 --format json

# Check what's cached
python tools/acl_anthology_search.py stats
```

Use this tier especially for NLP/CL-focused queries (language models, text generation, machine translation, dialogue, summarization, etc.). These are fully peer-reviewed papers — no credibility filter needed.

**D. arXiv search — bleeding edge preprint coverage**

Two modes available — use both for best coverage:

**D1. Local DB + LSA search** (best for recent papers, 7-30 days)

Fetches papers into a local SQLite cache, then searches with LSA semantic similarity.

```bash
# Populate the local database first (required before searching)
python tools/arxiv_search.py fetch --num-days <DAYS> --categories cs.AI,cs.CL,cs.CV,cs.LG,cs.NE,stat.ML,cs.IR

# LSA semantic search for best relevance
python tools/arxiv_search.py search -e "<query>" --limit 20

# Keyword search (fallback)
python tools/arxiv_search.py search "<query>" --limit 20

# JSON output for machine parsing
python tools/arxiv_search.py search -e "<query>" --limit 20 --format json
```

Pick `--num-days` based on the user's time range. Use `-e` for semantic search (better than pattern matching for research topics). If this is the first semantic search or the DB was just refreshed, add `-f` to force-retrain the LSA model.

**D2. Direct API search** (best for any time horizon, no fetching needed)

Queries the arXiv API directly, sorted by relevance. No local DB required — searches across all of arXiv history. Use this for broader surveys or when the user's time range exceeds what's practical to fetch locally.

```bash
# Relevance-ranked search across all of arXiv
python tools/arxiv_search.py search-api "<query>" --limit 20

# Filter by categories
python tools/arxiv_search.py search-api "<query>" --limit 20 --categories cs.AI,cs.CL,cs.LG

# JSON output
python tools/arxiv_search.py search-api "<query>" --limit 20 --format json
```

Use `--sort relevance` (default) for topical searches. Do NOT use `--sort date` — arXiv's API applies the date sort before the relevance filter, so results will be irrelevant to the query. To filter by date, use `--sort relevance` and filter results client-side by the `published` field.

When using search-api, **request more results than you need** (e.g., `--limit 30` when you want 10-15) — the credibility filter in Step 3 will exclude some papers, and you need a buffer.

**Search strategy: use multiple query angles.** A single query will miss relevant work. For any topic, run at least 3-5 queries that approach it from different angles. For example, for "memory architectures for LLM agents":
- Core framing: `"memory tool LLM agent"`
- Specific mechanisms: `"episodic semantic memory retrieval agent"`
- Training approach: `"reinforcement learning memory policy agent"`
- Storage backend: `"knowledge graph memory agent temporal"`
- Context management: `"context compression agent long horizon"`

Each query surfaces papers the others miss. Deduplicate across queries by arXiv ID.

**E. Cross-reference and deduplicate**

Papers often appear across multiple sources. Deduplicate by title or arXiv ID. When a paper appears in multiple tiers, note that in its trace — it's a strong relevance signal. A paper that appears in OpenReview + HF trending or ACL Anthology + HF trending is very high confidence. A paper that appears only on arXiv needs the credibility filter.

### Step 3: Abstract triage

For every paper returned by search, assess relevance based on its abstract. Operate autonomously — only ask the user if you are genuinely uncertain whether a paper fits their intent.

**Citation count enrichment**: After deduplication, enrich all candidate papers with citation counts from Semantic Scholar. Collect all arXiv IDs and run a batch lookup:

```bash
python tools/citation_counts.py lookup <arxiv_id_1> <arxiv_id_2> ... --format json
```

This returns `citationCount` and `influentialCitationCount` for each paper. Use these as quality/impact signals during triage:
- **High-impact** (50+ citations or 5+ influential): strong credibility signal, even from unknown authors
- **Moderate** (10-50 citations): established work, treat as credible
- **Low/zero** (0-10 citations): expected for very recent papers (<3 months old) — don't penalize recency, use other signals instead

Citation counts are most useful for arXiv-only papers where there's no peer-review or community curation signal.

**Credibility filter for arXiv-sourced papers**: HF Papers are community-curated and carry implicit quality signal (upvotes, trending). OpenReview and ACL Anthology papers are peer-reviewed. arXiv papers have no such filter — anyone can post. For papers sourced only from arXiv (not also appearing on HF or conference venues), apply a credibility check before including them:
- **Include** if the paper has significant citations (50+), OR authors are affiliated with reputed research institutions (e.g., universities with known ML programs) or established companies/labs (e.g., Google, Meta, Microsoft, Anthropic, OpenAI, DeepMind, Allen AI, Tsinghua, Stanford, CMU, MIT, Berkeley, etc.).
- **Downgrade or exclude** papers with zero citations AND unknown affiliations, solo authors without institutional backing, or preprints that read more like blog posts than research papers.
- Check the author list and any stated affiliations in the abstract or paper metadata. When in doubt, look up the authors — a single well-known co-author is sufficient.

**How to check affiliations** (in priority order — stop at the first that works):
1. `WebFetch https://ar5iv.labs.arxiv.org/html/<arxiv_id>` — ar5iv renders LaTeX to HTML and includes affiliations. Works for papers processed a few days after posting.
2. If ar5iv redirects (paper too recent), download the PDF with `curl -sL -o <local_path>.pdf https://arxiv.org/pdf/<arxiv_id>` and extract the first page with `pypdf` — affiliations are always on page 1.
3. Recognize well-known authors by name (e.g., Yejin Choi → UW/Allen AI, Julian McAuley → UCSD). A single well-known co-author is enough.
4. If none of the above work, downgrade to tangential rather than including as directly relevant.

For each paper, record a **triage entry** in the paper trace (see Paper Trace Format below):

- **Relevance verdict**: `directly relevant` / `tangentially relevant` / `not relevant`
- **Confidence**: `high` / `medium` / `low`
- **Reasoning**: 1-2 sentences explaining why

Papers marked `directly relevant` proceed to deep-read. Papers marked `tangentially relevant` are included in the review using their abstract only. Papers marked `not relevant` are excluded but still recorded in the trace for auditability.

### Step 4: Deep-read relevant papers

For every `directly relevant` paper (typically 3-7), retrieve the full paper content. Use a multi-tier fallback — try each in order and stop at the first success:

| Priority | Method | Notes |
|----------|--------|-------|
| 1 | `hf papers read <arxiv_id>` | Cleanest output, but only HF-indexed papers |
| 2 | `WebFetch https://ar5iv.labs.arxiv.org/html/<arxiv_id>` | Converts LaTeX to HTML on-demand, works for nearly any arxiv paper |
| 3 | Download PDF and extract with `pypdf` | Works for any paper with a PDF available (see below) |
| 4 | Abstract only | Last resort — always available from search results |

**PDF fallback** (Priority 3): When ar5iv fails (conversion error, redirect, or truncated content), download and extract the PDF directly:

For **arXiv papers**:
```bash
curl -sL -o $TMPDIR/<arxiv_id>.pdf https://arxiv.org/pdf/<arxiv_id>
python3 -c "
from pypdf import PdfReader
r = PdfReader('$TMPDIR/<arxiv_id>.pdf')
text = ''
for i in range(min(10, len(r.pages))):
    text += r.pages[i].extract_text() + '\n---PAGE BREAK---\n'
print(text[:12000])
"
```

For **OpenReview-only papers** (no arXiv preprint): Use the OpenReview PDF URL format:
```bash
curl -sL -o $TMPDIR/<paper_id>.pdf "https://openreview.net/pdf?id=<paper_id>"
python3 -c "
from pypdf import PdfReader
r = PdfReader('$TMPDIR/<paper_id>.pdf')
text = ''
for i in range(min(10, len(r.pages))):
    text += r.pages[i].extract_text() + '\n---PAGE BREAK---\n'
print(text[:12000])
"
```

This ensures every paper — whether on arXiv, OpenReview, or both — can be fully read. Never settle for abstract-only when a PDF is available.

Record which source was used in the paper's trace entry.

After reading each paper, extract structured notes:

- **Core contribution**: what is novel about this work
- **Methodology**: key techniques, architectures, training approaches
- **Key results**: main quantitative or qualitative findings, comparisons to prior work
- **Limitations**: what the authors acknowledge or what you observe
- **Relevance to query**: how this paper connects to the user's specific question

These notes go into the paper trace and feed directly into the synthesis step.

### Step 4b: Collaborative report structuring (optional)

If the user opted in during Step 1, pause here before writing the review. Propose a report structure based on what you found during research — section headings, the grouping/taxonomy you plan to use, which papers go where, and any cross-cutting themes you want to highlight. Present it as a short outline the user can react to.

Wait for the user's feedback. They may want to reorder sections, merge or split themes, emphasize different angles, add a background section, or adjust the level of detail. Incorporate their input before proceeding to Step 5.

If the user did not opt in, skip this step and proceed directly.

### Step 5: Synthesize into a review

The user's intent determines the review format. Infer the right format from how they phrased their request — don't ask them to pick a mode. Real requests often blend intents; adapt the structure to fit rather than forcing a template.

Every review shares this skeleton:

```markdown
# [Title reflecting the user's request]

## Overview
What this review covers, how papers were found, and a 2-3 sentence summary of the landscape.

[... body — format varies by intent, see below ...]

## Bibliography
- [Paper Title](traces/<arxiv_id>.md) — Author et al., Year. arXiv:<arxiv_id>. *Verdict: directly relevant*
- [Paper Title](traces/<arxiv_id>.md) — Author et al., Year. arXiv:<arxiv_id>. *Verdict: tangentially relevant*
- ...
```

Include all triaged papers in the Bibliography (not just ones cited in the body) so the user has a complete picture. Each entry links to that paper's trace file.

**Choose the body format based on intent:**

**Survey / Literature review** — "survey of RLHF techniques", "review work on X"
Organize thematically, not paper-by-paper. Group papers by shared ideas, methods, or findings. Synthesize across papers within each theme. Fewer well-developed themes beat many thin ones. End with key trends and open problems. Keep the tone academic — describe what each paper contributes, its methodology, and key results. Do NOT include implementation sketches, code blocks, tool signatures, or "how to build this" sections. The user provided an implementation context, not a request for a tutorial; the review should inform architectural decisions, not serve as a how-to guide.

**What's new / Digest** — "what's trending in ML this week", "recent papers on X"
Curated digest with short summaries per paper, loosely grouped. Lead with the most notable work. Emphasize what's new or surprising rather than exhaustive coverage. A few sentences per paper is fine.

**Comparison** — "compare approaches to efficient inference", "X vs Y"
Side-by-side analysis. Use a comparison table for key dimensions (approach, performance, compute cost, limitations), then prose discussing tradeoffs, when to prefer each, and gaps. Anchor around the dimensions the user cares about.

**Related work** — "find papers related to my work on X", "what's adjacent to this approach"
Connection-oriented. For each paper, emphasize how it relates to the user's specific work — shared methods, complementary results, potential conflicts. Group by nature of relationship (extends, competes with, enables, provides theory for).

**Deep dive** — "explain this paper and the work around it", "unpack paper X"
Anchor on one paper. Summarize it in depth, then map the surrounding landscape: what it builds on, what builds on it, concurrent related work. The anchor paper gets the most space; context papers support understanding.

**Landscape** — "who's working on multimodal reasoning", "map the field of X"
Research-group oriented. Cluster papers by lab, team, or institution. Highlight different approaches across groups, collaboration patterns, and where the field's effort is concentrated.

When the user's request blends intents (e.g., "compare recent RLHF approaches and what's trending"), compose elements from multiple formats. Use judgment — the goal is a document the user finds useful, not one that fits a template.

**Citation style**: When citing a paper in prose, always include the year — write "Author et al. (2025)" not just "Author et al." The year gives the reader immediate context on how recent the work is.

**Tables must reference papers**: When creating categorization or comparison tables, always include a column listing the specific papers that fall into each row/category. Don't just describe a category abstractly — ground it in the papers surveyed.

**Formatting rule**: Always add a blank line before and after every markdown table. Without these blank lines, `rich` (and some other renderers) will fail to parse the table and display it as inline text.

### Step 6: Save and present

Save outputs to a directory in the current working directory:

```
literature_review_<topic>/
  review.md                     # the final literature review
  traces/
    <arxiv_id>.md               # one trace file per paper encountered
```

Tell the user where the directory is and present a summary of the review in the conversation.

## Paper Trace Format

Create a trace file for **every** paper encountered during the review, regardless of whether it was included in the final synthesis. This provides full auditability — the user can click any bibliography entry and see exactly what was found, what was read, and why each decision was made.

Each paper gets its own file at `traces/<arxiv_id>.md`:

```markdown
# [Paper Title]

**arXiv ID**: <arxiv_id>
**Authors**: ...
**Date**: ...
**Source**: arxiv search / HF papers search / OpenReview / ACL Anthology
**Citations**: <N> (<M> influential) — via Semantic Scholar

## Abstract
<full abstract>

## Triage
- **Verdict**: directly relevant / tangentially relevant / not relevant
- **Confidence**: high / medium / low
- **Reasoning**: why this paper is or isn't relevant to the query

## Deep-Read Notes
*Omit this section entirely for papers not deep-read.*

- **Source used**: hf papers read / ar5iv HTML
- **Core contribution**: ...
- **Methodology**: ...
- **Key results**: ...
- **Limitations**: ...
- **Relevance to query**: ...
```

Write each trace file as you go — create it during triage, then append deep-read notes if the paper proceeds to Step 4. Don't write traces retroactively after synthesis.

## Command Reference

See [references/cli-reference.md](references/cli-reference.md) for the full command reference for both CLI tools.

## Tips

- If `arxiv-search` returns no results, the local DB may be empty — run `python tools/arxiv_search.py fetch` first.
- For the ar5iv fallback, strip the header/footer boilerplate (navigation links, "Generated by LaTeXML" footer). The paper content starts at the title and ends at the last reference.
- For very broad topics, break the search into sub-queries (e.g., "transformer efficiency" + "model compression" + "knowledge distillation") to get better coverage.
- When the user asks for "recent" work without specifying, default to the last 7 days. For a "survey" or "overview", go broader (30 days or more).
