---
name: paper-research
description: "Search, retrieve, and synthesize academic ML/CS papers into a literature review. Use this skill whenever the user asks for a literature review, research survey, paper search, state-of-the-art overview, 'what's the latest research on X', 'find papers about Y', 'summarize recent work on Z', or any task involving discovering and synthesizing academic papers. Also trigger when the user mentions arxiv papers, trending ML papers, or wants to understand the research landscape around a topic — even if they don't explicitly say 'literature review'."
---

# Literature Review Agent

Search, retrieve, and synthesize academic ML/CS papers into a structured literature review using a unified paper index spanning 60K+ papers across HF Papers, arXiv, OpenReview (ICLR/ICML/NeurIPS), and ACL Anthology (ACL/EMNLP/NAACL/EACL/Findings).

## Prerequisites

```bash
pip install arxiv                            # arxiv API client (v3+)
pip install openreview-py                    # OpenReview API client
pip install acl-anthology                    # ACL Anthology data access
pip install scikit-learn                     # for LSA semantic search
pip install tenacity                         # exponential backoff for API calls
pip install pypdf                            # for extracting affiliations from PDFs
pip install -U "huggingface_hub[cli]"        # hf papers CLI (>= 1.12.0)
```

## Tools

The skill uses a unified search index backed by four source caches:

- **`tools/unified_search.py`** — primary interface: update, search, enrich, stats across all sources
- `tools/hf_papers_cache.py` — source fetcher for HF daily papers
- `tools/arxiv_search.py` — source fetcher for arXiv + direct API search
- `tools/openreview_search.py` — source fetcher for OpenReview conferences
- `tools/acl_anthology_search.py` — source fetcher for ACL Anthology
- `tools/citation_counts.py` — Semantic Scholar citation lookup (used internally by unified_search)

## Workflow

### Step 0: Update cache

Before any research session, ensure the unified index is current:

```bash
python tools/unified_search.py update
```

This refreshes all four source caches (HF daily papers, arXiv last N days, OpenReview, ACL Anthology), rebuilds the unified index, and enriches new papers with citation counts from Semantic Scholar.

To refresh only one source: `python tools/unified_search.py update --source hf`

### Step 1: Understand the request

Before searching, clarify what the user needs:
- **Topic**: the core research question or area
- **Time range**: how far back to look (default: 7 days for recent work, 30+ days for broader surveys)
- **Depth**: quick scan (5-10 papers, summary-level) or deep review (10-20 papers, detailed analysis)
- **Focus**: any specific angles, methods, or applications they care about
- **Report structure**: ask if, after the research is done, the user would like to brainstorm the structure of the report before writing it. If yes, follow the collaborative structuring process described in Step 5b below.

If the user's request is clear enough, skip the clarification and proceed directly.

### Step 2: Keyword discovery

Given the user's intent, brainstorm 5-10 search queries that approach the topic from different angles. This is critical — a single query will miss relevant work.

For example, for "memory architectures for LLM agents":
- Core framing: `"memory tool LLM agent"`
- Specific mechanisms: `"episodic semantic memory retrieval agent"`
- Training approach: `"reinforcement learning memory policy agent"`
- Storage backend: `"knowledge graph memory agent temporal"`
- Context management: `"context compression agent long horizon"`
- Alternative framing: `"persistent state conversational agent"`

Generate these autonomously and proceed. Each query surfaces papers the others miss.

### Step 3: Search

For each discovered keyword, search **per source** with limit 50, plus live searches for freshness:

**A. Unified index search (per source, limit 50)**

```bash
# Peer-reviewed conferences (highest credibility)
python tools/unified_search.py search -e "<query>" --source openreview --limit 50 --format json
python tools/unified_search.py search -e "<query>" --source acl --limit 50 --format json

# Community-curated (trending signal)
python tools/unified_search.py search -e "<query>" --source hf --limit 50 --format json

# Preprints (broadest coverage)
python tools/unified_search.py search -e "<query>" --source arxiv --limit 50 --format json
```

**B. Live keyword search (catches papers too new for cache or differently indexed)**

```bash
# HF native search — pipe results into cache
hf papers search "<query>" --limit 20 --format json | python tools/unified_search.py ingest --source hf

# arXiv API direct search — pipe results into cache
python tools/arxiv_search.py search-api "<query>" --limit 20 --format json | python tools/unified_search.py ingest --source arxiv
```

Always run both cached semantic search AND live keyword search. They surface different papers — semantic search finds conceptually related work, keyword search finds exact terminology matches. Piping through `ingest` adds any new discoveries to the unified cache so they appear in subsequent searches and are preserved for future sessions.

**C. Trending browse (when user wants "what's new")**

```bash
hf papers list --sort trending --date today --format json
hf papers list --sort trending --week <YYYY-Www> --format json
```

**D. Deduplicate**

After all searches complete, deduplicate by arXiv ID or title. Papers appearing in multiple sources get noted — this is a strong quality signal.

### Step 4: Triage

For every paper returned by search, assess relevance based on its abstract. Operate autonomously — only ask the user if you are genuinely uncertain whether a paper fits their intent.

**Credibility ordering** (use this to rank papers of similar relevance):

1. **Peer-reviewed conferences** (OpenReview + ACL Anthology) — latest year first, then older years
2. **HF Papers** — community-curated, upvote signal confirms practitioner interest
3. **arXiv only** — needs credibility check (see below)

**Credibility filter for arXiv-only papers**: Papers from OpenReview or ACL Anthology are peer-reviewed — no further check needed. Papers from HF are community-curated. Papers sourced ONLY from arXiv need a credibility check:

- **Recent papers (<3 months)**: citation count of 0 is expected — use affiliation as the primary signal
- **Older papers (3+ months)**: 0 citations is a red flag — check affiliations carefully
- **Any paper with 50+ citations**: credible regardless of affiliation
- **Include** if authors are affiliated with reputed institutions (Google, Meta, Microsoft, Anthropic, OpenAI, DeepMind, Allen AI, Tsinghua, Stanford, CMU, MIT, Berkeley, etc.)
- **Downgrade or exclude** papers from unknown affiliations with 0 citations AND older than 3 months

**How to check affiliations** (in priority order — stop at the first that works):
1. `WebFetch https://ar5iv.labs.arxiv.org/html/<arxiv_id>` — ar5iv renders LaTeX to HTML and includes affiliations.
2. If ar5iv fails, download PDF: `curl -sL -o $TMPDIR/<arxiv_id>.pdf https://arxiv.org/pdf/<arxiv_id>` and extract page 1 with `pypdf`.
3. Recognize well-known authors by name (e.g., Yejin Choi → UW/Allen AI). A single well-known co-author is enough.
4. If none work, downgrade to tangential.

For each paper, record a **triage entry** in the paper trace (see Paper Trace Format below):

- **Relevance verdict**: `directly relevant` / `tangentially relevant` / `not relevant`
- **Confidence**: `high` / `medium` / `low`
- **Reasoning**: 1-2 sentences explaining why

Papers marked `directly relevant` proceed to deep-read. Papers marked `tangentially relevant` are included in the review using their abstract only. Papers marked `not relevant` are excluded but still recorded in the trace.

### Step 5: Deep-read relevant papers

For every `directly relevant` paper (typically 3-7), retrieve the full paper content. Use a multi-tier fallback — try each in order and stop at the first success:

| Priority | Method | Notes |
|----------|--------|-------|
| 1 | `hf papers read <arxiv_id>` | Cleanest output, but only HF-indexed papers |
| 2 | `WebFetch https://ar5iv.labs.arxiv.org/html/<arxiv_id>` | Converts LaTeX to HTML on-demand |
| 3 | Download PDF and extract with `pypdf` | Works for any paper with a PDF |
| 4 | Abstract only | Last resort — always available from search results |

**PDF fallback** (Priority 3):

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

For **OpenReview-only papers** (no arXiv preprint):
```bash
curl -sL -o $TMPDIR/<paper_id>.pdf "https://openreview.net/pdf?id=<paper_id>"
```

Record which source was used in the paper's trace entry.

After reading each paper, extract structured notes:

- **Core contribution**: what is novel about this work
- **Methodology**: key techniques, architectures, training approaches
- **Key results**: main quantitative or qualitative findings
- **Limitations**: what the authors acknowledge or what you observe
- **Relevance to query**: how this paper connects to the user's question

### Step 5b: Collaborative report structuring (optional)

If the user opted in during Step 1, pause here before writing the review. Propose a report structure based on what you found — section headings, the grouping/taxonomy you plan to use, which papers go where. Present it as a short outline the user can react to.

Wait for the user's feedback. Incorporate their input before proceeding.

### Step 6: Synthesize into a review

The user's intent determines the review format. Infer the right format from how they phrased their request.

Every review shares this skeleton:

```markdown
# [Title reflecting the user's request]

## Overview
What this review covers, how papers were found, and a 2-3 sentence summary of the landscape.

[... body — format varies by intent, see below ...]

## Bibliography
- [Paper Title](traces/<paper_id>.md) — Author et al., Year. arXiv:<id>. *Venue. Citations: N.* *Verdict: directly relevant*
- ...
```

Include all triaged papers in the Bibliography. Each entry links to that paper's trace file.

**Body format by intent:**

**Survey / Literature review** — "survey of RLHF techniques", "review work on X"
Organize thematically. Group papers by shared ideas, methods, or findings. Synthesize across papers within each theme. End with key trends and open problems.

**What's new / Digest** — "what's trending in ML this week", "recent papers on X"
Curated digest with short summaries per paper. Lead with the most notable work. Emphasize what's new or surprising.

**Comparison** — "compare approaches to efficient inference", "X vs Y"
Side-by-side analysis. Use a comparison table for key dimensions, then prose discussing tradeoffs.

**Related work** — "find papers related to my work on X"
Connection-oriented. For each paper, emphasize how it relates to the user's specific work.

**Deep dive** — "explain this paper and the work around it"
Anchor on one paper. Summarize it in depth, then map the surrounding landscape.

**Landscape** — "who's working on multimodal reasoning"
Research-group oriented. Cluster by lab/team/institution.

**Citation style**: Always write "Author et al. (2025)" — include the year.

**Tables must reference papers**: Always include a column listing specific papers in categorization tables.

**Formatting rule**: Always add a blank line before and after every markdown table.

### Step 7: Save and present

Save outputs to a directory in the current working directory:

```
literature_review_<topic>/
  review.md                     # the final literature review
  traces/
    <paper_id>.md               # one trace file per paper encountered
```

Tell the user where the directory is and present a summary of the review in the conversation.

## Paper Trace Format

Create a trace file for **every** paper encountered during the review. Each paper gets its own file at `traces/<paper_id>.md`:

```markdown
# [Paper Title]

**Paper ID**: <arxiv_id or openreview:id or acl:bibkey>
**Authors**: ...
**Date**: ...
**Source(s)**: hf, arxiv, openreview, acl (whichever found it)
**Venue**: ICLR 2025 / EMNLP 2024 / etc. (if applicable)
**Citations**: <N> (<M> influential) — via Semantic Scholar
**HF Upvotes**: <N> (if applicable)

## Abstract
<full abstract>

## Triage
- **Verdict**: directly relevant / tangentially relevant / not relevant
- **Confidence**: high / medium / low
- **Reasoning**: why this paper is or isn't relevant to the query

## Deep-Read Notes
*Omit this section entirely for papers not deep-read.*

- **Source used**: hf papers read / ar5iv HTML / PDF extraction
- **Core contribution**: ...
- **Methodology**: ...
- **Key results**: ...
- **Limitations**: ...
- **Relevance to query**: ...
```

Write each trace file as you go — create it during triage, then append deep-read notes if the paper proceeds to Step 5.

## Command Reference

See [references/cli-reference.md](references/cli-reference.md) for the full command reference.

## Tips

- Run `python tools/unified_search.py stats` to see cache coverage before starting.
- If semantic search returns poor results, try keyword search as fallback.
- For very broad topics, break into sub-queries across different facets.
- When the user asks for "recent" work without specifying, default to the last 7 days. For a "survey" or "overview", go broader (30 days or more).
- For the ar5iv fallback, strip header/footer boilerplate. The paper content starts at the title and ends at the last reference.
- Use `--since` flag to limit search to a time window: `python tools/unified_search.py search -e "query" --since 2025-04-01`
