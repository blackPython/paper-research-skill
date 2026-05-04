#!/usr/bin/env python3
"""
unified-search: Consolidated paper search across all sources.

Single SQLite index over HF Papers, OpenReview, ACL Anthology, and arXiv.
Supports keyword search, LSA semantic search, and citation-count enrichment.

Usage:
    python tools/unified_search.py update                              # refresh all sources + rebuild + enrich new
    python tools/unified_search.py update --source hf                  # refresh only HF, then rebuild
    python tools/unified_search.py update --source arxiv               # refresh only arXiv, then rebuild
    python tools/unified_search.py search "vision transformer" --limit 20
    python tools/unified_search.py search -e "multimodal reasoning" --limit 20
    python tools/unified_search.py search -e "query" --source hf --limit 10
    python tools/unified_search.py search -e "query" --since 2025-04-01
    python tools/unified_search.py enrich                              # enrich all papers missing citations
    python tools/unified_search.py enrich --paper 2401.06209           # enrich a specific paper
    python tools/unified_search.py stats                               # full stats
    python tools/unified_search.py stats --source hf                   # stats for one source
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = SKILL_DIR / ".cache"
DB_PATH = CACHE_DIR / "papers.db"
LSA_PATH = CACHE_DIR / "lsa_model.pkl"

# Source cache locations
HF_DB = Path.home() / ".local" / "share" / "hf-papers" / "papers.db"
ARXIV_DB = Path.home() / ".local" / "share" / "arxiv-search" / "papers.db"
OPENREVIEW_DIR = Path.home() / ".openreview"
ACL_DB = Path.home() / ".local" / "share" / "acl-anthology" / "papers.db"

ALL_SOURCES = ["hf", "arxiv", "openreview", "acl"]


def get_db():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            published_at TEXT,
            sources TEXT,
            venue TEXT,
            hf_upvotes INTEGER DEFAULT 0,
            citation_count INTEGER,
            influential_citations INTEGER,
            github_repo TEXT,
            github_stars INTEGER,
            web_url TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS build_log (
            source TEXT PRIMARY KEY,
            paper_count INTEGER,
            built_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS update_log (
            source TEXT PRIMARY KEY,
            last_updated TEXT
        )
    """)
    conn.commit()
    return conn


def _get_last_updated(conn, source):
    row = conn.execute(
        "SELECT last_updated FROM update_log WHERE source = ?", (source,)
    ).fetchone()
    return row["last_updated"] if row else None


def _set_last_updated(conn, source):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO update_log (source, last_updated) VALUES (?, ?)",
        (source, now),
    )
    conn.commit()


# --- Import functions ---

def _merge_paper(conn, paper_id, title, authors, abstract, published_at,
                 source, venue=None, hf_upvotes=0, github_repo=None,
                 github_stars=None, web_url=None):
    existing = conn.execute(
        "SELECT sources, hf_upvotes, venue FROM papers WHERE paper_id = ?",
        (paper_id,)
    ).fetchone()

    if existing:
        sources = set(existing["sources"].split(","))
        sources.add(source)
        new_upvotes = max(existing["hf_upvotes"] or 0, hf_upvotes or 0)
        conn.execute("""
            UPDATE papers SET sources = ?, hf_upvotes = ?, venue = COALESCE(?, venue),
                   github_repo = COALESCE(?, github_repo),
                   github_stars = COALESCE(?, github_stars),
                   web_url = COALESCE(?, web_url)
            WHERE paper_id = ?
        """, (",".join(sorted(sources)), new_upvotes, venue,
              github_repo, github_stars, web_url, paper_id))
    else:
        conn.execute("""
            INSERT INTO papers (paper_id, title, authors, abstract, published_at,
                               sources, venue, hf_upvotes, github_repo, github_stars, web_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (paper_id, title, authors, abstract, published_at,
              source, venue, hf_upvotes, github_repo, github_stars, web_url))


def _import_hf(conn):
    if not HF_DB.exists():
        print("  HF cache not found, skipping")
        return 0
    src = sqlite3.connect(str(HF_DB))
    src.row_factory = sqlite3.Row
    rows = src.execute("""
        SELECT DISTINCT arxiv_id, title, authors, summary, published_at,
               upvotes, github_repo, github_stars
        FROM papers
    """).fetchall()
    src.close()

    count = 0
    for r in rows:
        _merge_paper(
            conn,
            paper_id=r["arxiv_id"],
            title=r["title"],
            authors=r["authors"],
            abstract=r["summary"],
            published_at=r["published_at"],
            source="hf",
            hf_upvotes=r["upvotes"] or 0,
            github_repo=r["github_repo"],
            github_stars=r["github_stars"],
        )
        count += 1
    return count


def _import_arxiv(conn):
    if not ARXIV_DB.exists():
        print("  arXiv cache not found, skipping")
        return 0
    src = sqlite3.connect(str(ARXIV_DB))
    src.row_factory = sqlite3.Row
    rows = src.execute(
        "SELECT arxiv_id, title, authors, abstract, published, url FROM papers"
    ).fetchall()
    src.close()

    count = 0
    for r in rows:
        _merge_paper(
            conn,
            paper_id=r["arxiv_id"],
            title=r["title"],
            authors=json.dumps(r["authors"].split(", ")) if r["authors"] else "[]",
            abstract=r["abstract"],
            published_at=r["published"],
            source="arxiv",
            web_url=r["url"],
        )
        count += 1
    return count


def _import_openreview(conn):
    if not OPENREVIEW_DIR.exists():
        print("  OpenReview cache not found, skipping")
        return 0

    count = 0
    for path in sorted(OPENREVIEW_DIR.glob("*.json")):
        try:
            with open(path) as f:
                papers = json.load(f)
        except Exception:
            continue
        if not papers:
            continue

        conf_year = path.stem
        parts = conf_year.replace("_papers", "").split("_")
        venue = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else parts[0]

        for p in papers:
            title = p.get("title")
            if not title:
                continue
            paper_id = p.get("paper_id", "")
            authors = json.dumps(p.get("authors", [])[:20])
            abstract = p.get("abstract", "")
            year = p.get("year")

            _merge_paper(
                conn,
                paper_id=f"openreview:{paper_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                published_at=f"{year}-01-01" if year else None,
                source="openreview",
                venue=venue,
                web_url=f"https://openreview.net/forum?id={paper_id}",
            )
            count += 1
    return count


def _import_acl(conn):
    if not ACL_DB.exists():
        print("  ACL Anthology cache not found, skipping")
        return 0
    src = sqlite3.connect(str(ACL_DB))
    src.row_factory = sqlite3.Row
    rows = src.execute(
        "SELECT bibkey, title, authors, abstract, year, venue, web_url FROM papers"
    ).fetchall()
    src.close()

    count = 0
    for r in rows:
        venue_str = r["venue"].upper() if r["venue"] else ""
        year = r["year"]
        venue_display = f"{venue_str} {year}" if year else venue_str

        _merge_paper(
            conn,
            paper_id=f"acl:{r['bibkey']}",
            title=r["title"],
            authors=r["authors"],
            abstract=r["abstract"],
            published_at=f"{year}-01-01" if year else None,
            source="acl",
            venue=venue_display,
            web_url=r["web_url"],
        )
        count += 1
    return count


# --- Commands ---

def _rebuild_index():
    """Internal: rebuild unified index from all source caches, preserving citations."""
    conn = get_db()

    existing_citations = {}
    try:
        rows = conn.execute(
            "SELECT paper_id, citation_count, influential_citations FROM papers WHERE citation_count IS NOT NULL"
        ).fetchall()
        for r in rows:
            existing_citations[r["paper_id"]] = (r["citation_count"], r["influential_citations"])
    except Exception:
        pass

    conn.execute("DELETE FROM papers")
    conn.commit()
    now = datetime.now(timezone.utc).isoformat()

    print("Rebuilding unified index...")

    hf_count = _import_hf(conn)
    conn.commit()
    print(f"  HF Papers: {hf_count}")

    arxiv_count = _import_arxiv(conn)
    conn.commit()
    print(f"  arXiv: {arxiv_count}")

    or_count = _import_openreview(conn)
    conn.commit()
    print(f"  OpenReview: {or_count}")

    acl_count = _import_acl(conn)
    conn.commit()
    print(f"  ACL Anthology: {acl_count}")

    for source, count in [("hf", hf_count), ("arxiv", arxiv_count),
                          ("openreview", or_count), ("acl", acl_count)]:
        conn.execute(
            "INSERT OR REPLACE INTO build_log (source, paper_count, built_at) VALUES (?, ?, ?)",
            (source, count, now),
        )
    conn.commit()

    if existing_citations:
        restored = 0
        for paper_id, (cites, inf_cites) in existing_citations.items():
            cur = conn.execute(
                "UPDATE papers SET citation_count = ?, influential_citations = ? WHERE paper_id = ?",
                (cites, inf_cites, paper_id),
            )
            restored += cur.rowcount
        conn.commit()
        print(f"  Restored citation data for {restored} papers")

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    multi = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE sources LIKE '%,%'"
    ).fetchone()[0]
    conn.close()
    _invalidate_lsa()

    print(f"  Total: {total} unique papers ({multi} appear in multiple sources)")


def _get_arxiv_fetch_days(conn):
    """Determine how many days to fetch for arXiv based on last update."""
    last = _get_last_updated(conn, "arxiv")
    if not last:
        return 7
    last_dt = datetime.fromisoformat(last)
    days_since = (datetime.now(timezone.utc) - last_dt).days
    return max(days_since + 1, 1)


def _refresh_hf():
    import subprocess
    print("Refreshing HF papers cache...")
    hf_script = SKILL_DIR / "tools" / "hf_papers_cache.py"
    subprocess.run([sys.executable, str(hf_script), "update"], check=False)


def _refresh_arxiv(conn):
    import subprocess
    days = _get_arxiv_fetch_days(conn)
    print(f"Refreshing arXiv cache (last {days} days)...")
    arxiv_script = SKILL_DIR / "tools" / "arxiv_search.py"
    subprocess.run([sys.executable, str(arxiv_script), "fetch",
                    "--num-days", str(days),
                    "--categories", "cs.AI,cs.CL,cs.CV,cs.LG,cs.NE,stat.ML,cs.IR"],
                   check=False)


def _refresh_openreview():
    import subprocess
    import os
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        print("Refreshing OpenReview: OPENREVIEW_USERNAME/PASSWORD not set, skipping")
        return

    or_script = SKILL_DIR / "tools" / "openreview_search.py"
    conferences = [
        ("ICLR", 2025), ("ICLR", 2026),
        ("ICML", 2025), ("ICML", 2026),
        ("NEURIPS", 2025), ("NEURIPS", 2026),
        ("EMNLP", 2025), ("EMNLP", 2026),
        ("NAACL", 2025), ("NAACL", 2026),
    ]
    print(f"Refreshing OpenReview ({len(conferences)} venue-years)...")
    for conf, year in conferences:
        result = subprocess.run(
            [sys.executable, str(or_script), "fetch", "--conference", conf, "--year", str(year)],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[-1]
            if "Saved 0" not in line:
                print(f"  {line}")
        time.sleep(10)


def _refresh_acl():
    import subprocess
    acl_script = SKILL_DIR / "tools" / "acl_anthology_search.py"
    print("Refreshing ACL Anthology...")
    subprocess.run([sys.executable, str(acl_script), "fetch"], check=False)


def _enrich_new_papers():
    """Enrich only papers that don't have citation data yet."""
    sys.path.insert(0, str(SKILL_DIR / "tools"))
    from citation_counts import get_citation_counts

    conn = get_db()
    rows = conn.execute("""
        SELECT paper_id FROM papers
        WHERE paper_id NOT LIKE 'openreview:%'
          AND paper_id NOT LIKE 'acl:%'
          AND citation_count IS NULL
    """).fetchall()

    arxiv_ids = [r["paper_id"] for r in rows]
    if not arxiv_ids:
        print("  No new papers to enrich.")
        conn.close()
        return

    print(f"  Enriching {len(arxiv_ids)} new papers with citation counts...")
    results = get_citation_counts(arxiv_ids)

    updated = 0
    for aid, info in results.items():
        if info:
            conn.execute(
                "UPDATE papers SET citation_count = ?, influential_citations = ? WHERE paper_id = ?",
                (info["citationCount"], info["influentialCitationCount"], aid),
            )
            updated += 1

    conn.commit()
    conn.close()
    print(f"  Enriched {updated}/{len(arxiv_ids)} papers.")


def cmd_update(args):
    conn = get_db()
    if args.source:
        sources = [args.source]
    else:
        sources = ALL_SOURCES

    for source in sources:
        if source == "hf":
            _refresh_hf()
            _set_last_updated(conn, "hf")
        elif source == "arxiv":
            _refresh_arxiv(conn)
            _set_last_updated(conn, "arxiv")
        elif source == "openreview":
            _refresh_openreview()
            _set_last_updated(conn, "openreview")
        elif source == "acl":
            _refresh_acl()
            _set_last_updated(conn, "acl")
        print()

    conn.close()

    _rebuild_index()
    print()
    _enrich_new_papers()


def cmd_search(args):
    if args.semantic:
        _search_semantic(args.query, args.limit, args.format, args.source, args.since)
    else:
        _search_keyword(args.query, args.limit, args.format, args.source, args.since)


def _build_filters(source, since):
    clauses = []
    params = []
    if source:
        clauses.append("sources LIKE ?")
        params.append(f"%{source}%")
    if since:
        clauses.append("published_at >= ?")
        params.append(since)
    return clauses, params


def _search_keyword(query, limit, fmt, source, since):
    conn = get_db()
    terms = query.split()
    where_parts = [f"(title LIKE ? OR abstract LIKE ?)" for _ in terms]
    params = []
    for t in terms:
        params.extend([f"%{t}%", f"%{t}%"])

    filter_clauses, filter_params = _build_filters(source, since)
    if filter_clauses:
        where_parts.extend(filter_clauses)
        params.extend(filter_params)

    rows = conn.execute(
        f"""SELECT * FROM papers
            WHERE {' AND '.join(where_parts)}
            ORDER BY
                CASE WHEN citation_count IS NOT NULL THEN citation_count ELSE 0 END DESC,
                CASE WHEN hf_upvotes IS NOT NULL THEN hf_upvotes ELSE 0 END DESC,
                published_at DESC
            LIMIT ?""",
        params + [limit],
    ).fetchall()
    conn.close()
    _print_results(rows, fmt)


def _search_semantic(query, limit, fmt, source, since):
    import pickle
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    conn = get_db()
    filter_clauses, filter_params = _build_filters(source, since)
    where = "WHERE " + " AND ".join(filter_clauses) if filter_clauses else ""
    rows = conn.execute(
        f"SELECT * FROM papers {where}",
        filter_params,
    ).fetchall()
    conn.close()

    if not rows:
        print("No papers in index. Run 'update' first.", file=sys.stderr)
        sys.exit(1)

    docs = [f"{r['title']} {r['abstract'] or ''}" for r in rows]

    cache_key = f"{source}:{since}:{len(rows)}"
    model = None
    if LSA_PATH.exists():
        try:
            with open(LSA_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved.get("cache_key") == cache_key:
                model = saved["model"]
                doc_vectors = saved["doc_vectors"]
        except Exception:
            pass

    if model is None:
        n_components = min(200, len(docs) - 1) if len(docs) > 1 else 1
        model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
        ])
        doc_vectors = model.fit_transform(docs)
        with open(LSA_PATH, "wb") as f:
            pickle.dump({"model": model, "doc_vectors": doc_vectors,
                         "cache_key": cache_key}, f)

    query_vec = model.transform([query])
    scores = (doc_vectors @ query_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:limit]

    results = []
    for i in top_idx:
        if scores[i] <= 0:
            break
        results.append((rows[i], scores[i]))

    if fmt == "json":
        _print_json(results, with_score=True)
    else:
        for row, score in results:
            _print_paper(row, score=score)


def cmd_enrich(args):
    """Batch-enrich papers with citation counts from Semantic Scholar."""
    sys.path.insert(0, str(SKILL_DIR / "tools"))
    from citation_counts import get_citation_counts

    conn = get_db()

    if args.paper:
        # Enrich specific paper(s)
        paper_ids = [p.strip() for p in args.paper.split(",")]
        print(f"Enriching {len(paper_ids)} specific papers...")
        results = get_citation_counts(paper_ids)
        updated = 0
        for pid, info in results.items():
            if info:
                conn.execute(
                    "UPDATE papers SET citation_count = ?, influential_citations = ? WHERE paper_id = ?",
                    (info["citationCount"], info["influentialCitationCount"], pid),
                )
                updated += 1
                print(f"  {pid}: {info['citationCount']} citations ({info['influentialCitationCount']} influential)")
            else:
                print(f"  {pid}: not found on Semantic Scholar")
        conn.commit()
        conn.close()
        print(f"Updated {updated}/{len(paper_ids)} papers.")
        return

    # Enrich all papers missing citation data
    source_filter = ""
    if args.source:
        source_filter = f"AND sources LIKE '%{args.source}%'"

    rows = conn.execute(f"""
        SELECT paper_id FROM papers
        WHERE paper_id NOT LIKE 'openreview:%'
          AND paper_id NOT LIKE 'acl:%'
          AND citation_count IS NULL
          {source_filter}
    """).fetchall()

    arxiv_ids = [r["paper_id"] for r in rows]
    if not arxiv_ids:
        print("All papers already enriched (or no arXiv IDs to look up).")
        conn.close()
        return

    print(f"Enriching {len(arxiv_ids)} papers with citation counts...")
    results = get_citation_counts(arxiv_ids)

    updated = 0
    for aid, info in results.items():
        if info:
            conn.execute(
                "UPDATE papers SET citation_count = ?, influential_citations = ? WHERE paper_id = ?",
                (info["citationCount"], info["influentialCitationCount"], aid),
            )
            updated += 1

    conn.commit()
    conn.close()
    print(f"Done. Updated {updated}/{len(arxiv_ids)} papers.")


def cmd_stats(args):
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    if total == 0:
        print("Index is empty. Run 'update' first.")
        conn.close()
        return

    source_filter = args.source

    if source_filter:
        # Stats for a specific source
        cnt = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE sources LIKE ?", (f"%{source_filter}%",)
        ).fetchone()[0]
        with_cites = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE sources LIKE ? AND citation_count IS NOT NULL",
            (f"%{source_filter}%",)
        ).fetchone()[0]

        print(f"Source: {source_filter}")
        print(f"Papers: {cnt}")
        print(f"With citation data: {with_cites}")

        # Last update
        last = _get_last_updated(conn, source_filter)
        if last:
            print(f"Last updated: {last[:19]}")

        # Top papers from this source
        top = conn.execute("""
            SELECT paper_id, title, citation_count, hf_upvotes, venue
            FROM papers WHERE sources LIKE ?
            ORDER BY COALESCE(citation_count, 0) DESC LIMIT 10
        """, (f"%{source_filter}%",)).fetchall()

        if top:
            print(f"\nTop papers:")
            for r in top:
                cites = f"[{r['citation_count']} cites]" if r['citation_count'] else ""
                upvotes = f"↑{r['hf_upvotes']}" if r['hf_upvotes'] else ""
                print(f"  {cites:14s} {upvotes:6s}  {r['title'][:55]}")

        conn.close()
        return

    # Full stats
    multi_source = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE sources LIKE '%,%'"
    ).fetchone()[0]

    by_source = {}
    for src in ALL_SOURCES:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE sources LIKE ?", (f"%{src}%",)
        ).fetchone()[0]
        by_source[src] = cnt

    with_citations = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL"
    ).fetchone()[0]

    top_cited = conn.execute("""
        SELECT paper_id, title, citation_count, influential_citations, sources, venue
        FROM papers WHERE citation_count IS NOT NULL
        ORDER BY citation_count DESC LIMIT 10
    """).fetchall()

    top_upvoted = conn.execute("""
        SELECT paper_id, title, hf_upvotes, sources, venue
        FROM papers WHERE hf_upvotes > 0
        ORDER BY hf_upvotes DESC LIMIT 10
    """).fetchall()

    build_log = conn.execute("SELECT * FROM build_log ORDER BY source").fetchall()
    update_log = conn.execute("SELECT * FROM update_log ORDER BY source").fetchall()
    conn.close()

    print(f"Unified Paper Index: {DB_PATH}")
    print(f"Total papers: {total}")
    print(f"Multi-source papers: {multi_source}")
    print(f"Papers with citation data: {with_citations}\n")

    print("Source      | Papers | Last updated")
    print("------------|--------|--------------------")
    for src in ALL_SOURCES:
        last = next((r["last_updated"][:16] for r in update_log if r["source"] == src), "never")
        print(f"{src:12s}| {by_source[src]:6d} | {last}")

    if top_cited:
        print(f"\nTop cited papers:")
        for r in top_cited:
            print(f"  {r['citation_count']:5d} ({r['influential_citations']:3d} inf)  {r['title'][:55]}  [{r['sources']}]")

    if top_upvoted:
        print(f"\nTop HF upvoted papers:")
        for r in top_upvoted:
            print(f"  ↑{r['hf_upvotes']:4d}  {r['title'][:60]}  [{r['sources']}]")


def _print_results(rows, fmt):
    if fmt == "json":
        _print_json([(r, None) for r in rows], with_score=False)
    else:
        if not rows:
            print("No results found.")
            return
        for r in rows:
            _print_paper(r)


def _print_json(results, with_score=False):
    out = []
    for row, score in results:
        entry = {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "authors": json.loads(row["authors"]) if row["authors"] and row["authors"].startswith("[") else (row["authors"] or "").split(", ")[:5],
            "published_at": row["published_at"],
            "sources": row["sources"].split(","),
            "venue": row["venue"],
            "hf_upvotes": row["hf_upvotes"],
            "citation_count": row["citation_count"],
            "influential_citations": row["influential_citations"],
            "abstract": (row["abstract"] or "")[:300],
            "web_url": row["web_url"],
        }
        if with_score and score is not None:
            entry["score"] = round(float(score), 4)
        out.append(entry)
    print(json.dumps(out, indent=2))


def _print_paper(row, score=None):
    sources = row["sources"]
    venue = row["venue"] or ""
    upvotes = row["hf_upvotes"] or 0
    citations = row["citation_count"]
    score_str = f"  [score: {score:.4f}]" if score is not None else ""
    cite_str = f"  [{citations} cites]" if citations else ""
    upvote_str = f"  ↑{upvotes}" if upvotes else ""

    print(f"\n{row['paper_id']}{score_str}{cite_str}{upvote_str}")
    print(f"  {row['title']}")

    authors_raw = row["authors"] or ""
    if authors_raw.startswith("["):
        authors = json.loads(authors_raw)
    else:
        authors = authors_raw.split(", ")
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" +{len(authors)-3} more"
    print(f"  {author_str}")
    print(f"  {venue}  |  sources: {sources}  |  {(row['published_at'] or '')[:10]}")
    abstract = (row["abstract"] or "").replace("\n", " ")[:200]
    if abstract:
        print(f"  {abstract}...")


def _invalidate_lsa():
    if LSA_PATH.exists():
        LSA_PATH.unlink()


def main():
    parser = argparse.ArgumentParser(
        prog="unified-search",
        description="Consolidated paper search across HF, arXiv, OpenReview, and ACL Anthology",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser("update", help="Refresh source caches, rebuild index, enrich new papers")
    p_update.add_argument("--source", choices=ALL_SOURCES, default=None,
                          help="Refresh only this source (default: all)")

    p_search = sub.add_parser("search", help="Search the unified index")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-e", "--semantic", action="store_true", help="Use LSA semantic search")
    p_search.add_argument("-l", "--limit", type=int, default=20, help="Max results (default: 20)")
    p_search.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p_search.add_argument("--source", choices=ALL_SOURCES, default=None,
                          help="Filter by source (hf, arxiv, openreview, acl)")
    p_search.add_argument("--since", default=None, help="Only papers published after this date (YYYY-MM-DD)")

    p_enrich = sub.add_parser("enrich", help="Enrich papers with Semantic Scholar citation counts")
    p_enrich.add_argument("--paper", default=None,
                          help="Specific arXiv ID(s) to enrich, comma-separated")
    p_enrich.add_argument("--source", choices=ALL_SOURCES, default=None,
                          help="Only enrich papers from this source")

    p_stats = sub.add_parser("stats", help="Show index statistics")
    p_stats.add_argument("--source", choices=ALL_SOURCES, default=None,
                         help="Show stats for only this source")

    args = parser.parse_args()
    if args.command == "update":
        cmd_update(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
