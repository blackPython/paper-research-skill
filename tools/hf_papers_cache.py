#!/usr/bin/env python3
"""
hf-papers-cache: Local SQLite cache + semantic search for Hugging Face daily papers.

Fetches daily papers from the HF API into a local SQLite database, then provides
keyword and LSA semantic search over the cache.

Usage:
    python hf_papers_cache.py fetch                          # fetch missing days from 2025-01-01 to today
    python hf_papers_cache.py fetch --start 2025-03-01       # fetch from a specific start date
    python hf_papers_cache.py fetch --end 2025-04-01         # fetch up to a specific end date
    python hf_papers_cache.py search "vision transformer" --limit 10
    python hf_papers_cache.py search -e "multimodal reasoning" --limit 10
    python hf_papers_cache.py stats
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DB_DIR = Path.home() / ".local" / "share" / "hf-papers"
DB_PATH = DB_DIR / "papers.db"
LSA_PATH = DB_DIR / "lsa_model.pkl"

DEFAULT_START = date(2025, 1, 1)


def get_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT,
            title TEXT,
            authors TEXT,
            summary TEXT,
            published_at TEXT,
            upvotes INTEGER,
            source TEXT,
            ai_summary TEXT,
            ai_keywords TEXT,
            github_repo TEXT,
            github_stars INTEGER,
            project_page TEXT,
            fetched_date TEXT,
            PRIMARY KEY (arxiv_id, fetched_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            date TEXT PRIMARY KEY,
            paper_count INTEGER,
            fetched_at TEXT
        )
    """)
    conn.commit()
    return conn


def _get_fetched_dates(conn):
    rows = conn.execute("SELECT date FROM fetch_log").fetchall()
    return {r["date"] for r in rows}


def _dates_in_range(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def cmd_fetch(args):
    from huggingface_hub import HfApi

    conn = get_db()
    api = HfApi()

    start = date.fromisoformat(args.start) if args.start else DEFAULT_START
    end = date.fromisoformat(args.end) if args.end else date.today()

    fetched_dates = _get_fetched_dates(conn)
    missing = [d for d in _dates_in_range(start, end) if d.isoformat() not in fetched_dates]

    if not missing:
        print(f"Cache is up to date ({start} to {end}). No dates to fetch.")
        return

    print(f"Fetching {len(missing)} missing days ({missing[0]} to {missing[-1]})...")

    total_papers = 0
    total_days = 0
    errors = 0
    now = datetime.now(timezone.utc).isoformat()

    for i, d in enumerate(missing):
        date_str = d.isoformat()
        try:
            papers = list(api.list_daily_papers(date=date_str, limit=100))
        except Exception as e:
            errors += 1
            if errors > 10:
                print(f"\nToo many errors ({errors}), stopping.", file=sys.stderr)
                break
            print(f"  {date_str}: error ({e})", file=sys.stderr)
            time.sleep(1)
            continue

        for p in papers:
            authors = json.dumps([a.name for a in (p.authors or [])])
            keywords = json.dumps(p.ai_keywords) if p.ai_keywords else None
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO papers
                       (arxiv_id, title, authors, summary, published_at, upvotes,
                        source, ai_summary, ai_keywords, github_repo, github_stars,
                        project_page, fetched_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (p.id, p.title, authors, p.summary,
                     p.published_at.isoformat() if p.published_at else None,
                     p.upvotes, p.source, p.ai_summary, keywords,
                     p.github_repo, p.github_stars, p.project_page, date_str),
                )
            except sqlite3.IntegrityError:
                pass

        conn.execute(
            "INSERT OR REPLACE INTO fetch_log (date, paper_count, fetched_at) VALUES (?, ?, ?)",
            (date_str, len(papers), now),
        )
        conn.commit()

        total_papers += len(papers)
        total_days += 1

        if (i + 1) % 20 == 0:
            print(f"  ...{total_days}/{len(missing)} days fetched ({total_papers} papers)")

        time.sleep(0.3)

    conn.close()
    _invalidate_lsa()
    print(f"Done. Fetched {total_papers} papers across {total_days} days ({errors} errors).")


def cmd_search(args):
    if args.semantic:
        _search_semantic(args.query, args.limit, args.format, args.since)
    else:
        _search_keyword(args.query, args.limit, args.format, args.since)


def _date_filter_clause(since):
    if since:
        return "AND fetched_date >= ?", [since]
    return "", []


def _search_keyword(query, limit, fmt, since):
    conn = get_db()
    terms = query.split()
    where_parts = [f"(title LIKE ? OR summary LIKE ?)" for _ in terms]
    params = []
    for t in terms:
        params.extend([f"%{t}%", f"%{t}%"])

    date_clause, date_params = _date_filter_clause(since)
    rows = conn.execute(
        f"""SELECT DISTINCT arxiv_id, title, authors, summary, published_at, upvotes,
                   ai_keywords, github_repo, fetched_date
            FROM papers
            WHERE {' AND '.join(where_parts)} {date_clause}
            ORDER BY upvotes DESC, published_at DESC
            LIMIT ?""",
        params + date_params + [limit],
    ).fetchall()
    conn.close()
    _print_results(rows, fmt)


def _search_semantic(query, limit, fmt, since):
    import pickle
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    conn = get_db()
    date_clause, date_params = _date_filter_clause(since)
    rows = conn.execute(
        f"""SELECT DISTINCT arxiv_id, title, authors, summary, published_at, upvotes,
                   ai_keywords, github_repo, fetched_date
            FROM papers WHERE 1=1 {date_clause}""",
        date_params,
    ).fetchall()
    conn.close()

    if not rows:
        print("No papers in cache. Run 'fetch' first.", file=sys.stderr)
        sys.exit(1)

    seen = {}
    unique_rows = []
    for r in rows:
        if r["arxiv_id"] not in seen:
            seen[r["arxiv_id"]] = True
            unique_rows.append(r)
    rows = unique_rows

    docs = [f"{r['title']} {r['summary'] or ''}" for r in rows]

    model = None
    if LSA_PATH.exists():
        try:
            with open(LSA_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved["doc_count"] == len(rows) and saved.get("since") == since:
                model = saved["model"]
                doc_vectors = saved["doc_vectors"]
        except Exception:
            pass

    if model is None:
        n_components = min(100, len(docs) - 1) if len(docs) > 1 else 1
        model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
        ])
        doc_vectors = model.fit_transform(docs)
        with open(LSA_PATH, "wb") as f:
            pickle.dump({"model": model, "doc_vectors": doc_vectors,
                         "doc_count": len(rows), "since": since}, f)

    query_vec = model.transform([query])
    scores = (doc_vectors @ query_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:limit]

    results = []
    for i in top_idx:
        if scores[i] <= 0:
            break
        results.append((rows[i], scores[i]))

    if fmt == "json":
        out = []
        for row, score in results:
            out.append({
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "authors": json.loads(row["authors"]) if row["authors"] else [],
                "published_at": row["published_at"],
                "upvotes": row["upvotes"],
                "score": round(float(score), 4),
                "summary": (row["summary"] or "")[:300],
                "ai_keywords": json.loads(row["ai_keywords"]) if row["ai_keywords"] else [],
                "github_repo": row["github_repo"],
            })
        print(json.dumps(out, indent=2))
    else:
        for row, score in results:
            _print_paper(row, score=score)


def _print_results(rows, fmt):
    if fmt == "json":
        out = []
        for r in rows:
            out.append({
                "arxiv_id": r["arxiv_id"],
                "title": r["title"],
                "authors": json.loads(r["authors"]) if r["authors"] else [],
                "published_at": r["published_at"],
                "upvotes": r["upvotes"],
                "summary": (r["summary"] or "")[:300],
                "ai_keywords": json.loads(r["ai_keywords"]) if r["ai_keywords"] else [],
                "github_repo": r["github_repo"],
            })
        print(json.dumps(out, indent=2))
    else:
        if not rows:
            print("No results found.")
            return
        for r in rows:
            _print_paper(r)


def _print_paper(row, score=None):
    score_str = f"  [score: {score:.4f}]" if score is not None else ""
    upvotes = row["upvotes"] or 0
    print(f"\n{row['arxiv_id']}  ↑{upvotes}{score_str}")
    print(f"  {row['title']}")
    authors = json.loads(row["authors"]) if row["authors"] else []
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" +{len(authors)-3} more"
    print(f"  {author_str}")
    print(f"  {(row['published_at'] or '')[:10]}")
    summary = (row["summary"] or "").replace("\n", " ")[:200]
    print(f"  {summary}...")


def _invalidate_lsa():
    if LSA_PATH.exists():
        LSA_PATH.unlink()


def cmd_stats(args):
    conn = get_db()
    total = conn.execute("SELECT COUNT(DISTINCT arxiv_id) FROM papers").fetchone()[0]
    total_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    fetched_days = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]

    if total == 0:
        print("Cache is empty. Run 'fetch' first.")
        conn.close()
        return

    date_range = conn.execute(
        "SELECT MIN(date) as min_d, MAX(date) as max_d FROM fetch_log"
    ).fetchone()

    missing_days = conn.execute(
        "SELECT COUNT(*) FROM fetch_log WHERE paper_count = 0"
    ).fetchone()[0]

    monthly = conn.execute("""
        SELECT SUBSTR(fetched_date, 1, 7) as month, COUNT(DISTINCT arxiv_id) as cnt
        FROM papers GROUP BY month ORDER BY month DESC LIMIT 12
    """).fetchall()

    top_upvoted = conn.execute("""
        SELECT arxiv_id, title, upvotes FROM papers
        ORDER BY upvotes DESC LIMIT 5
    """).fetchall()

    conn.close()

    print(f"HF Papers Cache: {DB_PATH}")
    print(f"Unique papers: {total}")
    print(f"Total rows: {total_rows}")
    print(f"Fetched days: {fetched_days} ({date_range['min_d']} to {date_range['max_d']})")
    print(f"Days with 0 papers: {missing_days} (weekends/holidays)\n")

    print("Month       | Papers")
    print("------------|-------")
    for r in monthly:
        print(f"{r['month']}     | {r['cnt']}")

    print(f"\nTop upvoted papers:")
    for r in top_upvoted:
        print(f"  ↑{r['upvotes']:4d}  {r['arxiv_id']}  {r['title'][:60]}")


def cmd_update(args):
    """Convenience: fetch only missing days from last fetched date to today."""
    conn = get_db()
    last = conn.execute("SELECT MAX(date) as d FROM fetch_log").fetchone()["d"]
    conn.close()

    if last:
        args.start = last
    else:
        args.start = DEFAULT_START.isoformat()
    args.end = None
    cmd_fetch(args)


def main():
    parser = argparse.ArgumentParser(
        prog="hf-papers-cache",
        description="Local cache + search for Hugging Face daily papers",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch daily papers from HF API into local cache")
    p_fetch.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 2025-01-01)")
    p_fetch.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")

    p_update = sub.add_parser("update", help="Fetch only new days since last fetch")

    p_search = sub.add_parser("search", help="Search cached HF papers")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-e", "--semantic", action="store_true", help="Use LSA semantic search")
    p_search.add_argument("-l", "--limit", type=int, default=10, help="Max results (default: 10)")
    p_search.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p_search.add_argument("--since", default=None, help="Only search papers from this date onward (YYYY-MM-DD)")

    sub.add_parser("stats", help="Show cache statistics")

    args = parser.parse_args()
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "update":
        cmd_update(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
