#!/usr/bin/env python3
"""
arxiv-search: Local SQLite cache + LSA semantic search for arXiv papers.
Drop-in replacement for arxivterminal using arxiv v3 API.

Usage:
    python arxiv_search.py fetch --num-days 7 --categories cs.AI,cs.CL,cs.LG
    python arxiv_search.py search "memory agent LLM" --limit 10
    python arxiv_search.py search -e "memory mechanisms for agents" --limit 10
    python arxiv_search.py search-api "memory tool LLM agent" --limit 20
    python arxiv_search.py stats
"""

import argparse
import json
import sqlite3
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv

DB_DIR = Path.home() / ".local" / "share" / "arxiv-search"
DB_PATH = DB_DIR / "papers.db"
LSA_PATH = DB_DIR / "lsa_model.pkl"


def get_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            published TEXT,
            updated TEXT,
            categories TEXT,
            primary_category TEXT,
            url TEXT,
            fetched_at TEXT
        )
    """)
    conn.commit()
    return conn


def cmd_fetch(args):
    conn = get_db()
    since = datetime.now(timezone.utc) - timedelta(days=args.num_days)
    categories = [c.strip() for c in args.categories.split(",")]
    cat_query = " OR ".join(f"cat:{c}" for c in categories)

    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=5)
    search = arxiv.Search(
        query=cat_query,
        max_results=args.max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    count = 0
    skipped = 0
    now = datetime.now(timezone.utc).isoformat()
    for result in client.results(search):
        if result.published.replace(tzinfo=timezone.utc) < since.replace(tzinfo=timezone.utc):
            break
        aid = result.entry_id.split("/abs/")[-1].split("v")[0]
        authors = ", ".join(a.name for a in result.authors)
        cats = ", ".join(c for c in result.categories)
        try:
            conn.execute(
                """INSERT INTO papers (arxiv_id, title, authors, abstract, published, updated, categories, primary_category, url, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (aid, result.title, authors, result.summary, result.published.isoformat(),
                 result.updated.isoformat(), cats, result.primary_category, str(result.entry_id), now),
            )
            count += 1
        except sqlite3.IntegrityError:
            skipped += 1
        if count % 100 == 0 and count > 0:
            conn.commit()
            print(f"  ...{count} papers inserted", file=sys.stderr)
    conn.commit()
    conn.close()
    print(f"Fetched {count} new papers ({skipped} already in DB)")
    _invalidate_lsa()


def cmd_search(args):
    if args.semantic:
        _search_semantic(args.query, args.limit, args.force_retrain, args.format)
    else:
        _search_keyword(args.query, args.limit, args.format)


def _search_keyword(query, limit, fmt):
    conn = get_db()
    terms = query.split()
    where = " AND ".join(f"(title LIKE ? OR abstract LIKE ?)" for _ in terms)
    params = []
    for t in terms:
        params.extend([f"%{t}%", f"%{t}%"])
    rows = conn.execute(
        f"SELECT * FROM papers WHERE {where} ORDER BY published DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()
    _print_results(rows, fmt)


def _search_semantic(query, limit, force_retrain, fmt):
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    import numpy as np

    conn = get_db()
    rows = conn.execute("SELECT arxiv_id, title, abstract, authors, published, url FROM papers").fetchall()
    conn.close()

    if not rows:
        print("No papers in database. Run 'fetch' first.", file=sys.stderr)
        sys.exit(1)

    docs = [f"{r['title']} {r['abstract']}" for r in rows]

    model = None
    if LSA_PATH.exists() and not force_retrain:
        try:
            with open(LSA_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved["doc_count"] == len(rows):
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
            pickle.dump({"model": model, "doc_vectors": doc_vectors, "doc_count": len(rows)}, f)

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
                "authors": row["authors"],
                "published": row["published"],
                "score": round(float(score), 4),
                "abstract": row["abstract"],
                "url": row["url"],
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
                "authors": r["authors"],
                "published": r["published"],
                "abstract": r["abstract"],
                "url": r["url"],
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
    print(f"\n{row['arxiv_id']}{score_str}")
    print(f"  {row['title']}")
    print(f"  {row['authors'][:80]}{'...' if len(row['authors']) > 80 else ''}")
    print(f"  {row['published'][:10]}")
    abstract = row["abstract"].replace("\n", " ")[:200]
    print(f"  {abstract}...")


def _invalidate_lsa():
    if LSA_PATH.exists():
        LSA_PATH.unlink()


def cmd_search_api(args):
    query = args.query
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        cat_filter = " OR ".join(f"cat:{c}" for c in cats)
        query = f"({query}) AND ({cat_filter})"

    sort = arxiv.SortCriterion.Relevance if args.sort == "relevance" else arxiv.SortCriterion.SubmittedDate
    client = arxiv.Client(page_size=args.limit, delay_seconds=5, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=args.limit,
        sort_by=sort,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for result in client.results(search):
        aid = result.entry_id.split("/abs/")[-1].split("v")[0]
        authors = ", ".join(a.name for a in result.authors)
        results.append({
            "arxiv_id": aid,
            "title": result.title,
            "authors": authors,
            "published": result.published.isoformat(),
            "abstract": result.summary,
            "url": str(result.entry_id),
        })

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("No results found.")
            return
        for r in results:
            print(f"\n{r['arxiv_id']}")
            print(f"  {r['title']}")
            print(f"  {r['authors'][:80]}{'...' if len(r['authors']) > 80 else ''}")
            print(f"  {r['published'][:10]}")
            abstract = r["abstract"].replace("\n", " ")[:200]
            print(f"  {abstract}...")


def cmd_stats(args):
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    if total == 0:
        print("Database is empty. Run 'fetch' first.")
        conn.close()
        return
    date_counts = conn.execute(
        "SELECT SUBSTR(published, 1, 10) as date, COUNT(*) as cnt FROM papers GROUP BY date ORDER BY date DESC LIMIT 15"
    ).fetchall()
    cats = conn.execute(
        "SELECT primary_category, COUNT(*) as cnt FROM papers GROUP BY primary_category ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    conn.close()

    print(f"Total papers: {total}")
    print(f"DB path: {DB_PATH}\n")
    print("Date       | Count")
    print("-----------|------")
    for r in date_counts:
        print(f"{r['date']}  | {r['cnt']}")
    print(f"\nTop categories:")
    for r in cats:
        print(f"  {r['primary_category']}: {r['cnt']}")


def main():
    parser = argparse.ArgumentParser(prog="arxiv-search", description="arXiv paper search with SQLite cache and LSA")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch papers from arXiv API into local DB")
    p_fetch.add_argument("--num-days", type=int, default=7, help="Days of papers to fetch (default: 7)")
    p_fetch.add_argument("--categories", default="cs.AI,cs.CL,cs.CV,cs.LG,cs.NE,stat.ML,cs.IR", help="Comma-separated arXiv categories")
    p_fetch.add_argument("--max-results", type=int, default=2000, help="Max papers to fetch (default: 2000)")

    p_search = sub.add_parser("search", help="Search papers in local DB")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-e", "--semantic", action="store_true", help="Use LSA semantic search")
    p_search.add_argument("-f", "--force-retrain", action="store_true", help="Force retrain LSA model")
    p_search.add_argument("-l", "--limit", type=int, default=10, help="Max results (default: 10)")
    p_search.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    p_search_api = sub.add_parser("search-api", help="Search arXiv API directly (no local DB needed)")
    p_search_api.add_argument("query", help="Search query (arXiv query syntax supported)")
    p_search_api.add_argument("-l", "--limit", type=int, default=20, help="Max results (default: 20)")
    p_search_api.add_argument("--sort", choices=["relevance", "date"], default="relevance", help="Sort order (default: relevance)")
    p_search_api.add_argument("--categories", default=None, help="Comma-separated arXiv categories to filter")
    p_search_api.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    sub.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "search-api":
        cmd_search_api(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
