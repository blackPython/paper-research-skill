#!/usr/bin/env python3
"""
acl-anthology-search: Local cache + semantic search for ACL Anthology papers.

Caches papers from the ACL Anthology (ACL, EMNLP, NAACL, EACL, Findings, etc.)
into a local SQLite database, then provides keyword and LSA semantic search.

Usage:
    python acl_anthology_search.py fetch                           # fetch all supported venues (2024-2025)
    python acl_anthology_search.py fetch --collections 2024.emnlp,2025.acl
    python acl_anthology_search.py search "vision transformer" --limit 10
    python acl_anthology_search.py search -e "multimodal reasoning" --limit 10
    python acl_anthology_search.py stats
"""

import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path

DB_DIR = Path.home() / ".local" / "share" / "acl-anthology"
DB_PATH = DB_DIR / "papers.db"
LSA_PATH = DB_DIR / "lsa_model.pkl"

DEFAULT_COLLECTIONS = [
    "2024.acl", "2025.acl",
    "2024.emnlp", "2025.emnlp",
    "2024.naacl", "2025.naacl",
    "2024.eacl",
    "2024.findings", "2025.findings",
]


def get_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            bibkey TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            year INTEGER,
            venue TEXT,
            collection_id TEXT,
            web_url TEXT,
            pdf_name TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            collection_id TEXT PRIMARY KEY,
            paper_count INTEGER,
            fetched_at TEXT
        )
    """)
    conn.commit()
    return conn


def cmd_fetch(args):
    from acl_anthology import Anthology
    from datetime import datetime, timezone

    warnings.filterwarnings("ignore")
    print("Loading ACL Anthology data...")
    anth = Anthology.from_repo(verbose=False)
    conn = get_db()

    collections = args.collections.split(",") if args.collections else DEFAULT_COLLECTIONS
    now = datetime.now(timezone.utc).isoformat()
    total = 0

    for cid in collections:
        try:
            col = anth.get_collection(cid)
            if col is None:
                print(f"  {cid}: not found, skipping")
                continue
        except Exception as e:
            print(f"  {cid}: error ({e}), skipping")
            continue

        count = 0
        for vol in col.volumes():
            for paper in vol.papers():
                if paper.is_frontmatter:
                    continue
                authors = json.dumps([
                    f"{a.name.first} {a.name.last}" for a in paper.authors
                ])
                abstract = str(paper.abstract) if paper.abstract else None
                pdf_name = paper.pdf.name if paper.pdf else None
                venues = ",".join(paper.venue_ids) if paper.venue_ids else cid.split(".")[1]
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO papers
                           (bibkey, title, authors, abstract, year, venue, collection_id, web_url, pdf_name)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (paper.bibkey, str(paper.title), authors, abstract,
                         paper.year, venues, cid, paper.web_url, pdf_name),
                    )
                    count += 1
                except Exception:
                    pass

        conn.execute(
            "INSERT OR REPLACE INTO fetch_log (collection_id, paper_count, fetched_at) VALUES (?, ?, ?)",
            (cid, count, now),
        )
        conn.commit()
        total += count
        print(f"  {cid}: {count} papers")

    conn.close()
    _invalidate_lsa()
    print(f"Done. Cached {total} papers from {len(collections)} collections.")


def cmd_search(args):
    if args.semantic:
        _search_semantic(args.query, args.limit, args.format, args.venue)
    else:
        _search_keyword(args.query, args.limit, args.format, args.venue)


def _venue_filter(venue):
    if venue:
        return "AND venue LIKE ?", [f"%{venue}%"]
    return "", []


def _search_keyword(query, limit, fmt, venue):
    conn = get_db()
    terms = query.split()
    where_parts = [f"(title LIKE ? OR abstract LIKE ?)" for _ in terms]
    params = []
    for t in terms:
        params.extend([f"%{t}%", f"%{t}%"])

    venue_clause, venue_params = _venue_filter(venue)
    rows = conn.execute(
        f"""SELECT * FROM papers
            WHERE {' AND '.join(where_parts)} {venue_clause}
            ORDER BY year DESC
            LIMIT ?""",
        params + venue_params + [limit],
    ).fetchall()
    conn.close()
    _print_results(rows, fmt)


def _search_semantic(query, limit, fmt, venue):
    import pickle
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    conn = get_db()
    venue_clause, venue_params = _venue_filter(venue)
    rows = conn.execute(
        f"SELECT * FROM papers WHERE 1=1 {venue_clause}",
        venue_params,
    ).fetchall()
    conn.close()

    if not rows:
        print("No papers in cache. Run 'fetch' first.", file=sys.stderr)
        sys.exit(1)

    docs = [f"{r['title']} {r['abstract'] or ''}" for r in rows]

    model = None
    if LSA_PATH.exists():
        try:
            with open(LSA_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved["doc_count"] == len(rows) and saved.get("venue") == venue:
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
                         "doc_count": len(rows), "venue": venue}, f)

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
                "bibkey": row["bibkey"],
                "title": row["title"],
                "authors": json.loads(row["authors"]) if row["authors"] else [],
                "year": row["year"],
                "venue": row["venue"],
                "score": round(float(score), 4),
                "abstract": (row["abstract"] or "")[:300],
                "web_url": row["web_url"],
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
                "bibkey": r["bibkey"],
                "title": r["title"],
                "authors": json.loads(r["authors"]) if r["authors"] else [],
                "year": r["year"],
                "venue": r["venue"],
                "abstract": (r["abstract"] or "")[:300],
                "web_url": r["web_url"],
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
    print(f"\n{row['venue'].upper()} {row['year']}{score_str}")
    print(f"  {row['title']}")
    authors = json.loads(row["authors"]) if row["authors"] else []
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" +{len(authors)-3} more"
    print(f"  {author_str}")
    print(f"  {row['web_url']}")
    abstract = (row["abstract"] or "").replace("\n", " ")[:200]
    if abstract:
        print(f"  {abstract}...")


def _invalidate_lsa():
    if LSA_PATH.exists():
        LSA_PATH.unlink()


def cmd_stats(args):
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    if total == 0:
        print("Cache is empty. Run 'fetch' first.")
        conn.close()
        return

    by_collection = conn.execute("""
        SELECT collection_id, COUNT(*) as cnt FROM papers
        GROUP BY collection_id ORDER BY collection_id
    """).fetchall()

    by_venue = conn.execute("""
        SELECT venue, COUNT(*) as cnt FROM papers
        GROUP BY venue ORDER BY cnt DESC LIMIT 15
    """).fetchall()

    conn.close()

    print(f"ACL Anthology Cache: {DB_PATH}")
    print(f"Total papers: {total}\n")

    print("Collection       | Papers")
    print("-----------------|-------")
    for r in by_collection:
        print(f"{r['collection_id']:17s}| {r['cnt']}")

    print(f"\nBy venue:")
    for r in by_venue:
        print(f"  {r['venue']}: {r['cnt']}")


def main():
    parser = argparse.ArgumentParser(
        prog="acl-anthology-search",
        description="Local cache + search for ACL Anthology papers",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch papers from ACL Anthology into local cache")
    p_fetch.add_argument("--collections", default=None,
                         help="Comma-separated collection IDs (default: ACL/EMNLP/NAACL/EACL/Findings 2024-2025)")

    p_search = sub.add_parser("search", help="Search cached papers")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-e", "--semantic", action="store_true", help="Use LSA semantic search")
    p_search.add_argument("-l", "--limit", type=int, default=10, help="Max results (default: 10)")
    p_search.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p_search.add_argument("--venue", default=None, help="Filter by venue (e.g., emnlp, acl, naacl)")

    sub.add_parser("stats", help="Show cache statistics")

    args = parser.parse_args()
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
