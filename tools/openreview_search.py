#!/usr/bin/env python3
"""
openreview-search: Search peer-reviewed papers from top ML/AI conferences via OpenReview.

Uses cached JSON files from ~/.openreview/ and can fetch fresh data via the OpenReview API.

Usage:
    python openreview_search.py search "memory agent" --limit 10
    python openreview_search.py search "memory agent" --conferences ICLR,ICML --years 2025
    python openreview_search.py search -e "memory mechanisms for LLM agents" --limit 10
    python openreview_search.py fetch --conference ICLR --year 2025
    python openreview_search.py stats
"""

import argparse
import json
import os
import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".openreview"

CONFERENCE_VENUES = {
    "ICLR": "ICLR.cc/{year}/Conference",
    "ICML": "ICML.cc/{year}/Conference",
    "NEURIPS": "NeurIPS.cc/{year}/Conference",
    "EMNLP": "EMNLP/{year}/Conference",
    "NAACL": "aclweb.org/NAACL/{year}/Conference",
    "ACL": "aclweb.org/ACL/{year}/Conference",
}

DEFAULT_CONFERENCES = list(CONFERENCE_VENUES.keys())
DEFAULT_YEARS = [2024, 2025]


def _load_papers(conferences=None, years=None):
    conferences = [c.upper() for c in (conferences or DEFAULT_CONFERENCES)]
    years = years or DEFAULT_YEARS
    all_papers = []
    for conf in conferences:
        for year in years:
            path = CACHE_DIR / f"{conf}_{year}_papers.json"
            if path.exists() and path.stat().st_size > 2:
                with open(path) as f:
                    papers = json.load(f)
                all_papers.extend(papers)
    return all_papers


def cmd_search(args):
    papers = _load_papers(
        conferences=args.conferences.split(",") if args.conferences else None,
        years=[int(y) for y in args.years.split(",")] if args.years else None,
    )
    if not papers:
        print("No papers in cache. Run 'fetch' first or check ~/.openreview/", file=sys.stderr)
        sys.exit(1)

    if args.semantic:
        _search_semantic(papers, args.query, args.limit, args.format)
    else:
        _search_keyword(papers, args.query, args.limit, args.format)


def _search_keyword(papers, query, limit, fmt):
    terms = query.lower().split()
    results = []
    for p in papers:
        text = f"{p.get('title', '')} {p.get('abstract', '')} {' '.join(p.get('keywords', []))}".lower()
        if all(t in text for t in terms):
            results.append(p)
    results = results[:limit]
    _print_results(results, fmt)


def _search_semantic(papers, query, limit, fmt):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    import numpy as np

    docs = [f"{p.get('title', '')} {p.get('abstract', '')} {' '.join(p.get('keywords') or [])}" for p in papers]

    n_components = min(100, len(docs) - 1) if len(docs) > 1 else 1
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
        ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
    ])
    doc_vectors = model.fit_transform(docs)
    query_vec = model.transform([query])
    scores = (doc_vectors @ query_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:limit]

    results = []
    for i in top_idx:
        if scores[i] <= 0:
            break
        p = dict(papers[i])
        p["score"] = round(float(scores[i]), 4)
        results.append(p)
    _print_results(results, fmt)


def _print_results(results, fmt):
    if fmt == "json":
        out = []
        for p in results:
            out.append({
                "paper_id": p.get("paper_id"),
                "title": p.get("title"),
                "authors": p.get("authors", [])[:5],
                "venue": p.get("venue"),
                "year": p.get("year"),
                "keywords": p.get("keywords", []),
                "abstract": p.get("abstract", "")[:300],
                "score": p.get("score"),
            })
        print(json.dumps(out, indent=2))
    else:
        if not results:
            print("No results found.")
            return
        for p in results:
            score = f"  [score: {p['score']:.4f}]" if "score" in p else ""
            print(f"\n{p.get('venue', '?')} {p.get('year', '?')}{score}")
            print(f"  {p.get('title', 'No title')}")
            authors = p.get("authors", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" +{len(authors)-3} more"
            print(f"  {author_str}")
            kw = p.get("keywords", [])
            if kw:
                print(f"  Keywords: {', '.join(kw[:5])}")


def cmd_fetch(args):
    try:
        import openreview
    except ImportError:
        print("openreview-py not installed. Run: pip install openreview-py", file=sys.stderr)
        sys.exit(1)

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        print("Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD env vars.", file=sys.stderr)
        sys.exit(1)

    conf = args.conference.upper()
    year = args.year
    if conf not in CONFERENCE_VENUES:
        print(f"Unknown conference: {conf}. Supported: {', '.join(CONFERENCE_VENUES.keys())}", file=sys.stderr)
        sys.exit(1)

    venue_id = CONFERENCE_VENUES[conf].format(year=year)
    print(f"Fetching {conf} {year} (venue: {venue_id})...")

    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=username,
                password=password,
            )
            break
        except Exception as e:
            if "429" in str(e) or "RateLimit" in str(e):
                wait = 60 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    else:
        print("Failed to connect after retries.", file=sys.stderr)
        sys.exit(1)

    notes = client.get_all_notes(content={"venueid": venue_id})
    venue_str = f"{conf} {year}" if conf != "NEURIPS" else f"NeurIPS {year}"
    alt_notes = client.get_all_notes(content={"venue": venue_str})
    if len(alt_notes) > len(notes):
        notes = alt_notes
    records = []
    for note in notes:
        content = note.content
        records.append({
            "paper_id": note.id,
            "paper_number": note.number,
            "title": content.get("title", {}).get("value"),
            "abstract": content.get("abstract", {}).get("value"),
            "authors": content.get("authors", {}).get("value"),
            "author_ids": content.get("authorids", {}).get("value"),
            "keywords": content.get("keywords", {}).get("value"),
            "venue": venue_id,
            "year": year,
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{conf}_{year}_papers.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} papers to {path}")


def cmd_stats(args):
    if not CACHE_DIR.exists():
        print("No cache directory found. Run 'fetch' first.")
        return
    print(f"Cache: {CACHE_DIR}\n")
    total = 0
    for path in sorted(CACHE_DIR.glob("*.json")):
        try:
            with open(path) as f:
                papers = json.load(f)
            count = len(papers)
            total += count
            print(f"  {path.stem}: {count} papers")
        except Exception:
            print(f"  {path.stem}: (error reading)")
    print(f"\nTotal: {total} papers")


def main():
    parser = argparse.ArgumentParser(prog="openreview-search", description="Search peer-reviewed ML/AI conference papers via OpenReview")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Search cached conference papers")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-e", "--semantic", action="store_true", help="Use LSA semantic search")
    p_search.add_argument("-l", "--limit", type=int, default=10, help="Max results (default: 10)")
    p_search.add_argument("--conferences", default=None, help="Comma-separated conferences (default: all)")
    p_search.add_argument("--years", default=None, help="Comma-separated years (default: 2024,2025)")
    p_search.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    p_fetch = sub.add_parser("fetch", help="Fetch papers from OpenReview API into cache")
    p_fetch.add_argument("--conference", required=True, help="Conference name (ICLR, ICML, NeurIPS, etc.)")
    p_fetch.add_argument("--year", type=int, required=True, help="Conference year")

    sub.add_parser("stats", help="Show cache statistics")

    args = parser.parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
