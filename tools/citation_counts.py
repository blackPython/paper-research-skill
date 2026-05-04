#!/usr/bin/env python3
"""
citation-counts: Enrich papers with citation counts from Semantic Scholar.

Supports batch lookup by arXiv ID. Can be used standalone or imported by other tools.

Usage:
    python citation_counts.py lookup 2401.06209 2310.06825 2403.09611
    python citation_counts.py lookup --format json 2401.06209
    echo '2401.06209\\n2310.06825' | python citation_counts.py lookup --stdin
"""

import argparse
import json
import os
import sys
import time

import requests

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "citationCount,influentialCitationCount,title"
BATCH_SIZE = 400


def _get_headers():
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        return {"x-api-key": api_key}
    return {}


def get_citation_counts(arxiv_ids):
    """Look up citation counts for a list of arXiv IDs.

    Returns a dict mapping arxiv_id -> {citationCount, influentialCitationCount, title}
    """
    headers = _get_headers()
    results = {}
    for i in range(0, len(arxiv_ids), BATCH_SIZE):
        batch = arxiv_ids[i : i + BATCH_SIZE]
        s2_ids = [f"ArXiv:{aid}" for aid in batch]

        for attempt in range(3):
            try:
                r = requests.post(
                    S2_BATCH_URL,
                    params={"fields": S2_FIELDS},
                    json={"ids": s2_ids},
                    headers=headers,
                    timeout=30,
                )
                if r.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"Rate limited, waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    print(f"Failed after 3 attempts: {e}", file=sys.stderr)
                    return results
                time.sleep(2)

        for aid, paper in zip(batch, r.json()):
            if paper and isinstance(paper, dict):
                results[aid] = {
                    "citationCount": paper.get("citationCount", 0),
                    "influentialCitationCount": paper.get("influentialCitationCount", 0),
                    "title": paper.get("title"),
                }
            else:
                results[aid] = None

        if i + BATCH_SIZE < len(arxiv_ids):
            time.sleep(1)

    return results


def cmd_lookup(args):
    if args.stdin:
        ids = [line.strip() for line in sys.stdin if line.strip()]
    else:
        ids = args.arxiv_ids

    if not ids:
        print("No arXiv IDs provided.", file=sys.stderr)
        sys.exit(1)

    results = get_citation_counts(ids)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        for aid in ids:
            info = results.get(aid)
            if info:
                print(
                    f"{aid}: {info['citationCount']} citations "
                    f"({info['influentialCitationCount']} influential)  "
                    f"{info['title'][:60]}"
                )
            else:
                print(f"{aid}: not found on Semantic Scholar")


def main():
    parser = argparse.ArgumentParser(
        prog="citation-counts",
        description="Look up citation counts from Semantic Scholar",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_lookup = sub.add_parser("lookup", help="Look up citation counts for arXiv IDs")
    p_lookup.add_argument("arxiv_ids", nargs="*", help="arXiv IDs (e.g., 2401.06209)")
    p_lookup.add_argument("--stdin", action="store_true", help="Read IDs from stdin (one per line)")
    p_lookup.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()
    if args.command == "lookup":
        cmd_lookup(args)


if __name__ == "__main__":
    main()
