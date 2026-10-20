#!/usr/bin/env python3
"""Run real search/retrieval and fail if every result is snippet-only.

    docker exec -i odysseus-odysseus-1 python - < scripts/verify_web_search.py
    ./venv/bin/python scripts/verify_web_search.py --query "latest technology news"
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]) if __file__ != "<stdin>" else "/app")

from src.assistant_preferences import AssistantPreferences, assistant_turn, search_reports
from services.search.core import comprehensive_web_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="latest technology news")
    args = parser.parse_args()
    started = time.monotonic()
    with assistant_turn(AssistantPreferences(search_mode="always")):
        _, sources = comprehensive_web_search(args.query, max_pages=3, return_sources=True)
        reports = search_reports()
    result = {
        "query": args.query, "elapsed_seconds": round(time.monotonic() - started, 2),
        "reports": reports,
        "sources": [{key: source.get(key) for key in (
            "url", "title", "read_status", "partial", "retrieval_method", "failure_reason",
        )} for source in sources],
    }
    print(json.dumps(result, indent=2))
    return 0 if sources and any(source.get("read_status") == "read" for source in sources) else 1


if __name__ == "__main__":
    raise SystemExit(main())
