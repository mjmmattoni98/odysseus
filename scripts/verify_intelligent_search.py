#!/usr/bin/env python3
"""Check d1 search routing against a real local Ollama. No web searches run.

    ./venv/bin/python scripts/verify_intelligent_search.py --url http://localhost:11434
"""

import argparse
import asyncio
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import ChatMessage
from src.assistant_preferences import AssistantPreferences
from src.intelligent_search import decide_search_for_turn


CASES = [
    ("Hello!", False, []),
    ("Explain why the sky is blue.", False, []),
    ("Translate into Spanish: The meeting starts at noon.", False, []),
    ("Write a Python function to reverse a list.", False, []),
    ("Check my calendar for today.", False, []),
    ("Search my past chats for the deployment fix.", False, []),
    ("¿Qué tiempo hace hoy en Madrid?", True, []),
    ("What are the latest developments in fusion energy?", True, []),
    ("Find the official documentation for the latest Ollama release.", True, []),
    ("Recommend a laptop I can buy for 1000 euros.", True, []),
    ("And in Barcelona?", True, [ChatMessage("user", "What is the weather in Madrid today?")]),
    ("Make that answer shorter.", False, [ChatMessage("user", "What is the weather in Madrid today?"), ChatMessage("assistant", "Madrid is sunny today, around 22 degrees.")]),
]


async def run(url):
    preferences = AssistantPreferences(search_mode="auto")
    failed = 0
    for message, expected, history in CASES:
        session = SimpleNamespace(endpoint_url=url, headers={}, history=history)
        started = time.monotonic()
        decision = await decide_search_for_turn(preferences, session, message)
        passed = decision is not None and decision.needs_search is expected
        failed += not passed
        probability = decision.probability if decision is not None else None
        print(f"{'PASS' if passed else 'FAIL'} search={getattr(decision, 'needs_search', None)} "
              f"p={probability} {time.monotonic() - started:.2f}s {message}")
    # Saved explicit modes bypass d1 even if its endpoint is unavailable.
    session = SimpleNamespace(endpoint_url="", headers={}, history=[])
    for mode, expected in [("off", False), ("always", True)]:
        decision = await decide_search_for_turn(AssistantPreferences(search_mode=mode), session, "Hello!")
        passed = decision is not None and decision.needs_search is expected
        failed += not passed
        print(f"{'PASS' if passed else 'FAIL'} saved Web: {mode}, search={getattr(decision, 'needs_search', None)}")
    print(f"{len(CASES) + 2 - failed}/{len(CASES) + 2} passed")
    return bool(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Local Ollama URL with d1:3b installed")
    sys.exit(asyncio.run(run(parser.parse_args().url)))
