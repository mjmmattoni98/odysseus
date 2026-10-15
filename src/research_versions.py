# src/research_versions.py
"""Saved-research versions and refresh change summaries.

A refresh re-runs a saved question with the same settings and source rules
and saves the result as a NEW research file; the previous file is never
modified. Versions of one question share a ``lineage_id`` (the first
version's id) and carry ``version`` and ``previous_version_id``.

The change summary compares the two versions' sources by canonical URL and
page-content fingerprint (``content_hash``) and adds a short LLM-written note
on how the conclusions moved.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.research_sources import canonical_url
from src.research_utils import strip_thinking

logger = logging.getLogger(__name__)

# Previous sources re-read first by a refresh so their content can be compared.
MAX_REFRESH_SEED_URLS = 6
_REPORT_EXCERPT_CHARS = 6000
_LIST_LIMIT = 12

CHANGE_SUMMARY_INSTRUCTIONS = """\
You compare two versions of a research answer to the same question and \
explain what changed. The previous and new answers follow as separate \
messages.

**Question:** {question}

**Evidence changes:** {added} source(s) added, {removed} removed, \
{changed} with changed page content, {unchanged} unchanged.

Write 2-5 short bullet points on how the CONCLUSIONS changed: new facts, \
conclusions that were reversed, strengthened or weakened, and claims that \
were dropped. If the conclusions are materially the same, say that in one \
sentence instead. Do not restate the whole answer and do not add new claims."""


def lineage_id(session_id: str, data: Optional[Dict[str, Any]]) -> str:
    data = data if isinstance(data, dict) else {}
    return str(data.get("lineage_id") or session_id)


def version_number(data: Optional[Dict[str, Any]]) -> int:
    try:
        return max(1, int((data or {}).get("version") or 1))
    except (TypeError, ValueError):
        return 1


def _source_entry(source: Dict[str, Any]) -> Dict[str, str]:
    return {
        "url": str(source.get("url") or ""),
        "title": str(source.get("title") or source.get("url") or ""),
    }


def compare_sources(previous: Iterable[Any], current: Iterable[Any]) -> Dict[str, List[Dict[str, str]]]:
    """Diff two source lists by canonical URL and content fingerprint.

    ``unverified`` holds sources present in both versions where at least one
    side has no fingerprint (reports saved before fingerprints existed), so
    the comparison never claims "unchanged" without evidence.
    """
    prev = {}
    for source in previous or []:
        if isinstance(source, dict) and source.get("url"):
            prev.setdefault(canonical_url(source["url"]), source)
    cur = {}
    for source in current or []:
        if isinstance(source, dict) and source.get("url"):
            cur.setdefault(canonical_url(source["url"]), source)
    diff: Dict[str, List[Dict[str, str]]] = {
        "added": [], "removed": [], "changed": [], "unchanged": [], "unverified": [],
    }
    for key, source in cur.items():
        if key not in prev:
            diff["added"].append(_source_entry(source))
            continue
        old_hash = str(prev[key].get("content_hash") or "")
        new_hash = str(source.get("content_hash") or "")
        if not old_hash or not new_hash:
            diff["unverified"].append(_source_entry(source))
        elif old_hash != new_hash:
            diff["changed"].append(_source_entry(source))
        else:
            diff["unchanged"].append(_source_entry(source))
    for key, source in prev.items():
        if key not in cur:
            diff["removed"].append(_source_entry(source))
    return diff


def _excerpt(text: Any) -> str:
    text = strip_thinking(str(text or "")) or ""
    if len(text) <= _REPORT_EXCERPT_CHARS:
        return text
    return text[:_REPORT_EXCERPT_CHARS].rsplit("\n", 1)[0] + "\n[...]"


async def summarize_conclusion_changes(
    question: str,
    previous_report: str,
    new_report: str,
    diff: Dict[str, List[Dict[str, str]]],
    *,
    llm_endpoint: str,
    llm_model: str,
    llm_headers: Optional[dict] = None,
    timeout: int = 90,
) -> str:
    """Ask the research model how the conclusions changed; ``""`` on failure."""
    if not previous_report or not new_report:
        return ""
    from src.llm_core import llm_call_async
    from src.prompt_security import untrusted_context_message

    instructions = CHANGE_SUMMARY_INSTRUCTIONS.format(
        question=question,
        added=len(diff.get("added", [])),
        removed=len(diff.get("removed", [])),
        changed=len(diff.get("changed", [])),
        unchanged=len(diff.get("unchanged", [])),
    )
    try:
        response = await llm_call_async(
            url=llm_endpoint,
            model=llm_model,
            messages=[
                {"role": "user", "content": instructions},
                untrusted_context_message("previous research answer", _excerpt(previous_report)),
                untrusted_context_message("new research answer", _excerpt(new_report)),
            ],
            temperature=0.2,
            max_tokens=1024,
            headers=llm_headers,
            timeout=timeout,
            max_retries=1,
            workload="research",
        )
    except Exception as exc:  # noqa: BLE001 - the deterministic diff still stands
        logger.warning("Research change summary failed: %s", exc)
        return ""
    return (strip_thinking(response) or "").strip()


def build_change_summary(
    previous_version: Dict[str, Any],
    current_sources: Iterable[Any],
    *,
    conclusion_summary: str = "",
) -> Dict[str, Any]:
    """Persistable change summary for a refreshed version."""
    diff = compare_sources(previous_version.get("sources") or [], current_sources)
    return {
        "previous_version_id": str(previous_version.get("id") or ""),
        "previous_version": version_number(previous_version),
        "counts": {key: len(items) for key, items in diff.items()},
        **{key: items[:_LIST_LIMIT] for key, items in diff.items()},
        "conclusions": conclusion_summary,
    }


def refresh_seed_urls(previous_version: Dict[str, Any], rules=None) -> List[str]:
    """Previous sources a refresh re-reads first (bounded, rule-filtered)."""
    urls: List[str] = []
    for source in previous_version.get("sources") or []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "")
        if not url.startswith(("http://", "https://")) or url in urls:
            continue
        if rules is not None and not rules.allows(url):
            continue
        urls.append(url)
        if len(urls) >= MAX_REFRESH_SEED_URLS:
            break
    return urls


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable files are skipped
        return None
    return data if isinstance(data, dict) else None


def list_versions(data_dir: Path, lineage: str, owner: str) -> List[Dict[str, Any]]:
    """All saved versions of one question owned by ``owner``, oldest first."""
    versions = []
    for path in Path(data_dir).glob("*.json"):
        data = _read_json(path)
        if data is None or data.get("owner") != owner:
            continue
        if lineage_id(path.stem, data) != lineage:
            continue
        versions.append({
            "id": path.stem,
            "version": version_number(data),
            "status": data.get("status", "done"),
            "completed_at": data.get("completed_at", 0),
            "source_count": len(data.get("sources") or []),
            "previous_version_id": data.get("previous_version_id") or "",
        })
    versions.sort(key=lambda v: (v["version"], v["completed_at"] or 0))
    return versions


def next_version_number(data_dir: Path, lineage: str, owner: str) -> int:
    existing = list_versions(data_dir, lineage, owner)
    return (max(v["version"] for v in existing) if existing else 1) + 1
