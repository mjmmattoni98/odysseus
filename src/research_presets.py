# src/research_presets.py
"""Deep Research presets sized to the research model's hardware.

A preset bundles every knob that scales how much work (and how many model
calls) one research run costs: rounds, queries, URLs, page size fed to
extraction, generation budgets, how many findings synthesis sees, and whether
mechanical steps run with thinking disabled. ``auto`` picks small/medium/large
from the research model's context window and, for Ollama, its parameter size;
``custom`` keeps the pre-preset behavior driven by the individual settings.

The table here is the single source of truth; ``specs/research.md`` documents
it and ``/api/research/preset`` exposes it to the settings UI.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

PRESET_CHOICES = ("auto", "small", "medium", "large", "custom")

# Default of ``research_max_tokens`` in src/settings.py. A saved value that
# differs from it predates presets and is treated as explicit tuning.
_LEGACY_MAX_TOKENS_DEFAULT = 16384

# Same chars→tokens ratio as src.model_context.estimate_tokens (chars * 0.3).
TOKENS_PER_CHAR = 0.3

# Auto thresholds: small covers ≤16K windows and sub-10B models (Ollama reports
# e.g. "9.0B"/"9.2B" for the 9B class), medium covers windows up to 64K.
_SMALL_MAX_WINDOW = 16384
_SMALL_MAX_PARAMS_B = 10.0
_MEDIUM_MAX_WINDOW = 65536


@dataclass(frozen=True)
class ResearchPreset:
    name: str
    label: str
    description: str
    max_rounds: int             # cap used when the user picks Auto rounds
    min_rounds: int             # rounds before the stop check is consulted
    queries_first_round: int
    queries_per_round: int
    urls_per_query: int
    page_chars: int             # page text fed to one extraction call
    extraction_max_tokens: int
    query_max_tokens: int
    synthesis_max_tokens: int   # 0 → research_max_tokens setting
    report_max_tokens: int      # 0 → research_max_tokens setting
    report_min_words: int       # final report target length
    expand_below_words: int     # re-ask for a longer report below this
    synthesis_findings: int     # findings carried into each synthesis
    mechanical_think: Optional[bool]  # False → think=False on mechanical steps


PRESETS = {
    "small": ResearchPreset(
        name="small", label="Small",
        description="Small local models (≤16K context or under 10B parameters): "
                    "few short calls, compact prompts.",
        max_rounds=4, min_rounds=2,
        queries_first_round=3, queries_per_round=2, urls_per_query=2,
        page_chars=6000, extraction_max_tokens=768, query_max_tokens=768,
        synthesis_max_tokens=2048, report_max_tokens=3072,
        report_min_words=600, expand_below_words=250, synthesis_findings=5,
        mechanical_think=False,
    ),
    "medium": ResearchPreset(
        name="medium", label="Medium",
        description="Mid-size local models (up to 64K context): balanced depth "
                    "and call count.",
        max_rounds=6, min_rounds=2,
        queries_first_round=4, queries_per_round=3, urls_per_query=2,
        page_chars=12000, extraction_max_tokens=1536, query_max_tokens=1024,
        synthesis_max_tokens=4096, report_max_tokens=6144,
        report_min_words=1000, expand_below_words=300, synthesis_findings=8,
        mechanical_think=False,
    ),
    "large": ResearchPreset(
        name="large", label="Large",
        description="Large-context or cloud models (over 64K context): deeper "
                    "rounds, full pages and long reports.",
        max_rounds=10, min_rounds=3,
        queries_first_round=5, queries_per_round=3, urls_per_query=3,
        page_chars=20000, extraction_max_tokens=2048, query_max_tokens=2048,
        synthesis_max_tokens=8192, report_max_tokens=12288,
        report_min_words=1500, expand_below_words=400, synthesis_findings=12,
        mechanical_think=False,
    ),
    "custom": ResearchPreset(
        name="custom", label="Custom",
        description="Pre-preset behavior: Max Tokens and the other Deep Research "
                    "settings apply as configured.",
        max_rounds=20, min_rounds=2,
        queries_first_round=4, queries_per_round=3, urls_per_query=3,
        page_chars=15000, extraction_max_tokens=2048, query_max_tokens=4096,
        synthesis_max_tokens=0, report_max_tokens=0,
        report_min_words=1500, expand_below_words=400, synthesis_findings=10,
        mechanical_think=None,
    ),
}

AUTO_DESCRIPTION = ("Pick Small, Medium or Large from the research model's "
                    "context window and parameter size.")


@dataclass(frozen=True)
class ResearchProfile:
    """A preset resolved for one research model."""
    requested: str                    # configured choice (auto/small/…/custom)
    preset: ResearchPreset            # effective preset, token budgets filled
    context_window: Optional[int]     # known window, None when unknown
    parameter_size_b: Optional[float]  # billions of parameters, None when unknown
    reason: str


def chars_for_tokens(tokens: int) -> int:
    """Inverse of the estimate_tokens ratio: how many chars fit in ``tokens``."""
    return max(0, int(tokens / TOKENS_PER_CHAR))


def configured_preset() -> str:
    """The configured ``research_preset`` choice.

    An empty/unknown value means "never chosen": installs whose saved
    ``research_max_tokens`` differs from its default keep their explicit
    tuning as ``custom``; everyone else gets ``auto``.
    """
    from src.settings import get_setting
    raw = str(get_setting("research_preset", "") or "").strip().lower()
    if raw in PRESET_CHOICES:
        return raw
    try:
        tuned = int(get_setting("research_max_tokens", _LEGACY_MAX_TOKENS_DEFAULT)) != _LEGACY_MAX_TOKENS_DEFAULT
    except (TypeError, ValueError):
        tuned = False
    return "custom" if tuned else "auto"


def auto_preset_name(context_window: Optional[int], parameter_size_b: Optional[float]) -> str:
    """Map a model's known window / size to small, medium or large."""
    if context_window and context_window <= _SMALL_MAX_WINDOW:
        return "small"
    if parameter_size_b is not None and parameter_size_b < _SMALL_MAX_PARAMS_B:
        return "small"
    if context_window and context_window <= _MEDIUM_MAX_WINDOW:
        return "medium"
    if context_window:
        return "large"
    return "medium"  # nothing known: stay in the middle


def research_round_limits(max_rounds: int, preset: ResearchPreset) -> Tuple[int, int, bool]:
    """Return ``(max_rounds, min_rounds, auto)`` for a requested round count.

    ``max_rounds <= 0`` is Auto: the preset caps the run and the model's stop
    check runs after the preset's small minimum. An explicit count keeps its
    old meaning — roughly that many rounds (the stop check starts two rounds
    before the end, never before the preset minimum).
    """
    try:
        requested = int(max_rounds or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0:
        cap = max(1, preset.max_rounds)
        return cap, min(preset.min_rounds, cap), True
    return requested, max(min(requested, preset.min_rounds), requested - 2), False


# ---------------------------------------------------------------------------
# Ollama parameter size (/api/show details.parameter_size)
# ---------------------------------------------------------------------------
_PARAM_CACHE_TTL = 600.0
_PARAM_FAILURE_TTL = 60.0
_PARAM_PROBE_TIMEOUT = 1.5
_param_cache: dict = {}
_param_lock = threading.Lock()
_PARAM_RE = re.compile(r"^\s*([\d.]+)\s*([KMBT])?\s*$", re.IGNORECASE)
_PARAM_SCALE_B = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}


def parse_parameter_size(value) -> Optional[float]:
    """Parse Ollama's ``parameter_size`` ("27.3B", "23M") into billions."""
    match = _PARAM_RE.match(str(value or ""))
    if not match:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    unit = (match.group(2) or "B").upper()
    return number * _PARAM_SCALE_B[unit] if number > 0 else None


def ollama_parameter_size_b(url: str, model: str) -> Optional[float]:
    """Parameter size of a local Ollama model in billions, ``None`` if unknown.

    Read-only ``POST /api/show`` with a short timeout, cached per model.
    Non-Ollama and non-local endpoints are never probed.
    """
    from src.model_context import is_local_endpoint
    from src.ollama_capabilities import ollama_api_root

    root = ollama_api_root(url)
    model = str(model or "").strip()
    if not root or not model or not is_local_endpoint(url):
        return None
    key = (root, model)
    now = time.time()
    with _param_lock:
        cached = _param_cache.get(key)
    if cached is not None:
        ts, size = cached
        if now - ts < (_PARAM_FAILURE_TTL if size is None else _PARAM_CACHE_TTL):
            return size
    size = None
    try:
        import httpx
        response = httpx.post(f"{root}/api/show", json={"model": model}, timeout=_PARAM_PROBE_TIMEOUT)
        if response.is_success:
            details = (response.json() or {}).get("details") or {}
            size = parse_parameter_size(details.get("parameter_size"))
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        logger.debug("Ollama parameter-size probe failed for %s: %s", model, exc)
    with _param_lock:
        _param_cache[key] = (now, size)
    return size


def _fmt_window(tokens: Optional[int]) -> str:
    if not tokens:
        return "unknown context"
    return f"{tokens // 1024}K context" if tokens >= 1024 else f"{tokens}-token context"


def resolve_research_profile(
    url: str,
    model: str,
    *,
    requested: Optional[str] = None,
    max_report_tokens: Optional[int] = None,
) -> ResearchProfile:
    """Resolve the configured preset for ``model`` at ``url``.

    Blocking (may probe the endpoint) — call via ``asyncio.to_thread`` from
    async code. Probe failures degrade to "unknown"; never raises.
    """
    choice = (requested or configured_preset()).strip().lower()
    if choice not in PRESET_CHOICES:
        choice = "auto"

    window: Optional[int] = None
    try:
        from src.model_context import get_context_length_known
        ctx, known = get_context_length_known(url, model)
        window = int(ctx) if known and ctx else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Research context probe failed for %s: %s", model, exc)

    params: Optional[float] = None
    if choice == "auto":
        try:
            params = ollama_parameter_size_b(url, model)
        except Exception:  # noqa: BLE001
            params = None
        name = auto_preset_name(window, params)
        facts = [_fmt_window(window)]
        if params is not None:
            facts.append(f"{params:g}B parameters")
        reason = f"{', '.join(facts)} → {PRESETS[name].label}"
    else:
        name = choice
        reason = f"{PRESETS[name].label} selected"

    preset = PRESETS[name]
    if name == "custom":
        try:
            budget = int(max_report_tokens or _LEGACY_MAX_TOKENS_DEFAULT)
        except (TypeError, ValueError):
            budget = _LEGACY_MAX_TOKENS_DEFAULT
        preset = replace(preset, synthesis_max_tokens=budget, report_max_tokens=budget)
    return ResearchProfile(
        requested=choice,
        preset=preset,
        context_window=window,
        parameter_size_b=params,
        reason=reason,
    )


def preset_table() -> list:
    """Serializable preset table for the settings UI."""
    rows = [{"name": "auto", "label": "Auto", "description": AUTO_DESCRIPTION}]
    rows.extend(asdict(PRESETS[name]) for name in ("small", "medium", "large", "custom"))
    return rows


# ---------------------------------------------------------------------------
# Report length
# ---------------------------------------------------------------------------
# A length is chosen at launch (``report_length``) or read from the question
# ("briefly", "in 200 words", "one paragraph"). It reshapes the resolved
# preset: Brief does less work (fewer rounds, queries and pages, no per-round
# synthesis) as well as writing less; an explicit word count becomes a target
# with a hard limit the final prompt and its token budget both enforce.
REPORT_LENGTH_CHOICES = ("auto", "brief", "standard", "detailed")
REPORT_LENGTH_LABELS = {"brief": "Brief", "standard": "Standard", "detailed": "Detailed"}
BRIEF_DEFAULT_WORDS = 250
# Explicit word targets up to this size use the Brief work profile.
BRIEF_MAX_WORDS = 600
DETAILED_MIN_WORDS = 2000
# Hard limit allowed above an explicit/brief target before the writer is
# asked to condense.
WORD_LIMIT_SLACK = 1.25
# Output tokens per allowed word: prose plus markdown links (URLs are token
# heavy) and some headroom for models that think inside the output budget.
_TOKENS_PER_LIMIT_WORD = 3
_MIN_LIMITED_REPORT_TOKENS = 1536

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "single": 1, "a single": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "a few": 3, "few": 3, "a couple of": 2,
}
_COUNT = r"(\d{1,2}|a single|a couple of|a few|an|a|one|single|two|three|four|five|six|few)"
_BRIEF_RE = re.compile(
    r"\b(?:brief(?:ly)?|concise(?:ly)?|succinct(?:ly)?|tl;?dr|in a nutshell|keep it short|"
    r"short (?:summary|answer|overview|report|explanation|version)|"
    r"quick (?:summary|overview|answer|rundown)|high[- ]level (?:summary|overview)|"
    r"(?:summary|answer|overview) (?:only|in brief))\b",
    re.IGNORECASE,
)
_DETAILED_RE = re.compile(
    r"\b(?:in[- ]depth|detailed|comprehensive|exhaustive|thorough(?:ly)?|deep[- ]dive|"
    r"long[- ]form|extensive(?:ly)?)\b",
    re.IGNORECASE,
)
_MIN_WORDS_RE = re.compile(r"\b(?:at least|minimum of|no fewer than|min\.?)\s*(\d{2,5})\s*words?\b", re.IGNORECASE)
_WORDS_RE = re.compile(
    r"\b(?:(?:in|within|under|below|about|around|approximately|roughly|at most|no more than|"
    r"max(?:imum)?(?: of)?|up to|less than|fewer than)\s*~?\s*(\d{2,5})\s*words?|(\d{2,5})[- ]words?)\b",
    re.IGNORECASE,
)
_SENTENCES_RE = re.compile(rf"\b{_COUNT}\s+(?:short\s+)?sentences?\b", re.IGNORECASE)
_PARAGRAPHS_RE = re.compile(rf"\b{_COUNT}\s+(?:short\s+)?paragraphs?\b", re.IGNORECASE)


@dataclass(frozen=True)
class ReportLengthPlan:
    """How long the final report should be and how much work to spend."""
    name: str                   # brief / standard / detailed
    word_target: Optional[int]  # None → the preset's report_min_words
    word_limit: Optional[int]   # hard maximum; None → open-ended ("at least")
    source: str                 # setting / question / default

    @property
    def label(self) -> str:
        return REPORT_LENGTH_LABELS.get(self.name, self.name.title())

    def describe(self, preset_words: Optional[int] = None) -> str:
        words = self.word_target or preset_words
        return f"{self.label} (~{words} words)" if words else self.label

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "word_target": self.word_target,
                "word_limit": self.word_limit, "source": self.source}


def _count(token: str) -> int:
    token = (token or "").strip().lower()
    if token.isdigit():
        return max(1, min(int(token), 30))
    return _NUMBER_WORDS.get(token, 1)


def detect_length_request(question: str) -> Tuple[Optional[str], Optional[int], bool]:
    """Read an explicit length request from the question.

    Returns ``(length_name, word_count, is_minimum)``. ``word_count`` is set
    when the question names a size ("in 200 words", "two paragraphs");
    ``is_minimum`` marks "at least N words".
    """
    text = str(question or "")
    match = _MIN_WORDS_RE.search(text)
    if match:
        words = int(match.group(1))
        return ("detailed" if words >= DETAILED_MIN_WORDS else "standard"), words, True
    match = _WORDS_RE.search(text)
    words = None
    if match:
        words = int(match.group(1) or match.group(2))
    else:
        match = _PARAGRAPHS_RE.search(text)
        if match:
            words = _count(match.group(1)) * 150
        else:
            match = _SENTENCES_RE.search(text)
            if match:
                words = max(40, _count(match.group(1)) * 25)
    if words:
        words = max(30, min(words, 20000))
        if words <= BRIEF_MAX_WORDS:
            return "brief", words, False
        return ("detailed" if words >= DETAILED_MIN_WORDS else "standard"), words, False
    if _BRIEF_RE.search(text):
        return "brief", None, False
    if _DETAILED_RE.search(text):
        return "detailed", None, False
    return None, None, False


def resolve_report_length(requested: Optional[str], question: str) -> ReportLengthPlan:
    """Combine the launch choice with any length request in the question.

    An explicit launch choice (Brief/Standard/Detailed) picks the work
    profile; Auto (or nothing) follows the question's wording. A word count
    named in the question is always the target, since it is the most
    specific instruction the user gave.
    """
    choice = str(requested or "auto").strip().lower()
    if choice not in REPORT_LENGTH_CHOICES:
        choice = "auto"
    detected, words, is_minimum = detect_length_request(question)
    if choice != "auto":
        name, source = choice, "setting"
    elif detected:
        name, source = detected, "question"
    else:
        name, source = "standard", "default"
    if name == "brief" and not words:
        words = BRIEF_DEFAULT_WORDS
    limit = None
    if words and not is_minimum:
        limit = max(words + 20, int(round(words * WORD_LIMIT_SLACK)))
    return ReportLengthPlan(name=name, word_target=words, word_limit=limit, source=source)


def limited_report_tokens(word_limit: int) -> int:
    """Output-token budget for a report with a hard word limit."""
    return max(_MIN_LIMITED_REPORT_TOKENS, int(word_limit) * _TOKENS_PER_LIMIT_WORD)


def apply_report_length(preset: ResearchPreset, plan: Optional[ReportLengthPlan]) -> ResearchPreset:
    """Reshape a resolved preset (token budgets already filled) for ``plan``."""
    if plan is None:
        return preset
    changes = {}
    if plan.name == "brief":
        # Less work, not just less text: two rounds at most, fewer queries and
        # pages, a smaller synthesis window, and no "too short" expansion.
        changes.update(
            max_rounds=min(preset.max_rounds, 2), min_rounds=1,
            queries_first_round=min(preset.queries_first_round, 3),
            queries_per_round=min(preset.queries_per_round, 2),
            urls_per_query=min(preset.urls_per_query, 2),
            synthesis_findings=min(preset.synthesis_findings, 6),
            expand_below_words=0,
        )
    elif plan.name == "detailed" and not plan.word_target:
        changes.update(
            report_min_words=max(preset.report_min_words * 3 // 2, preset.report_min_words + 300),
            expand_below_words=max(preset.expand_below_words, preset.report_min_words // 2),
        )
    if plan.word_target:
        changes["report_min_words"] = plan.word_target
        if plan.word_limit:
            budget = limited_report_tokens(plan.word_limit)
            changes["report_max_tokens"] = budget
            # The evolving report only needs to carry what a report this
            # size can use; a smaller budget also keeps synthesis fast.
            changes["synthesis_max_tokens"] = min(preset.synthesis_max_tokens or budget * 2, budget * 2)
            changes["expand_below_words"] = 0
    return replace(preset, **changes) if changes else preset


# ---------------------------------------------------------------------------
# Word counting and the hard word limit
# ---------------------------------------------------------------------------
# A report's length is its visible prose, the words a reader sees: link text
# counts, link targets and markdown markers (#, -, |) do not, and a trailing
# Sources/References list is citations rather than answer (the UI lists
# sources separately). ``trim_report_to_words`` enforces ``word_limit`` on the
# same count, cutting at block and sentence boundaries so headings, lists,
# code blocks and links stay well formed.
_LINK_RE = re.compile(r"!?\[([^\]\n]*)\]\((?:<[^>\n]*>|[^)\s]*)(?:\s+\"[^\"\n]*\")?\)")
_SOURCES_LABEL_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
    r"(?:sources|references|citations|bibliography|works cited)(?:[ \t]+used)?"
    r"[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]|$)", re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]{0,3}(```|~~~)")
_LIST_ITEM_RE = re.compile(r"^[ \t]{0,3}(?:[-*+]|\d{1,3}[.)])[ \t]+")
_RULE_RE = re.compile(r"^[ \t]{0,3}(?:[-*_][ \t]*){3,}$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"'”’)\]]*(?=\s)")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")
_ELLIPSIS = "…"


def _split_sources_section(markdown: str) -> Tuple[str, str]:
    """``(body, trailing sources list)``; the list is "" when there is none."""
    matches = list(_SOURCES_LABEL_RE.finditer(markdown))
    if not matches or _HEADING_LINE_RE.search(markdown, matches[-1].end()):
        return markdown, ""
    start = matches[-1].start()
    return markdown[:start], markdown[start:]


def _visible_words(text: str) -> list:
    text = _LINK_RE.sub(lambda m: m.group(1), text)
    return [token for token in text.split() if any(ch.isalnum() for ch in token)]


def count_report_words(markdown) -> int:
    """Words a reader sees in the report body (see the section comment)."""
    body, _sources = _split_sources_section(str(markdown or ""))
    return len(_visible_words(body))


def _markdown_blocks(body: str) -> list:
    """Blank-line separated blocks; a fenced code block is never split."""
    blocks, current, fence = [], [], None
    for line in body.split("\n"):
        marker = _FENCE_RE.match(line)
        if fence:
            current.append(line)
            if marker and marker.group(1) == fence:
                fence = None
            continue
        if marker:
            fence = marker.group(1)
            current.append(line)
        elif line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def _is_structural(block: str) -> bool:
    """Only headings or horizontal rules: nothing to read on its own."""
    lines = [line for line in block.split("\n") if line.strip()]
    return bool(lines) and all(_HEADING_LINE_RE.match(line) or _RULE_RE.match(line) for line in lines)


def _protect_links(text: str) -> Tuple[str, list]:
    """Replace links by whitespace-free placeholders so cuts never split one."""
    links: list = []

    def _hold(match):
        links.append(match.group(0))
        return f"\x00{len(links) - 1}\x00"

    return _LINK_RE.sub(_hold, text), links


def _restore_links(text: str, links: list) -> str:
    return _PLACEHOLDER_RE.sub(lambda m: links[int(m.group(1))], text)


def _trim_text(text: str, room: int, *, allow_fragment: bool) -> str:
    """Longest prefix of ``text`` ending at a sentence with ``room`` words.

    With ``allow_fragment`` and no whole sentence that fits, cut at a word
    instead and mark the cut with an ellipsis.
    """
    if room <= 0:
        return ""
    protected, links = _protect_links(text)

    def words(piece: str) -> int:
        return len(_visible_words(_restore_links(piece, links)))

    best = ""
    for match in _SENTENCE_END_RE.finditer(protected):
        piece = protected[:match.end()]
        if words(piece) > room:
            break
        best = piece
    if best:
        # Keep a citation link that directly follows the sentence it supports.
        tail = re.match(r"\s*(?:\x00\d+\x00[.,;:]?\s*)+", protected[len(best):])
        if tail and words(best + tail.group(0)) <= room:
            best += tail.group(0)
        return _restore_links(best, links).rstrip()
    if not allow_fragment:
        return ""
    kept, used = [], 0
    for token in re.split(r"(\s+)", protected):
        count = words(token)
        if used + count > room:
            break
        kept.append(token)
        used += count
    fragment = _restore_links("".join(kept), links).rstrip().rstrip(",;:")
    if not fragment:
        return ""
    if fragment.count("**") % 2:
        fragment += "**"
    if fragment.count("`") % 2:
        fragment += "`"
    return fragment + _ELLIPSIS


def _trim_list(block: str, room: int, *, allow_fragment: bool) -> str:
    """Whole list items that fit; the next item only up to a sentence."""
    items: list = []
    for line in block.split("\n"):
        if _LIST_ITEM_RE.match(line) or not items:
            items.append([line])
        else:
            items[-1].append(line)
    kept, used = [], 0
    for item in ("\n".join(lines) for lines in items):
        count = len(_visible_words(item))
        if used + count <= room:
            kept.append(item)
            used += count
            continue
        partial = _trim_text(item, room - used, allow_fragment=allow_fragment and not kept)
        if partial and _visible_words(partial):
            kept.append(partial)
        break
    return "\n".join(kept)


def _trim_block(block: str, room: int, *, allow_fragment: bool) -> str:
    if room <= 0 or _is_structural(block):
        return ""
    first = block.lstrip("\n").split("\n", 1)[0]
    if _FENCE_RE.match(first) or first.lstrip().startswith("|"):
        return ""  # code blocks and tables are kept whole or not at all
    if _LIST_ITEM_RE.match(first):
        return _trim_list(block, room, allow_fragment=allow_fragment)
    return _trim_text(block, room, allow_fragment=allow_fragment)


def trim_report_to_words(markdown, limit: Optional[int]) -> str:
    """Cut a report to at most ``limit`` words (``count_report_words``).

    Whole blocks are kept in order; the first block that does not fit is cut
    at a sentence (a list at an item), and code blocks and tables are never
    split. A heading left without its section is dropped, and a trailing
    sources list is kept as is (it is not counted). Only when not even one
    sentence fits is the text cut at a word, marked with an ellipsis.
    """
    text = str(markdown or "")
    if not limit or limit <= 0 or count_report_words(text) <= limit:
        return text
    body, sources = _split_sources_section(text)
    kept: list = []
    used = 0
    prose_kept = False
    for block in _markdown_blocks(body):
        count = len(_visible_words(block))
        if used + count <= limit:
            kept.append(block)
            used += count
            prose_kept = prose_kept or not _is_structural(block)
            continue
        partial = _trim_block(block, limit - used, allow_fragment=not prose_kept)
        if partial:
            kept.append(partial)
        break
    while kept and _is_structural(kept[-1]):
        kept.pop()
    out = "\n\n".join(kept).strip()
    if sources.strip():
        out = f"{out}\n\n{sources.strip()}" if out else sources.strip()
    return out
