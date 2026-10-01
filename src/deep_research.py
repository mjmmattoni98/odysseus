# src/deep_research.py
"""
IterResearch-style deep research engine.

Implements an iterative Think→Search→Extract→Synthesize loop where the LLM
drives every decision: what to search, what's relevant, what's missing, and
when to stop.  Inspired by Alibaba's IterResearch approach.
"""
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

from src.research_utils import strip_thinking, is_low_quality
from src.model_context import estimate_tokens

from src.goal_based_extractor import EXTRACTOR_SYSTEM
from src.prompt_security import untrusted_context_message

logger = logging.getLogger(__name__)


def current_date_context() -> str:
    """Preamble that grounds query-generation/planning LLMs in the real current
    date. Without it the model falls back to its training-cutoff year and emits
    queries like "best Python tutorials 2025" when the year is actually 2026.
    System TZ-local so it matches what the user sees. Portable strftime only."""
    now = datetime.now().astimezone()
    return (
        f"Today's date is {now.strftime('%B %d, %Y')} ({now.strftime('%Y-%m-%d')}). "
        f"When a search query needs a year or refers to 'latest'/'current'/"
        f"'this year', use {now.strftime('%Y')} or relative wording — never a "
        f"year inferred from training data.\n\n"
    )

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
RESEARCH_PLAN_PROMPT = """\
You are a research strategist. Before searching, analyze this question and create a research plan.

**Question:** {question}

Break this question down:
1. What are the key sub-topics that need to be covered for a comprehensive answer?
2. What specific data points, facts, or perspectives should we look for?
3. What would a complete, high-quality answer include?

Return a JSON object with:
- "sub_questions": Array of 3-6 specific sub-questions to investigate
- "key_topics": Array of key topics/angles to cover
- "success_criteria": One sentence describing what a complete answer looks like

Example:
{{
  "sub_questions": ["What is the cost of living in X?", "How is the healthcare system?"],
  "key_topics": ["economy", "healthcare", "safety", "culture"],
  "success_criteria": "A balanced comparison covering cost, quality of life, and practical considerations."
}}
"""

QUERY_GEN_PROMPT = """\
You are a research assistant planning web searches.

**Original question:** {question}

**Research plan:**
{research_plan}

**What we know so far:**
{report}

**Round:** {round_num}

Generate {num_queries} focused search queries that will help answer the question.
{round_instruction}

Return ONLY a JSON array of query strings, nothing else.
Example: ["query one", "query two", "query three"]
"""

SYNTHESIZE_PROMPT = """\
You are updating an evolving research report.

**Original question:** {question}

**Current report:**
{report}

**New findings from this round:**
{new_findings}

Integrate the new findings into the existing report. Produce an updated, well-organized \
report that answers the original question as completely as possible given all evidence so far. \
Remove redundancy, resolve contradictions, and maintain logical flow. \
Keep source URLs as inline citations where relevant.

Write only the updated report — no preamble or meta-commentary.
"""

STOP_PROMPT = """\
You are deciding whether a research report is comprehensive enough.

**Original question:** {question}

**Current report:**
{report}

{rounds_line}

Based on the report so far, do we have enough information to answer the question \
comprehensively?  Consider:
- Are the key aspects of the question addressed?
- Are there obvious gaps or unanswered sub-questions?
- Is the evidence sufficient and from multiple sources?

{rounds_guidance}

Reply with ONLY "YES" or "NO" followed by a brief one-sentence reason.
Example: "YES — The report covers all major aspects with evidence from multiple sources."
Example: "NO — We still lack information about the economic impact."
"""

FINAL_REPORT_PROMPT = """\
Write a **long, detailed, comprehensive** research report answering this question:

**Question:** {question}

**All collected evidence and analysis:**
{report}

Requirements:
- Write at MINIMUM {min_words} words — this should be a thorough, magazine-quality article
- Use clear ## headings and ### subheadings to organize into logical sections
- Each section should have multiple detailed paragraphs, not just bullet points
- Synthesize and analyze the information — explain WHY things matter, draw comparisons, provide context
- Include specific data points, numbers, and statistics from the evidence
- Include source URLs as inline citations [like this](url)
- Note where sources agree and where they disagree
- Add a brief executive summary at the top
- End with a clear conclusion that directly answers the question
- Write in an engaging, informative style — not dry or robotic
"""

CATEGORY_PROMPTS = {
    "product": """IMPORTANT FORMAT OVERRIDE — this is a PRODUCT research report:
- Structure as a RANKED LIST of products/options (best first)
- For EACH product include: name as ### heading, approximate price, 2-3 sentence summary, **Pros:** bullet list, **Cons:** bullet list, **Where to buy:** URLs as links
- Start with a quick-compare markdown table of top picks (columns: Name, Price, Best For, Rating)
- End with a ## Verdict section picking Best Overall and Best Value
- Still include source citations inline""",

    "comparison": """IMPORTANT FORMAT OVERRIDE — this is a COMPARISON report:
- Create a ## Comparison Table as a markdown table comparing ALL options across key criteria (rows = criteria, columns = options)
- Use checkmarks, ratings, or short values in cells
- Write a ## section per option with its strengths, weaknesses, and ideal use case
- End with ## Best For verdicts (e.g., "**Best for small teams:** Option A because...")
- Include a ## Shared Considerations section for things that apply to all options""",

    "howto": """IMPORTANT FORMAT OVERRIDE — this is a HOW-TO guide:
- Start with ## Quick Guide — a super concise numbered list (one line per step, no details, just the action). Example: 1. Install X  2. Run Y  3. Configure Z
- Then ## Prerequisites listing what's needed before starting
- Then the detailed steps: ## Step 1: ..., ## Step 2: ...
- Each step should have a clear heading and detailed instructions
- Use blockquotes (> ) for tips and warnings: > **Tip:** ... or > **Warning:** ...
- End with ## Common Mistakes section
- Add estimated time and difficulty level near the top""",

    "factcheck": """IMPORTANT FORMAT OVERRIDE — this is a FACT-CHECK report:
- Start with ## The Claim restating what's being checked
- Create ## Evidence For and ## Evidence Against sections
- Each piece of evidence should be a ### with source name, what it found, and how strong the evidence is
- Include a ## Verdict section with one of: **Supported**, **Mixed Evidence**, or **Unsupported**
- End with ## Nuance & Caveats for important context and limitations
- Be balanced and cite sources for every claim""",
}

_ROUNDS_EXPLICIT = "**Rounds completed:** {round_num} of {max_rounds}"
_ROUNDS_EXPLICIT_GUIDANCE = (
    "If rounds completed is well below the target, prefer continuing unless the "
    "report is already exhaustive."
)
_ROUNDS_AUTO = "**Rounds completed:** {round_num} (automatic mode, at most {max_rounds})"
_ROUNDS_AUTO_GUIDANCE = (
    "There is no fixed round target: answer YES as soon as the key aspects are "
    "covered with evidence, NO only when a concrete gap remains."
)

# ---------------------------------------------------------------------------
# Structured-output schemas (response_schema). Local Ollama/OpenAI-compatible
# servers constrain decoding to these; cloud providers ignore them, so every
# caller keeps its prompt instructions and defensive parsing. Keep them to
# plain objects/arrays/strings/enums so llama.cpp-style grammars accept them.
# ---------------------------------------------------------------------------
_STRING_LIST_SCHEMA = {"type": "array", "items": {"type": "string"}}

RESEARCH_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_questions": _STRING_LIST_SCHEMA,
        "key_topics": _STRING_LIST_SCHEMA,
        "success_criteria": {"type": "string"},
    },
    "required": ["sub_questions", "key_topics", "success_criteria"],
}

QUERY_LIST_SCHEMA = _STRING_LIST_SCHEMA

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "rational": {"type": "string"},
        "evidence": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["rational", "evidence", "summary"],
}

CATEGORY_SCHEMA = {"type": "string", "enum": [*CATEGORY_PROMPTS, "general"]}

STOP_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["YES", "NO"]},
        "reason": {"type": "string"},
    },
    "required": ["answer", "reason"],
}

# Prompt scaffolding (instructions, question, headings) reserved on top of the
# embedded report/findings when fitting a prompt into the context window.
_PROMPT_OVERHEAD_TOKENS = 800
# Embedded report + findings never take more than this share of the window.
_EMBED_WINDOW_FRACTION = 0.6
# Findings get at most this share of the embed budget; the report gets the rest.
_FINDINGS_BUDGET_FRACTION = 0.5
_CONDENSED_MARKER = "[Condensed to fit the model's context window — section details are shortened.]\n\n"

# ---------------------------------------------------------------------------
# DeepResearcher
# ---------------------------------------------------------------------------
class DeepResearcher:
    """
    Iterative research engine following the IterResearch pattern.

    Each round: LLM generates queries → SearXNG search → LLM extracts from
    top pages → LLM synthesizes into evolving report → LLM decides continue/stop.
    """

    # Preset-controlled knobs (see src/research_presets.py). Class-level
    # defaults reproduce the pre-preset behavior for callers that build the
    # researcher without them.
    queries_first_round: int = 4
    queries_per_round: int = 3
    extraction_max_tokens: int = 2048
    query_max_tokens: int = 4096
    synthesis_max_tokens: Optional[int] = None  # None → max_report_tokens
    report_min_words: int = 1500
    expand_below_words: int = 400
    mechanical_think: Optional[bool] = None
    context_window: Optional[int] = None  # known window; None = don't bound
    auto_rounds: bool = False
    preset_label: str = ""

    def __init__(
        self,
        llm_endpoint: str,
        llm_model: str,
        llm_headers: Optional[Dict] = None,
        max_rounds: int = 8,
        max_time: int = 300,
        max_urls_per_round: int = 3,
        max_content_chars: int = 15000,
        max_report_tokens: int = 8192,
        extraction_timeout: int = 90,
        planning_timeout: int = 90,
        query_timeout: int = 120,
        extraction_concurrency: int = 3,
        min_rounds: int = 2,
        max_empty_rounds: int = 2,
        synthesis_window: int = 10,
        progress_callback: Optional[Callable] = None,
        search_provider: Optional[str] = None,
        category: Optional[str] = None,
        queries_first_round: int = 4,
        queries_per_round: int = 3,
        extraction_max_tokens: int = 2048,
        query_max_tokens: int = 4096,
        synthesis_max_tokens: Optional[int] = None,
        report_min_words: int = 1500,
        expand_below_words: int = 400,
        mechanical_think: Optional[bool] = None,
        context_window: Optional[int] = None,
        auto_rounds: bool = False,
        preset_label: str = "",
    ):
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.llm_headers = llm_headers
        self.search_provider_override = search_provider
        self.category = category
        self.max_rounds = max_rounds
        self.max_time = max_time
        self.max_urls_per_round = max_urls_per_round
        self.max_content_chars = max_content_chars
        self.max_report_tokens = max_report_tokens
        self.extraction_timeout = min(3600, max(15, int(extraction_timeout or 90)))
        self.planning_timeout = min(3600, max(15, int(planning_timeout or 90)))
        self.query_timeout = min(3600, max(15, int(query_timeout or 120)))
        self.extraction_concurrency = min(12, max(1, int(extraction_concurrency or 3)))
        self.min_rounds = min_rounds
        self.max_empty_rounds = max_empty_rounds
        self.synthesis_window = synthesis_window
        self.queries_first_round = max(1, int(queries_first_round or 4))
        self.queries_per_round = max(1, int(queries_per_round or 3))
        self.extraction_max_tokens = max(256, int(extraction_max_tokens or 2048))
        self.query_max_tokens = max(128, int(query_max_tokens or 4096))
        self.synthesis_max_tokens = synthesis_max_tokens
        self.report_min_words = max(100, int(report_min_words or 1500))
        self.expand_below_words = max(0, int(expand_below_words or 0))
        self.mechanical_think = mechanical_think
        self.context_window = int(context_window) if context_window else None
        self.auto_rounds = bool(auto_rounds)
        self.preset_label = preset_label or ""
        self._progress = progress_callback
        self._cancelled = False
        self._start_time: float = 0
        self.queries_used: Set[str] = set()
        self.urls_fetched: Set[str] = set()
        self.analyzed_urls: List[Dict[str, str]] = []
        self.round_count: int = 0
        # Track which search providers actually returned results during the
        # run, in arrival order — surfaced in the visual report so users can
        # see whether searxng / brave / tavily etc. carried the work.
        self.providers_used: List[str] = []
        self.findings: List[Dict] = []
        self.evolving_report: str = ""
        self.research_plan: str = ""

    def cancel(self):
        """Request cooperative cancellation of the research loop."""
        self._cancelled = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def research(
        self,
        question: str,
        prior_report: str = "",
        prior_findings: Optional[List[Dict]] = None,
        prior_urls: Optional[Set[str]] = None,
    ) -> str:
        """Run iterative research and return a final report.

        Args:
            question: The research question.
            prior_report: Previous report to continue from (for follow-up research).
            prior_findings: Previous findings to build on.
            prior_urls: URLs already visited (won't be re-fetched).
        """
        self._start_time = time.time()
        findings: List[Dict] = list(prior_findings) if prior_findings else []
        report = prior_report or ""

        # PLAN: Analyze the question and create a research strategy
        if not prior_report:
            self._emit(phase="planning")
            self.research_plan = await self._create_plan(question)
            logger.info(f"Research plan: {self.research_plan[:200]}")
        else:
            # Continuation — plan around the follow-up
            self._emit(phase="planning")
            self.research_plan = await self._create_plan(question)
            logger.info(f"Continuation plan: {self.research_plan[:200]}")
        if not self.category and not prior_report:
            self.category = await self._classify_category(question)
            if self.category:
                logger.info(f"Auto-detected category: {self.category}")

        if prior_urls:
            self.urls_fetched.update(prior_urls)
        self.findings = findings  # expose for handler
        consecutive_empty_rounds = 0

        for round_num in range(1, self.max_rounds + 1):
            self.round_count = round_num
            if self._cancelled:
                logger.info(f"Research cancelled after {round_num - 1} rounds")
                break
            if self._time_exceeded():
                logger.info(f"Time limit reached after {round_num - 1} rounds")
                break

            logger.info(f"=== Research Round {round_num} ===")
            self._emit(phase="searching", round=round_num, total_sources=len(self.urls_fetched))

            # THINK: generate queries
            queries = await self._generate_queries(question, report, round_num)
            if not queries:
                logger.warning(f"Round {round_num}: no queries generated, stopping")
                break

            self._emit(phase="searching", round=round_num, queries=len(queries),
                       query_preview=queries[0] if queries else "",
                       total_sources=len(self.urls_fetched))

            # SEARCH + EXTRACT
            round_findings = await self._search_and_extract(queries, question)
            if round_findings:
                findings.extend(round_findings)
                consecutive_empty_rounds = 0
                logger.info(f"Round {round_num}: extracted {len(round_findings)} findings")
                self._emit(phase="reading", round=round_num,
                           new_sources=len(round_findings),
                           total_sources=len(self.urls_fetched),
                           total_findings=len(findings))
            else:
                consecutive_empty_rounds += 1
                logger.info(f"Round {round_num}: no new findings ({consecutive_empty_rounds} consecutive empty)")
                if consecutive_empty_rounds >= self.max_empty_rounds:
                    logger.warning(f"Search appears to be down — {self.max_empty_rounds} consecutive rounds with no results")
                    err_detail = getattr(self, '_last_search_error', 'unknown error')
                    self._emit(phase="error", message=f"Search engine unavailable: {err_detail}")
                    if not findings:
                        return (
                            f"**Search unavailable** — Web search failed after "
                            f"{round_num} rounds. Error: {err_detail}\n\n"
                            "Please check your search provider settings and ensure the service is running."
                        )
                    break

            # SYNTHESIZE
            if findings:
                self._emit(phase="analyzing", round=round_num,
                           total_sources=len(self.urls_fetched),
                           total_findings=len(findings))
                report = await self._synthesize(question, findings, report)

            # DECIDE
            if round_num >= self.min_rounds:
                should_stop = await self._should_stop(question, report, round_num)
                if should_stop:
                    logger.info(f"LLM decided to stop after round {round_num}")
                    break

        # FINAL REPORT
        self._emit(phase="writing", total_sources=len(self.urls_fetched),
                   total_findings=len(findings))
        if not report:
            # Synthesis can fail (e.g. the LLM timed out) even though the search
            # rounds did gather findings. Don't throw that work away — return the
            # gathered findings as a basic compiled report instead of claiming
            # nothing was found (#1551).
            if findings:
                logger.warning(
                    "Synthesis produced no report; returning %d gathered "
                    "finding(s) as a fallback", len(findings)
                )
                return self._fallback_report(question, findings)
            return "No information could be gathered for this question."

        self.evolving_report = report  # preserve pre-synthesis report
        final = await self._final_report(question, report)
        elapsed = time.time() - self._start_time
        logger.info(
            f"Research complete: {self.round_count} rounds, "
            f"{len(findings)} findings, {len(self.urls_fetched)} URLs, "
            f"{elapsed:.1f}s"
        )
        return final

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------
    async def _llm(self, messages: List[Dict], temperature: float = 0.3,
                   max_tokens: int = 4096, timeout: int = 60,
                   think=None, response_schema: Optional[Dict] = None) -> str:
        """Call the LLM asynchronously and strip thinking tags.

        ``think``/``response_schema`` are forwarded to ``llm_call_async``
        (thinking override and structured output for local servers).
        """
        from src.llm_core import llm_call_async
        response = await llm_call_async(
            url=self.llm_endpoint,
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=self.llm_headers,
            timeout=timeout,
            workload="research",
            think=think,
            response_schema=response_schema,
        )
        return strip_thinking(response)

    # ------------------------------------------------------------------
    # Context-window fitting
    # ------------------------------------------------------------------
    def _output_budget(self, max_tokens: int) -> int:
        """Clamp a generation budget to half the known window."""
        window = self.context_window
        if not window:
            return max_tokens
        return max(256, min(max_tokens, window // 2))

    def _embed_budget_chars(self, max_tokens: int) -> Optional[int]:
        """Chars of report/findings/page text that fit next to ``max_tokens``
        of output, or ``None`` when the window is unknown (no bounding)."""
        window = self.context_window
        if not window:
            return None
        from src.research_presets import chars_for_tokens
        tokens = min(int(window * _EMBED_WINDOW_FRACTION),
                     window - max_tokens - _PROMPT_OVERHEAD_TOKENS)
        return max(chars_for_tokens(256), chars_for_tokens(tokens))

    @staticmethod
    def _cut(text: str, limit: int) -> str:
        """Cut ``text`` to ``limit`` chars at a paragraph/sentence boundary."""
        if len(text) <= limit:
            return text
        cut = text[:limit]
        for sep in ("\n\n", "\n", ". "):
            pos = cut.rfind(sep)
            if pos > limit * 0.6:
                return cut[:pos + (1 if sep == ". " else 0)].rstrip()
        return cut.rstrip()

    def _condense_report(self, report: str, max_chars: Optional[int]) -> str:
        """Fit an evolving report into ``max_chars``.

        Keeps every section heading and shortens the bodies of the longest
        sections first (water-filling), so synthesis still sees the whole
        structure instead of losing the report's head or tail.
        """
        if not report or max_chars is None or len(report) <= max_chars:
            return report
        sections = re.split(r'\n(?=#{1,6} )', report)
        allowance = max(0, max_chars - len(_CONDENSED_MARKER)) - len(sections)
        # Water-fill: short sections keep their text, long ones share the rest.
        shares = {}
        remaining = allowance
        pending = sorted(range(len(sections)), key=lambda i: len(sections[i]))
        while pending:
            fair = remaining // len(pending)
            i = pending[0]
            if len(sections[i]) <= fair:
                shares[i] = len(sections[i])
                remaining -= shares[i]
                pending.pop(0)
                continue
            for j in pending:
                shares[j] = fair
            break
        parts = []
        for i, section in enumerate(sections):
            share = shares.get(i, 0)
            if len(section) <= share:
                parts.append(section)
                continue
            heading, _, body = section.partition("\n")
            if not heading.lstrip().startswith("#"):
                heading, body = "", section
            body_limit = max(0, share - len(heading) - 3)
            short = self._cut(body, body_limit) if body_limit else ""
            parts.append("\n".join(p for p in (heading, short + " …" if short else "") if p))
        condensed = "\n".join(parts)
        if len(condensed) > max_chars:
            condensed = condensed[:max(0, max_chars - len(_CONDENSED_MARKER))]
        logger.info(f"Condensed report from {len(report)} to {len(condensed)} chars for the context window")
        return _CONDENSED_MARKER + condensed

    # ------------------------------------------------------------------
    # PLAN: create research strategy
    # ------------------------------------------------------------------
    async def _create_plan(self, question: str) -> str:
        """LLM analyzes the question and creates a research plan."""
        prompt = current_date_context() + RESEARCH_PLAN_PROMPT.format(question=question)
        try:
            response = await self._llm(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                timeout=getattr(self, "planning_timeout", 90),
                think=self.mechanical_think,
                response_schema=RESEARCH_PLAN_SCHEMA,
            )
            # Try to parse as JSON for structured plan
            parsed = self._parse_json_object(response)
            if parsed:
                parts = []
                if parsed.get("sub_questions"):
                    parts.append("Sub-questions: " + "; ".join(parsed["sub_questions"]))
                if parsed.get("key_topics"):
                    parts.append("Key topics: " + ", ".join(parsed["key_topics"]))
                if parsed.get("success_criteria"):
                    parts.append("Success: " + parsed["success_criteria"])
                return "\n".join(parts) if parts else response
            return response
        except Exception as e:
            logger.warning(f"Research planning failed: {e}")
            self._emit(phase="warning", message="Planning step failed, proceeding with direct search")
            return ""

    async def _classify_category(self, question: str) -> Optional[str]:
        """Fast LLM call to classify the research question into a category."""
        valid = ", ".join(CATEGORY_PROMPTS.keys())
        prompt = (
            f"Classify this research question into exactly ONE category.\n"
            f"Categories: {valid}\n"
            f"If none fit well, respond with: general\n\n"
            f"Question: {question}\n\n"
            f"Respond with ONLY the category name, nothing else."
        )
        try:
            result = await self._llm(
                [{"role": "user", "content": prompt}],
                temperature=0, max_tokens=20, timeout=15,
                think=self.mechanical_think,
                response_schema=CATEGORY_SCHEMA,
            )
            cat = (result or "").strip().lower()
            # Clean one-word answer first.
            parts = cat.split()
            first = parts[0].strip(".,\"'*:") if parts else ""
            if first in CATEGORY_PROMPTS:
                return first
            # Weak local models often wrap the label in preamble ("the category
            # is product") — scan the whole reply for any known category word
            # before giving up (which would default to the generic format).
            for c in CATEGORY_PROMPTS:
                if c in cat:
                    return c
            return None
        except Exception as e:
            logger.warning(f"Category classification failed: {e}")
            return None

    # ------------------------------------------------------------------
    # THINK: generate search queries
    # ------------------------------------------------------------------
    async def _generate_queries(self, question: str, report: str,
                                round_num: int) -> List[str]:
        if round_num == 1:
            num_queries = self.queries_first_round
            round_instruction = (
                "This is the first round — generate broad, diverse queries "
                "that explore the key facets of the question."
            )
        else:
            num_queries = self.queries_per_round
            round_instruction = (
                "We already have partial findings.  Generate targeted follow-up "
                "queries to fill gaps, verify claims, or explore specific aspects "
                "that the report doesn't yet cover well."
            )

        query_tokens = self._output_budget(self.query_max_tokens)
        prompt = current_date_context() + QUERY_GEN_PROMPT.format(
            question=question,
            research_plan=self.research_plan or "(No plan — search broadly.)",
            report=self._condense_report(report, self._embed_budget_chars(query_tokens)) or "(No findings yet.)",
            round_num=round_num,
            num_queries=num_queries,
            round_instruction=round_instruction,
        )

        try:
            response = await self._llm(
                [{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=query_tokens,
                timeout=getattr(self, "query_timeout", 120),
                think=self.mechanical_think,
                response_schema=QUERY_LIST_SCHEMA,
            )
            queries = self._parse_json_array(response)
            # Deduplicate, then cap at the requested count so a chatty model
            # can't multiply the round's extraction calls.
            new_queries = [q for q in queries if q.strip() and q not in self.queries_used][:num_queries]
            self.queries_used.update(new_queries)
            logger.info(f"Round {round_num} queries: {new_queries}")
            return new_queries
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            self._emit(phase="warning", message=f"Query generation failed: {e}")
            return []

    # ------------------------------------------------------------------
    # SEARCH + EXTRACT
    # ------------------------------------------------------------------
    async def _search_and_extract(self, queries: List[str],
                                  question: str) -> List[Dict]:
        """Search each query and extract relevant info from top results."""
        all_findings: List[Dict] = []

        # Search all queries in parallel
        search_tasks = [self._search(q) for q in queries]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect URLs to fetch from all search results
        urls_to_fetch = []
        for result in search_results:
            if isinstance(result, Exception):
                logger.warning(f"Search error: {result}")
                continue
            if not result:
                continue
            for r in result:
                url = r.get("url", "")
                if url and url not in self.urls_fetched:
                    urls_to_fetch.append(r)
                    self.urls_fetched.add(url)
                    self.analyzed_urls.append({
                        "url": url,
                        "title": r.get("title", "") or url,
                    })
                if len(urls_to_fetch) >= self.max_urls_per_round * len(queries):
                    break

        if self._cancelled or self._time_exceeded():
            return all_findings

        # Fetch and extract URLs with backpressure. Local model servers often
        # serialize requests behind one GPU; flooding them makes every request
        # slower and can trip the extraction timeout.
        semaphore = asyncio.Semaphore(self.extraction_concurrency)

        async def _bounded_extract(result: Dict) -> Optional[Dict]:
            async with semaphore:
                return await self._fetch_and_extract(result["url"], question, result.get("title", ""))

        extract_tasks = [_bounded_extract(r) for r in urls_to_fetch]
        results_gathered = await asyncio.gather(*extract_tasks, return_exceptions=True)

        for result in results_gathered:
            if isinstance(result, Exception):
                logger.warning(f"Extraction error: {result}")
                continue
            if result:
                all_findings.append(result)

        return all_findings

    async def _search(self, query: str) -> List[Dict]:
        """Run a search query using the configured research search provider."""
        try:
            from src.search.providers import _get_search_settings
            from src.search.core import _call_provider, _build_provider_chain

            settings = _get_search_settings()
            provider = (self.search_provider_override or "").strip()
            if not provider:
                provider = (settings.get("research_search_provider") or "").strip()
            if not provider:
                provider = settings.get("search_provider", "searxng")

            if provider == "disabled":
                logger.info("Search is disabled for research")
                return []

            # Try primary provider, then fallbacks
            chain = _build_provider_chain(provider)
            raised = False
            for prov in chain:
                try:
                    results = await asyncio.to_thread(_call_provider, prov, query, 10)
                    if results:
                        logger.info(f"Research search: {prov} returned {len(results)} results")
                        if prov not in self.providers_used:
                            self.providers_used.append(prov)
                        return results
                except Exception as e:
                    raised = True
                    logger.warning(f"Research search: {prov} failed: {e}")
                    self._last_search_error = f"{prov}: {e}"
            # Every provider ran but none returned results. If none of them
            # raised, record an actionable reason here — otherwise this empty
            # path leaves `_last_search_error` unset and the caller surfaces a
            # bare "unknown error" (issue #344). This is exactly the SearXNG
            # case where the service is reachable but all its engines fail, so
            # each provider returns [] without throwing.
            if not raised:
                self._last_search_error = (
                    f"no results from search provider(s): "
                    f"{', '.join(chain) if chain else provider}"
                )
            return []
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            self._last_search_error = str(e)
            return []

    async def _fetch_and_extract(self, url: str, question: str,
                                 title: str) -> Optional[Dict]:
        """Fetch a URL's content and use LLM to extract relevant info."""
        display = title or url
        self._emit(phase="reading", url=url, title=display,
                   total_sources=len(self.urls_fetched))
        try:
            from src.search import fetch_webpage_content
            page = await asyncio.to_thread(fetch_webpage_content, url, 10)
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

        if not page.get("success") or not page.get("content"):
            return None

        content = page["content"]
        extraction_tokens = self._output_budget(self.extraction_max_tokens)
        max_chars = self.max_content_chars
        fit = self._embed_budget_chars(extraction_tokens)
        if fit is not None:
            max_chars = min(max_chars, fit)
        # Truncate to avoid blowing up context, preferring paragraph boundary
        if len(content) > max_chars:
            truncated = content[:max_chars]
            last_para = truncated.rfind('\n\n')
            if last_para > max_chars * 0.8:
                content = truncated[:last_para]
            else:
                content = truncated

        try:
            response = await self._llm(
                [
                    {"role": "user", "content": EXTRACTOR_SYSTEM.format(goal=question)},
                    untrusted_context_message("webpage", content),
                ],
                temperature=0.2,
                max_tokens=extraction_tokens,
                timeout=self.extraction_timeout,
                think=self.mechanical_think,
                response_schema=EXTRACTION_SCHEMA,
            )
            parsed = self._parse_json_object(response) or self._salvage_extraction(response)
            if parsed:
                parsed["url"] = url
                parsed["title"] = title or page.get("title", "")
                parsed["og_image"] = page.get("og_image", "")
                # Skip findings where the LLM says the page is useless
                if is_low_quality(parsed.get("summary", "")):
                    logger.info(f"Skipping low-quality extraction from {url}")
                    return None
                return parsed
            # If JSON parsing fails, treat entire response as evidence
            return {
                "url": url,
                "title": title or page.get("title", ""),
                "og_image": page.get("og_image", ""),
                "rational": "LLM extraction (raw)",
                "evidence": response[:3000],
                "summary": response[:500],
            }
        except Exception as e:
            logger.warning(f"LLM extraction failed for {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # SYNTHESIZE
    # ------------------------------------------------------------------
    async def _synthesize(self, question: str, findings: List[Dict],
                          current_report: str) -> str:
        """LLM synthesizes all findings into an updated report."""
        # Format findings for the prompt
        window = findings[-self.synthesis_window:]
        if len(findings) > self.synthesis_window:
            logger.info(f"Synthesis using last {self.synthesis_window} of {len(findings)} findings")
        max_tokens = self._output_budget(self.synthesis_max_tokens or self.max_report_tokens)
        budget = self._embed_budget_chars(max_tokens)
        findings_text = self._format_findings(
            window,
            max_chars=None if budget is None else int(budget * _FINDINGS_BUDGET_FRACTION),
        )
        report_budget = None if budget is None else max(0, budget - len(findings_text))

        prompt = SYNTHESIZE_PROMPT.format(
            question=question,
            report=self._condense_report(current_report, report_budget) or "(First round — no report yet.)",
            new_findings=findings_text,
        )

        try:
            return await self._llm(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                # Synthesis is a heavy generation call like the final report
                # (which gets 180s); a slow local model (e.g. a 20B served from
                # LM Studio) routinely needs >60s for it. The old 60s cap timed
                # out mid-stream and discarded the round's findings (#1551).
                timeout=180,
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            self._emit(phase="warning", message="Synthesis failed, keeping previous report")
            return current_report  # keep the old report on failure

    # ------------------------------------------------------------------
    # DECIDE
    # ------------------------------------------------------------------
    async def _should_stop(self, question: str, report: str,
                           round_num: int) -> bool:
        """Let the LLM decide whether the report is comprehensive enough."""
        rounds_line, guidance = (
            (_ROUNDS_AUTO, _ROUNDS_AUTO_GUIDANCE) if self.auto_rounds
            else (_ROUNDS_EXPLICIT, _ROUNDS_EXPLICIT_GUIDANCE)
        )
        prompt = STOP_PROMPT.format(
            question=question,
            report=self._condense_report(report, self._embed_budget_chars(128)),
            rounds_line=rounds_line.format(round_num=round_num, max_rounds=self.max_rounds),
            rounds_guidance=guidance,
        )

        try:
            response = await self._llm(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=128,
                think=self.mechanical_think,
                response_schema=STOP_SCHEMA,
            )
            # Reasoning models prepend a <think>...</think> block — strip it
            # before checking for YES/NO, otherwise the answer always looks
            # like it starts with "<THINK>" and the engine never stops.
            clean = strip_thinking(response).strip()
            # Structured output: {"answer": "YES"|"NO", "reason": "..."}.
            parsed = self._parse_json_object(clean) if clean.startswith(("{", "`")) else None
            if parsed and isinstance(parsed.get("answer"), str):
                clean = f"{parsed['answer']} — {parsed.get('reason', '')}".strip(" —")
            # Tolerate "**YES**", "Yes.", quotes, etc.
            answer = re.sub(r'^[\s*_`"\'>#\-]+', '', clean).upper()
            should_stop = answer.startswith("YES")
            logger.info(f"Stop decision (round {round_num}): {clean[:120]}")
            return should_stop
        except Exception as e:
            logger.warning(f"Stop decision failed: {e}")
            return False  # continue on error

    # ------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------
    async def _final_report(self, question: str, report: str) -> str:
        """LLM writes a polished final report, retrying if too short."""
        max_tokens = self._output_budget(self.max_report_tokens)
        prompt = FINAL_REPORT_PROMPT.format(
            question=question,
            report=self._condense_report(report, self._embed_budget_chars(max_tokens)),
            min_words=self.report_min_words,
        )
        cat_extra = CATEGORY_PROMPTS.get(self.category or "", "")
        if cat_extra:
            prompt += "\n\n" + cat_extra

        try:
            result = await self._llm(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=180,
            )

            # If report is too short, ask the LLM to expand it — unless the
            # follow-up (prompt + draft + new report) would overflow the window.
            fits = (not self.context_window or estimate_tokens(
                [{"content": prompt}, {"content": result}]) + max_tokens + _PROMPT_OVERHEAD_TOKENS
                <= self.context_window)
            if len(result.split()) < self.expand_below_words and fits:
                logger.info(f"Final report too short ({len(result.split())} words), requesting expansion")
                self._emit(phase="writing", message="Expanding report...")
                expanded = await self._llm(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": result},
                        {"role": "user", "content":
                            "This report is too brief. Please expand it significantly:\n"
                            "- Add detailed paragraphs for each section (not just bullet points)\n"
                            "- Include specific data, numbers, and comparisons from the evidence\n"
                            "- Explain context and significance — don't just list facts\n"
                            "- Use ## headings and ### subheadings\n"
                            f"- Target at least {self.report_min_words * 2 // 3} words\n"
                            "Write the full expanded report now."
                        },
                    ],
                    temperature=0.4,
                    max_tokens=max_tokens,
                    timeout=180,
                )
                if len(expanded.split()) > len(result.split()):
                    return expanded

            return result
        except Exception as e:
            logger.error(f"Final report generation failed: {e}")
            return report  # return the evolving report as-is

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _emit(self, **kwargs):
        """Send a progress event via the callback, if one is registered."""
        if self._progress:
            try:
                self._progress(kwargs)
            except Exception:
                pass

    def _time_exceeded(self) -> bool:
        return (time.time() - self._start_time) > self.max_time

    # _strip_think_tags removed — use research_utils.strip_thinking()

    @staticmethod
    def _strip_code_block(text: str) -> str:
        """Strip markdown code-block fences (```json ... ```) if present."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        return text.strip()

    def _parse_json_array(self, text: str) -> List[str]:
        """Extract a JSON array of strings from LLM output."""
        text = self._strip_code_block(text)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass

        # Handle truncated arrays — e.g. '["query one", "query two", "query thr'
        # Repair from the LAST array start so an echoed example array earlier
        # in the reply is not harvested into the real query set.
        last_start = text.rfind('[')
        truncated = last_start != -1 and ']' not in text[last_start:]
        if truncated:
            complete_items = re.findall(r'"([^"]*)"', text[last_start:])
            if complete_items:
                logger.info(f"Repaired truncated JSON array: recovered {len(complete_items)} items")
                return complete_items

        # Greedy match to capture the full outermost array
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass

        # Multiple complete arrays in one reply (e.g. the model echoes the
        # prompt's Example: [...] before the real array). The greedy match
        # above spans them all and fails to parse, so scan non-greedily and
        # keep the LAST parseable array, which is the model's actual answer.
        last_parsed = None
        for m in re.finditer(r'\[[\s\S]*?\]', text):
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    last_parsed = parsed
            except json.JSONDecodeError:
                continue
        if last_parsed is not None:
            return [str(item) for item in last_parsed]

        # Last resort: harvest quoted strings from the first array start
        arr_start = text.find('[')
        if arr_start != -1:
            fragment = text[arr_start:]
            # Find the last complete quoted string
            complete_items = re.findall(r'"([^"]*)"', fragment)
            if complete_items:
                logger.info(f"Repaired truncated JSON array: recovered {len(complete_items)} items")
                return complete_items

        logger.warning(f"Could not parse JSON array from: {text[:200]}")
        return []

    def _parse_json_object(self, text: str) -> Optional[Dict]:
        """Extract a JSON object from LLM output."""
        text = self._strip_code_block(text)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Greedy match to capture the full outermost object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _salvage_extraction(text: str) -> Optional[Dict]:
        """Recover extraction fields from a JSON object cut off by max_tokens.

        A length-capped structured reply stops mid-string; keep the fields
        that were (partially) written instead of storing raw JSON as evidence.
        """
        fields = {}
        for key in ("rational", "evidence", "summary"):
            m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)' % key, text or "")
            if m:
                try:
                    fields[key] = json.loads(f'"{m.group(1)}"')
                except json.JSONDecodeError:
                    fields[key] = m.group(1)
        if not (fields.get("summary") or fields.get("evidence")):
            return None
        if not fields.get("summary"):
            fields["summary"] = fields["evidence"][:500]
        fields.setdefault("evidence", fields["summary"])
        fields.setdefault("rational", "LLM extraction (truncated)")
        return fields

    def _format_findings(self, findings: List[Dict], max_chars: Optional[int] = None) -> str:
        """Format findings list into readable text for synthesis prompt.

        ``max_chars`` bounds the whole block by shortening each finding's
        content evenly (links and titles are kept).
        """
        per_finding = None
        if max_chars is not None and findings:
            per_finding = max(200, max_chars // len(findings) - 120)
        parts = []
        for i, f in enumerate(findings, 1):
            url = f.get("url", "unknown")
            title = f.get("title", "")
            summary = f.get("summary", "")
            evidence = f.get("evidence", "")
            # Use summary if available, fall back to truncated evidence
            content = summary if summary else (evidence[:1000] if evidence else "(no content)")
            if per_finding is not None and len(content) > per_finding:
                content = self._cut(content, per_finding) + " …"
            parts.append(f"**Finding {i}** — [{title}]({url})\n{content}")
        return "\n\n".join(parts)

    def _fallback_report(self, question: str, findings: List[Dict]) -> str:
        """Compile gathered findings into a basic report.

        Used when the LLM synthesis step produced no report (e.g. it timed out)
        but the search rounds did collect findings — so the user still gets the
        material that was gathered instead of "No information could be gathered"
        (#1551).
        """
        return (
            f"# {question}\n\n"
            "_Automatic synthesis did not complete, so this report lists the "
            f"{len(findings)} finding(s) gathered during research._\n\n"
            f"{self._format_findings(findings)}"
        )

    def get_stats(self) -> Dict:
        """Return research statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        stats = {
            "Duration": f"{elapsed:.1f}s",
            "Rounds": self.round_count,
            "Queries": len(self.queries_used),
            "URLs": len(self.urls_fetched),
            "Model": self.llm_model,
        }
        if self.providers_used:
            stats["Search"] = ", ".join(self.providers_used)
        if self.category:
            stats["Category"] = self.category.capitalize()
        if self.preset_label:
            stats["Preset"] = self.preset_label
        return stats
