# src/research_sources.py
"""Source rules, usable-source selection, and citation checks for Deep Research.

``SourceRules`` holds the per-run allowed/excluded domains and the
official-sources-only switch. The research engine consults it when it picks
search results, before it fetches a page, and when the final source list is
built; ``infer_source_rules`` fills it from unambiguous wording in the
question ("site:nasa.gov", "only official NASA sources", "no Wikipedia").

``usable_findings`` is the single definition of which findings become the
report's sources, so the saved source list, the source-coverage counters,
and the rendered evidence profile all count the same population.

``verify_citations`` compares the links a finished report cites with the
pages that were actually read, so a report cannot silently cite a page that
was never opened, failed to load, or violates the run's source rules.
"""
from __future__ import annotations

import hashlib
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.research_navigator import is_official_host
from src.research_utils import is_low_quality

MAX_DOMAIN_RULES = 20

_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_DOMAIN_TOKEN = r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}"
_DOMAIN_LIST = rf"{_DOMAIN_TOKEN}(?:\s*(?:,|\band\b|\bor\b|/)\s*{_DOMAIN_TOKEN})*"
_SOURCE_NOUNS = (
    r"(?:sources?|websites?|sites?|pages?|data|documents?|documentation|"
    r"publications?|materials?|references?|results?)"
)
# Upper-case words that are not organisations; "only US sources" must not
# become a rule that keeps hosts with a "us" label.
_NON_ORG_ACRONYMS = {
    "AI", "API", "CPU", "CSV", "EU", "EUR", "FAQ", "GPU", "HR", "HTML", "IT",
    "JSON", "LLM", "ML", "OK", "PC", "PDF", "PR", "QA", "RSS", "SEO", "SQL",
    "TV", "UI", "UK", "URL", "US", "USA", "USD", "UX",
}
# Sites people commonly exclude by name. A name is only unambiguous when it
# is a well-known site, so other words after "avoid"/"no" are left alone.
_EXCLUDABLE_SITE_NAMES = {
    "wikipedia", "reddit", "quora", "youtube", "pinterest", "wikihow",
}
_UTM_PARAM_RE = re.compile(r"^(?:utm_[a-z]+|fbclid|gclid|mc_cid|mc_eid)$", re.IGNORECASE)


def normalize_domain(raw: Any) -> str:
    """Normalize a user-supplied domain rule; raise ``ValueError`` if invalid.

    Accepts ``nasa.gov``, ``https://www.nasa.gov/mars``, ``site:nasa.gov`` or
    ``*.nasa.gov`` (all -> ``nasa.gov``). A single label such as ``nasa`` or
    ``wikipedia`` is kept as a label rule that matches any host containing
    that label (``science.nasa.gov``, ``en.wikipedia.org``).
    """
    text = str(raw or "").strip().lower()
    original = text
    text = re.sub(r"^-?site:", "", text)
    if "://" in text:
        text = urllib.parse.urlparse(text).hostname or ""
    else:
        text = re.split(r"[/?#]", text, maxsplit=1)[0]
        text = text.rsplit("@", 1)[-1]
        if text.count(":") == 1:
            text = text.split(":", 1)[0]
    text = text.strip().lstrip("*").strip(".")
    if text.startswith("www."):
        text = text[4:]
    try:
        text = text.encode("idna").decode("ascii") if text else ""
    except UnicodeError:
        raise ValueError(f"Invalid domain: {original or raw!r}")
    labels = text.split(".") if text else []
    if (not labels or len(text) > 253
            or not all(_LABEL_RE.fullmatch(label) for label in labels)):
        raise ValueError(f"Invalid domain: {original or raw!r}")
    return text


def normalize_domain_list(values: Any, *, field_name: str = "domains") -> Tuple[str, ...]:
    """Normalize a list (or comma/space separated string) of domain rules."""
    if values is None:
        return ()
    if isinstance(values, str):
        items = re.split(r"[\s,;]+", values)
    elif isinstance(values, (list, tuple, set)):
        items = []
        for value in values:
            items.extend(re.split(r"[\s,;]+", str(value or "")))
    else:
        raise ValueError(f"{field_name} must be a list of domains")
    out: List[str] = []
    for item in items:
        if not item.strip():
            continue
        domain = normalize_domain(item)
        if domain not in out:
            out.append(domain)
    if len(out) > MAX_DOMAIN_RULES:
        raise ValueError(f"Too many {field_name}: at most {MAX_DOMAIN_RULES} are allowed")
    return tuple(out)


def url_host(url: Any) -> str:
    try:
        host = (urllib.parse.urlparse(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def domain_matches(host: str, rule: str) -> bool:
    """True when ``host`` is ``rule`` or a subdomain of it (label rules match any label)."""
    host = str(host or "").lower()
    if not host or not rule:
        return False
    if "." in rule:
        return host == rule or host.endswith("." + rule)
    return rule in host.split(".")


def _rule_covers(excluded: str, allowed: str) -> bool:
    """True when every host the ``allowed`` rule admits is also ``excluded``."""
    if "." in allowed:
        return domain_matches(allowed, excluded)
    # A label rule ("nasa") admits hosts no dotted exclusion can cover.
    return excluded == allowed


def _allowed_after_exclusions(allowed: Iterable[str], excluded: Iterable[str]) -> Tuple[str, ...]:
    """Drop allowed rules that exclusions fully cover, but never empty the list.

    An empty allow-list means "anywhere", so excluding every allowed domain
    must keep the restriction (exclusions still win at check time, and the
    rules then permit nothing) instead of silently lifting it.
    """
    allowed = tuple(allowed)
    excluded = tuple(excluded)
    remaining = tuple(a for a in allowed if not any(_rule_covers(e, a) for e in excluded))
    return remaining or allowed


@dataclass(frozen=True)
class SourceRules:
    """Per-run source restrictions."""

    allowed_domains: Tuple[str, ...] = ()
    excluded_domains: Tuple[str, ...] = ()
    official_only: bool = False
    # Human-readable notes on which rules came from the question's wording.
    inferred: Tuple[str, ...] = field(default=(), compare=False)

    @property
    def active(self) -> bool:
        return bool(self.allowed_domains or self.excluded_domains or self.official_only)

    @property
    def permits_nothing(self) -> bool:
        """True when the exclusions rule out every allowed domain."""
        return bool(self.allowed_domains) and all(
            any(_rule_covers(e, a) for e in self.excluded_domains) for a in self.allowed_domains
        )

    def violation(self, url: Any) -> str:
        """Return why ``url`` breaks these rules, or ``""`` when it is allowed."""
        if not self.active:
            return ""
        host = url_host(url)
        if not host:
            return "not a web page URL"
        for rule in self.excluded_domains:
            if domain_matches(host, rule):
                return f"excluded domain {rule}"
        if self.allowed_domains and not any(domain_matches(host, rule) for rule in self.allowed_domains):
            return "outside allowed domains " + ", ".join(self.allowed_domains)
        if self.official_only:
            # A domain the user listed explicitly counts as official for
            # this run. Otherwise only the hostname decides, matched on label
            # boundaries against official-domain patterns: never the path,
            # title or search snippet, and never a substring of the host.
            explicit = any("." in rule and domain_matches(host, rule) for rule in self.allowed_domains)
            if not explicit and not is_official_host(host):
                return "not an official source"
        return ""

    def allows(self, url: Any) -> bool:
        return not self.violation(url)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "allowed_domains": list(self.allowed_domains),
            "excluded_domains": list(self.excluded_domains),
            "official_only": bool(self.official_only),
        }
        if self.inferred:
            data["inferred"] = list(self.inferred)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "SourceRules":
        """Lenient loader for saved rules; invalid entries are dropped."""
        if isinstance(data, SourceRules):
            return data
        if not isinstance(data, dict):
            return cls()

        def _clean(values) -> Tuple[str, ...]:
            out: List[str] = []
            for value in values if isinstance(values, (list, tuple)) else []:
                try:
                    domain = normalize_domain(value)
                except ValueError:
                    continue
                if domain not in out:
                    out.append(domain)
            return tuple(out[:MAX_DOMAIN_RULES])

        inferred = data.get("inferred")
        return cls(
            allowed_domains=_clean(data.get("allowed_domains")),
            excluded_domains=_clean(data.get("excluded_domains")),
            official_only=bool(data.get("official_only")),
            inferred=tuple(str(x) for x in inferred) if isinstance(inferred, list) else (),
        )

    def describe(self) -> str:
        """Short human-readable summary for reports and the panel."""
        parts = []
        if self.allowed_domains:
            parts.append("only " + ", ".join(self.allowed_domains))
        if self.official_only:
            parts.append("official sources only")
        if self.excluded_domains:
            parts.append("excluding " + ", ".join(self.excluded_domains))
        return "; ".join(parts)

    def prompt_text(self) -> str:
        """Instructions for the planner and writer prompts."""
        if not self.active:
            return ""
        lines = ["**Source rules for this run (pages outside them are discarded):**"]
        dotted = [d for d in self.allowed_domains if "." in d]
        if self.allowed_domains:
            lines.append("- Only use pages on: " + ", ".join(self.allowed_domains))
        if dotted:
            lines.append("- Search within them, e.g. " + " / ".join(f"site:{d}" for d in dotted[:3]))
        if self.official_only:
            lines.append(
                "- Only use official sources: government, military, academic or intergovernmental "
                "sites (.gov, .mil, .edu, .int, gov.uk-style domains, europa.eu)"
                + (" or the domains listed above" if dotted else "")
            )
        if self.excluded_domains:
            lines.append("- Never use: " + ", ".join(self.excluded_domains))
        return "\n".join(lines)


def _domains_in(text: str) -> List[str]:
    out = []
    for token in re.findall(_DOMAIN_TOKEN, text.lower()):
        try:
            domain = normalize_domain(token)
        except ValueError:
            continue
        if domain not in out:
            out.append(domain)
    return out


def infer_source_rules(question: Any) -> SourceRules:
    """Read unambiguous source restrictions from the question's wording."""
    text = str(question or "")
    lower = text.lower()
    allowed: List[str] = []
    excluded: List[str] = []
    notes: List[str] = []
    official = False

    def _add(target: List[str], values: Iterable[str], note: str) -> None:
        added = [v for v in values if v and v not in target]
        target.extend(added)
        if added:
            notes.append(note.format(", ".join(added)))

    # Search operators: "-site:x" excludes, "site:x" restricts.
    _add(excluded, _domains_in(" ".join(re.findall(rf"(?<!\S)-site:\s*({_DOMAIN_TOKEN})", lower))),
         "excluded {} (from -site:)")
    _add(allowed, _domains_in(" ".join(re.findall(rf"(?<![\w-])site:\s*({_DOMAIN_TOKEN})", lower))),
         "allowed {} (from site:)")

    # Domain names with explicit restriction wording.
    only_domains = re.findall(
        rf"\bonly\s+(?:use\s+|using\s+|cite\s+|from\s+|on\s+)?(?:the\s+)?({_DOMAIN_LIST})", lower)
    only_domains += re.findall(rf"\b({_DOMAIN_LIST})\s+(?:{_SOURCE_NOUNS}\s+)?(?:only|exclusively)\b", lower)
    only_domains += re.findall(
        rf"\b(?:restrict(?:ed)?|limit(?:ed)?)\s+(?:it\s+|research\s+|sources\s+)?to\s+({_DOMAIN_LIST})", lower)
    _add(allowed, _domains_in(" ".join(only_domains)), "allowed {} (from the question)")
    no_domains = re.findall(
        rf"\b(?:exclude|excluding|avoid|without|except|skip|ignore|not\s+from|no)\s+"
        rf"(?:using\s+|sources\s+from\s+)?({_DOMAIN_LIST})", lower)
    _add(excluded, _domains_in(" ".join(no_domains)), "excluded {} (from the question)")

    # Well-known site names: "no Wikipedia", "avoid Reddit sources".
    names = re.findall(
        r"\b(?:exclude|excluding|avoid|without|except|skip|ignore|not\s+from|not|no)\s+(?:using\s+)?"
        r"(" + "|".join(sorted(_EXCLUDABLE_SITE_NAMES)) + r")\b",
        lower,
    )
    _add(excluded, names, "excluded {} (from the question)")

    # Organisation acronyms: "only NASA sources", "using only official NOAA data",
    # "from NASA only". Matching is case-sensitive on the acronym so ordinary
    # words ("only recent sources") never become rules.
    org_patterns = (
        rf"\b(?i:only|exclusively)\s+(?i:use\s+|using\s+|cite\s+|from\s+)?(?i:the\s+)?(?i:official\s+)?"
        rf"([A-Z][A-Z0-9&]{{1,9}})(?:'s)?\s+(?i:official\s+)?(?i:{_SOURCE_NOUNS})\b",
        r"\b(?i:only\s+from)\s+(?:the\s+)?([A-Z][A-Z0-9&]{1,9})\b",
        rf"\b(?i:from)\s+([A-Z][A-Z0-9&]{{1,9}})(?:\s+(?i:{_SOURCE_NOUNS}))?\s+(?i:only)\b",
    )
    orgs: List[str] = []
    for pattern in org_patterns:
        for acronym in re.findall(pattern, text):
            label = acronym.replace("&", "").lower()
            if acronym in _NON_ORG_ACRONYMS or not _LABEL_RE.fullmatch(label):
                continue
            if label not in orgs:
                orgs.append(label)
    _add(allowed, orgs, "allowed hosts named {} (from the question)")

    # Require a source noun so "the only official language" is not a rule.
    source_noun = r"(?:sources?|websites?|sites?|pages?|documentation|publications?|references?)"
    if re.search(
        r"\b(?:only|exclusively)\s+(?:use\s+|using\s+|cite\s+|from\s+)?(?:the\s+)?"
        r"(?:official|primary|first[- ]party|authoritative|government)\s+"
        rf"(?:[\w.&'-]+\s+){{0,3}}?{source_noun}\b"
        rf"|\b(?:official|primary|first[- ]party)[- ]{source_noun}[- ](?:only|exclusively)\b",
        lower,
    ):
        official = True
        notes.append("official sources only (from the question)")

    excluded = excluded[:MAX_DOMAIN_RULES]
    return SourceRules(
        allowed_domains=_allowed_after_exclusions(allowed, excluded)[:MAX_DOMAIN_RULES],
        excluded_domains=tuple(excluded),
        official_only=official,
        inferred=tuple(notes),
    )


def effective_source_rules(explicit: Any, question: Any) -> SourceRules:
    """Combine launch-time rules with rules inferred from the question.

    Explicit allowed domains replace inferred ones (the user chose them
    directly); exclusions and the official-only switch accumulate. Exclusions
    never turn a non-empty allow-list into "anywhere": when they cover every
    allowed domain the result ``permits_nothing``.
    """
    rules = SourceRules.from_dict(explicit) if not isinstance(explicit, SourceRules) else explicit
    inferred = infer_source_rules(question)
    allowed = rules.allowed_domains or inferred.allowed_domains
    excluded = tuple(dict.fromkeys([*rules.excluded_domains, *inferred.excluded_domains]))[:MAX_DOMAIN_RULES]
    notes = list(rules.inferred)
    for note in inferred.inferred:
        if note.startswith("allowed") and rules.allowed_domains:
            continue
        if note not in notes:
            notes.append(note)
    return SourceRules(
        allowed_domains=_allowed_after_exclusions(allowed, excluded)[:MAX_DOMAIN_RULES],
        excluded_domains=excluded,
        official_only=rules.official_only or inferred.official_only,
        inferred=tuple(notes),
    )


def finding_allowed(finding: Dict[str, Any], rules: SourceRules) -> bool:
    """A finding obeys the rules at its URL and where its redirects ended."""
    final_url = finding.get("final_url")
    return rules.allows(finding.get("url")) and (not final_url or rules.allows(final_url))


def usable_findings(findings: Any, rules: Optional[SourceRules] = None) -> List[Dict[str, Any]]:
    """Findings that become report sources: one per URL, readable, and allowed."""
    seen = set()
    usable: List[Dict[str, Any]] = []
    for finding in findings or []:
        if not isinstance(finding, dict):
            continue
        url = finding.get("url", "")
        summary = finding.get("summary", "") or finding.get("evidence", "")
        if not url or url in seen or is_low_quality(summary):
            continue
        if rules is not None and not finding_allowed(finding, rules):
            continue
        seen.add(url)
        usable.append(finding)
    return usable


def content_hash(text: Any) -> str:
    """Stable fingerprint of readable page text (whitespace-insensitive)."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def canonical_url(url: Any) -> str:
    """Comparable URL form: no fragment, tracking params, www. or trailing slash."""
    raw = str(url or "").strip()
    try:
        parsed = urllib.parse.urlsplit(raw)
        # ``port`` raises for malformed ports ("host:bad", "host:99999");
        # such a URL cannot be canonicalized and is compared verbatim.
        port = parsed.port
    except ValueError:
        return raw
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return raw
    host = parsed.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "")
    path = path.rstrip("/")
    query = urllib.parse.urlencode([
        (k, v) for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not _UTM_PARAM_RE.match(k)
    ])
    return urllib.parse.urlunsplit(("https", host, path, query, ""))


_MD_LINK_RE = re.compile(r"(!?)\[[^\]]*\]\(\s*<?(https?://[^\s)>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_BARE_URL_RE = re.compile(r"(?<![(\[<\"'=])https?://[^\s<>\"'\])]+")


def cited_urls(report_markdown: Any) -> List[str]:
    """Links a report cites, in order of first appearance (images excluded)."""
    text = str(report_markdown or "")
    # Fenced code is not prose; a URL in an example block is not a citation.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    urls: List[str] = []
    for match in _MD_LINK_RE.finditer(text):
        if match.group(1):
            continue
        urls.append(match.group(2))
    stripped = _MD_LINK_RE.sub(" ", text)
    urls.extend(_BARE_URL_RE.findall(stripped))
    out: List[str] = []
    seen = set()
    for url in urls:
        url = url.rstrip(".,;:!?*_")
        key = canonical_url(url)
        if key not in seen:
            seen.add(key)
            out.append(url)
    return out


def verify_citations(
    report_markdown: Any,
    *,
    read_urls: Iterable[str],
    attempted_urls: Iterable[str] = (),
    rules: Optional[SourceRules] = None,
) -> List[Dict[str, str]]:
    """Warnings for cited links that were not actually read during research.

    ``read_urls`` are pages whose text was extracted into the report's
    sources; ``attempted_urls`` are pages that were opened but yielded no
    usable text. Every cited link outside ``read_urls`` gets one warning.
    """
    read = {canonical_url(u) for u in read_urls or [] if u}
    attempted = {canonical_url(u) for u in attempted_urls or [] if u}
    warnings: List[Dict[str, str]] = []
    for url in cited_urls(report_markdown):
        key = canonical_url(url)
        violation = rules.violation(url) if rules is not None else ""
        if violation:
            warnings.append({
                "url": url, "status": "excluded",
                "reason": f"Cites a source outside this run's source rules ({violation}).",
            })
        elif key in read:
            continue
        elif key in attempted:
            warnings.append({
                "url": url, "status": "unread",
                "reason": "Cited, but the page was opened without readable or useful text.",
            })
        else:
            warnings.append({
                "url": url, "status": "not_opened",
                "reason": "Cited, but the page was never opened during research.",
            })
    return warnings
