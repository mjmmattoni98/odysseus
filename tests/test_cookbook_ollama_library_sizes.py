"""ollama.com cards: only real size tags become `<name>:<size>` rows.

The library scraper matched any `>3.7M<`-shaped text, so the download
counter on every card ("3.7M" downloads) became a fake "3.7m" size tag. The
Cookbook then built `qwen3.8:3.7m` rows and estimated their memory fit from a
"3.7M parameter" model. Cards below mirror ollama.com's search markup (current
and older x-test-* variants); sizes, names and counts are synthetic.
"""
import pytest

from routes.cookbook_helpers import parse_ollama_library_card

# Current ollama.com search card: capability + size chips, then a download
# counter whose visible text is "3.7M".
CURRENT_CARD = """
<div class="min-w-0 flex-1">
  <h2 title="talker"><span>talker</span></h2>
  <p class="mt-1">Synthetic chat model.</p>
  <div class="mt-3 flex flex-wrap items-center gap-2">
    <span class="inline-flex rounded-md bg-indigo-50 text-indigo-600">vision</span>
    <span class="inline-flex rounded-md bg-indigo-50 text-indigo-600">tools</span>
    <span class="inline-flex rounded-md bg-[#ddf4ff] text-blue-600">8b</span>
    <span class="inline-flex rounded-md bg-[#ddf4ff] text-blue-600">70b</span>
    <span class="inline-flex rounded-md bg-[#ddf4ff] text-blue-600">270m</span>
  </div>
</div>
<span class="inline-flex shrink-0 tabular-nums" title="3,720,127 downloads"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 16.5v2.25"/></svg><span >3.7M</span></span>
"""

# Older markup: x-test-* counters with trailing labels, plus a disk size.
LEGACY_CARD = """
<div>
  <span x-test-size class="chip">8x7b</span>
  <span x-test-size class="chip">0.5b</span>
  <span class="chip">4.7GB</span>
  <p class="flex space-x-5">
    <span><span x-test-pull-count>12M</span> <span>Pulls</span></span>
    <span><span x-test-tag-count>5</span> <span>Tags</span></span>
    <span>Updated <span x-test-updated>2 weeks ago</span></span>
  </p>
  <span>120m</span> <span>pulls</span>
</div>
"""


def test_download_counter_is_not_a_size_tag():
    card = parse_ollama_library_card(CURRENT_CARD)

    assert card["sizes"] == ["8b", "70b", "270m"]
    assert card["capabilities"] == ["vision", "tools"]


def test_legacy_pull_tag_counters_and_disk_sizes_are_ignored():
    card = parse_ollama_library_card(LEGACY_CARD)

    # "12M" (pulls), "5" (tags), "120m pulls" and "4.7GB" are not pullable tags.
    assert card["sizes"] == ["8x7b", "0.5b"]


@pytest.mark.parametrize("html", ["", None, "<span>", "<div><span>8b</span"])
def test_malformed_or_empty_cards_do_not_raise(html):
    card = parse_ollama_library_card(html)

    assert isinstance(card["sizes"], list)
    assert isinstance(card["capabilities"], list)


class _FakeResponse:
    status_code = 200
    text = f'<a href="/library/talker" class="group w-full">{CURRENT_CARD}</a>'


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *args, **kwargs):
        return _FakeResponse()


@pytest.mark.asyncio
async def test_library_route_returns_only_real_sizes(monkeypatch):
    import httpx
    from routes.cookbook_routes import setup_cookbook_routes

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    router = setup_cookbook_routes()
    endpoint = next(r.endpoint for r in router.routes if getattr(r, "path", "") == "/api/cookbook/ollama/library")

    payload = await endpoint(refresh=1, request=None, owner="tester")
    talker = next(m for m in payload["models"] if m["name"] == "talker")

    assert talker["sizes"] == ["8b", "70b", "270m"]
    assert "3.7m" not in talker["sizes"]
    assert talker["task"] == "vision"
