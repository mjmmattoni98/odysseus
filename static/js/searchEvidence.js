export function sourceStatusLabel(source) {
  if (source.read_status === 'read') return source.partial ? 'Page excerpt retrieved' : 'Page text retrieved';
  if (source.read_status === 'failed') {
    if (source.snippet_available === false) return 'Page text unavailable';
    return source.failure_reason === 'access_restricted' ? 'Search snippet · site restricted access' : 'Search snippet · page text unavailable';
  }
  if (source.read_status === 'snippet') return 'Search snippet only';
  return '';
}

export function searchStatusText(report) {
  if (report.state === 'reused') return 'Sources from the previous answer · no new search';
  if (report.state === 'decision') {
    if (report.needs_search === true) return report.mode === 'always' ? 'Web: Always — searching before answering.' : 'Web: Auto — search needed.';
    if (report.needs_search === false) return report.mode === 'off' ? 'Answered without search · Web: Off' : 'Answered without search';
    return 'Auto could not decide; checking your request for a web lookup.';
  }
  if (report.state === 'disabled') return 'Web search is disabled.';
  if (report.state === 'failed') return 'Web search failed. Current information could not be verified.';
  if (report.state === 'empty') return 'Web search returned no usable results.';
  const results = Number(report.results) || 0;
  const pages = Number(report.pages_read) || 0;
  if (results > 0 && pages === 0) return `Found ${results} sources. Only search snippets were available; full page text could not be retrieved.`;
  let text = `Searched the web: found ${results} sources; retrieved text from ${pages} pages.`;
  if (report.fallback) text += ` Used fallback provider ${report.provider || ''}.`;
  if (report.retrieval_recovery) text += ' Found alternative readable sources.';
  if (report.pages_failed) {
    const failed = Number(report.pages_failed);
    text += ` ${failed} ${failed === 1 ? 'source has' : 'sources have'} snippets only because page text was unavailable.`;
  }
  return text;
}
