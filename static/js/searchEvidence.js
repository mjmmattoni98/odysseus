export function sourceStatusLabel(source) {
  if (source.read_status === 'read') return source.partial ? 'Partial page text' : 'Page text retrieved';
  if (source.read_status === 'failed') return source.snippet_available === false ? 'Page unavailable' : 'Page unavailable; snippet only';
  if (source.read_status === 'snippet') return 'Search snippet only';
  return '';
}

export function searchStatusText(report) {
  if (report.state === 'disabled') return 'Web search is disabled.';
  if (report.state === 'failed') return 'Web search failed. Current information could not be verified.';
  if (report.state === 'empty') return 'Web search returned no usable results.';
  const results = Number(report.results) || 0;
  const pages = Number(report.pages_read) || 0;
  let text = `Search found ${results} sources; retrieved text from ${pages} pages.`;
  if (report.fallback) text += ` Used fallback provider ${report.provider || ''}.`;
  if (report.pages_failed) text += ` ${Number(report.pages_failed)} pages could not be read.`;
  return text;
}
