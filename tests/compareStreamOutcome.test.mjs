// Compare pane failure handling: provider stream errors and empty replies
// must fail the pane, and failed panes must drop out of voting. These feed
// the exact SSE text /api/chat_stream sends through the reader stream.js uses.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  consumePaneStream, createPaneStreamReader, paneResponseState, searchResponseState, voteAvailability,
  EMPTY_RESPONSE_MESSAGE, PANE_FAILED, PANE_OK, PANE_PARTIAL, PANE_PENDING,
} from '../static/js/compare/streamOutcome.js';

/** Run raw SSE text through a reader line by line, as stream.js does. */
function readStream(sse, { awaitingChoice = false } = {}) {
  const reader = createPaneStreamReader();
  const payloads = [];
  for (const line of sse.split('\n')) {
    const json = reader.read(line);
    if (reader.ended) break;
    if (json) payloads.push(json);
  }
  return { reader, payloads, failure: reader.failure({ awaitingChoice }) };
}

// What the QA instance logged for clef-flash:latest (Ollama HTTP 400).
const OLLAMA_REJECTION =
  'event: error\n'
  + 'data: {"status": 400, "text": "Ollama returned HTTP 400: \\"clef-flash:latest\\" does not support chat"}\n\n'
  + 'data: [DONE]\n\n';

test('a named error event fails the pane with the provider text', () => {
  const { reader, payloads, failure } = readStream(OLLAMA_REJECTION);
  assert.equal(reader.ended, true);
  assert.deepEqual(payloads, [], 'the error payload must not reach the renderer');
  assert.ok(failure, 'stream must not pass for a clean completion');
  assert.equal(failure.status, 400);
  assert.match(failure.message, /does not support chat/);
  assert.equal(failure.terminalStreamError, true);
  assert.equal(paneResponseState({ streamOk: false, awaitingChoice: false, aborted: false, sawOutput: reader.sawOutput }), PANE_FAILED);
});

test('error payload shapes from the backend all normalize to a message', () => {
  const cases = [
    ['event: error\ndata: {"error": "Read timeout", "status": 504}\n', /Read timeout/],
    ['event: error\ndata: {"error": {"message": "quota exceeded"}}\n', /quota exceeded/],
    ['data: {"type": "error", "text": "tool schema rejected"}\n', /tool schema rejected/],
    ['data: {"status": 502, "text": "Upstream protocol error"}\n', /Upstream protocol error/],
    ['event: error\ndata: upstream exploded\n', /upstream exploded/],
  ];
  for (const [sse, expected] of cases) {
    const { failure } = readStream(sse);
    assert.ok(failure, sse);
    assert.match(failure.message, expected, sse);
  }
});

test('an error after partial text still fails the pane', () => {
  const { reader, payloads, failure } = readStream(
    'data: {"delta": "The answer"}\n\nevent: error\ndata: {"status": 503, "text": "runner crashed"}\n\n',
  );
  assert.deepEqual(payloads, [{ delta: 'The answer' }]);
  assert.equal(reader.sawOutput, true);
  assert.match(failure.message, /runner crashed/);
});

test('a cleanly closed stream with no content is a failure, not a success', () => {
  for (const sse of [
    '',
    'data: {"type": "metrics", "data": {"output_tokens": 0}}\n\ndata: [DONE]\n\n',
    'data: {"delta": "\\n  "}\n\ndata: [DONE]\n\n',
  ]) {
    const { reader, failure } = readStream(sse);
    assert.ok(failure, JSON.stringify(sse));
    assert.equal(failure.message, EMPTY_RESPONSE_MESSAGE);
    assert.equal(paneResponseState({ streamOk: false, awaitingChoice: false, aborted: false, sawOutput: reader.sawOutput }), PANE_FAILED);
  }
});

test('a normal answer succeeds and keeps rendering after DONE', () => {
  const { payloads, failure } = readStream(
    'data: {"delta": "42"}\n\ndata: [DONE]\n\ndata: {"type": "metrics", "data": {"output_tokens": 1}}\n\n',
  );
  assert.equal(failure, null);
  assert.deepEqual(payloads.map(p => p.delta || p.type), ['42', 'metrics']);
  assert.equal(paneResponseState({ streamOk: true, awaitingChoice: false, aborted: false, sawOutput: true }), PANE_OK);
});

test('tool activity and pending questions are not empty replies', () => {
  assert.equal(readStream('data: {"type": "tool_start", "tool": "bash"}\n').failure, null);
  const asked = readStream('data: {"type": "ask_user", "data": {}}\n', { awaitingChoice: true });
  assert.equal(asked.failure, null);
  assert.equal(paneResponseState({ streamOk: false, awaitingChoice: true, aborted: false, sawOutput: true }), PANE_PENDING);
});

test('stopping a pane after some output is partial, an empty stop is failed', () => {
  assert.equal(paneResponseState({ streamOk: false, awaitingChoice: false, aborted: true, sawOutput: true }), PANE_PARTIAL);
  assert.equal(paneResponseState({ streamOk: false, awaitingChoice: false, aborted: true, sawOutput: false }), PANE_FAILED);
});

test('a failed pane can never win or tie, and one survivor cannot be voted on', () => {
  // The audit case: Gemma answered, Clef failed.
  const twoPanes = voteAvailability([PANE_OK, PANE_FAILED]);
  assert.deepEqual(twoPanes.panes, [false, false]);
  assert.equal(twoPanes.tie, false);
  assert.deepEqual(twoPanes.contestants, [true, false]);
  assert.match(twoPanes.reasons[1], /failed/);

  const threePanes = voteAvailability([PANE_OK, PANE_FAILED, PANE_OK]);
  assert.deepEqual(threePanes.panes, [true, false, true]);
  assert.equal(threePanes.tie, true);
});

test('only a pane that finished with content is votable: pending, partial and unstamped are not', () => {
  // Review finding: Tie was recordable before either model answered.
  for (const states of [
    [PANE_PENDING, PANE_PENDING],
    [PANE_OK, PANE_PENDING],
    [PANE_OK, PANE_PARTIAL],
    [PANE_OK, ''],          // a new round's bubble before its stream starts
    ['', ''],
  ]) {
    const avail = voteAvailability(states);
    assert.equal(avail.tie, false, JSON.stringify(states));
    assert.deepEqual(avail.panes, [false, false], JSON.stringify(states));
  }
  assert.match(voteAvailability([PANE_OK, PANE_PARTIAL]).reasons[1], /stopped before it finished/);
  assert.equal(voteAvailability([PANE_OK, PANE_PENDING]).reasons[1], '', 'a pane still answering is not blamed');
});

test('vote and tie open as soon as the second pane finishes', () => {
  assert.equal(voteAvailability([PANE_OK, PANE_PENDING]).tie, false);
  const done = voteAvailability([PANE_OK, PANE_OK]);
  assert.equal(done.tie, true);
  assert.deepEqual(done.panes, [true, true]);
});

test('early voting needs two finished panes and records only those as contestants', () => {
  const avail = voteAvailability([PANE_OK, PANE_PENDING, PANE_OK, PANE_PARTIAL]);
  assert.deepEqual(avail.panes, [true, false, true, false]);
  assert.equal(avail.tie, true);
  assert.deepEqual(avail.contestants, [true, false, true, false]);
});

test('nothing is votable before a prompt or after a vote, and contestants survive the vote', () => {
  assert.equal(voteAvailability([PANE_OK, PANE_OK], { hasPrompt: false }).tie, false);
  const after = voteAvailability([PANE_OK, PANE_OK, PANE_FAILED], { voted: true });
  assert.deepEqual(after.panes, [false, false, false]);
  assert.equal(after.tie, false);
  // _saveVote reads contestants after marking the round voted.
  assert.deepEqual(after.contestants, [true, true, false]);
});

test('a search pane is answered when its provider returned results', () => {
  assert.equal(searchResponseState({ results: [{ title: 't', url: 'https://example.com' }] }), PANE_OK);
  assert.equal(searchResponseState({ results: [] }), PANE_FAILED, 'no results is nothing to vote on');
  assert.equal(searchResponseState({ results: [], error: 'Stopped' }), PANE_FAILED);
  assert.equal(searchResponseState({ results: [{ title: 't' }], error: 'quota' }), PANE_FAILED);
  assert.equal(searchResponseState(null), PANE_FAILED);
});

// ── Search-mode summary stream (_runSynthForPane reads through consumePaneStream) ──

/** A response body delivered in arbitrary network chunks (split mid-line). */
function chunkedBody(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

test('summary stream: a named error event split across chunks fails the summary', async () => {
  const seen = [];
  const reader = await consumePaneStream(chunkedBody([
    'data: {"delta": "Partial sum"}\n\nevent: err',
    'or\ndata: {"status": 400, "text": "Ollama returned HTTP 400: \\"clef-flash:latest\\" does no',
    't support chat"}\n\ndata: {"delta": "never rendered"}\n\n',
  ]), json => seen.push(json));
  assert.deepEqual(seen, [{ delta: 'Partial sum' }], 'nothing after the error reaches the renderer');
  const failure = reader.failure();
  assert.equal(failure.status, 400);
  assert.match(failure.message, /does not support chat/);
});

test('summary stream: an empty summary is a failure', async () => {
  const reader = await consumePaneStream(chunkedBody(['data: {"type": "metrics", "data": {}}\n\n', 'data: [DONE]\n\n']), () => {});
  assert.equal(reader.failure().message, EMPTY_RESPONSE_MESSAGE);
});

test('summary stream: a summary split mid-line still renders in full', async () => {
  let text = '';
  const reader = await consumePaneStream(chunkedBody(['data: {"del', 'ta": "Both providers "}\n\ndata: {"delta": "agree."}']), json => {
    text += json.delta || '';
  });
  assert.equal(text, 'Both providers agree.', 'a final line without a trailing newline is still read');
  assert.equal(reader.failure(), null);
});
