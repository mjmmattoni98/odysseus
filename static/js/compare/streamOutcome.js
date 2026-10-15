// compare/streamOutcome.js — pane stream outcome + vote eligibility (no DOM)
import { createTerminalStreamError } from '../chatStreamErrors.js';

// Response states stamped on a pane's AI message (data-response-state). The
// attribute travels with the message when panes are shuffled.
export const PANE_PENDING = 'pending';
export const PANE_OK = 'ok';
export const PANE_PARTIAL = 'partial';   // stopped or timed out after some output
export const PANE_FAILED = 'failed';

export const EMPTY_RESPONSE_MESSAGE = 'The model returned an empty response.';

// Records that put something in the pane even without a text delta.
const _OUTPUT_TYPES = new Set(['tool_start', 'tool_output', 'tool_approval_resolved', 'ask_user']);

/** Whether an SSE payload is a provider/stream failure rather than content. */
export function isStreamErrorPayload(eventName, json) {
  if (eventName === 'error') return true;
  if (!json || typeof json !== 'object') return false;
  if (json.type === 'error') return true;
  return Number(json.status) >= 400;
}

/**
 * Line-level reader for one compare pane's chat stream.
 *
 * read() returns the JSON payload a data line carries, or null for event
 * names, blank lines, comments and malformed data. A named `event: error`
 * (and any `type: "error"` / HTTP-status payload) ends the stream with a
 * normalized error instead of being handed to the renderer, so a provider
 * rejection can never pass for a clean, empty completion.
 */
export function createPaneStreamReader() {
  let eventName = '';
  const reader = {
    ended: false,
    error: null,
    sawOutput: false,
    read(line) {
      if (reader.ended) return null;
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
        return null;
      }
      if (!line.trim()) {
        eventName = '';
        return null;
      }
      if (!line.startsWith('data:')) return null;
      const data = line.slice(5).replace(/^ /, '');
      const name = eventName;
      eventName = '';
      // Metrics and save receipts can still follow DONE; the body's close
      // ends the stream.
      if (data === '[DONE]') return null;
      let json;
      try {
        json = JSON.parse(data);
      } catch (_) {
        if (name !== 'error') return null;
        json = { text: data };
      }
      if (isStreamErrorPayload(name, json)) {
        reader.error = createTerminalStreamError(json && typeof json === 'object' ? json : { text: String(json) });
        reader.ended = true;
        return null;
      }
      if (json && typeof json === 'object') {
        // Whitespace-only deltas leave the pane blank, so they don't count.
        if ((typeof json.delta === 'string' && json.delta.trim()) || _OUTPUT_TYPES.has(json.type)) {
          reader.sawOutput = true;
        }
      }
      return json;
    },
    /**
     * The error that ends this stream once its body closed, or null on
     * success. A clean close with no output at all is a failure too.
     */
    failure({ awaitingChoice = false } = {}) {
      if (reader.error) return reader.error;
      if (!reader.sawOutput && !awaitingChoice) return createTerminalStreamError({ text: EMPTY_RESPONSE_MESSAGE });
      return null;
    },
  };
  return reader;
}

/**
 * Read a whole chat_stream body through a pane stream reader. Lines split
 * across network chunks are reassembled; `onPayload` gets each content
 * payload. Reading stops at a provider error. Returns the reader, so the
 * caller can ask `reader.failure()` once the body has ended.
 */
export async function consumePaneStream(body, onPayload) {
  const reader = createPaneStreamReader();
  const bodyReader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const feed = (line) => {
    const json = reader.read(line);
    if (json) onPayload(json);
    return reader.ended;
  };
  while (true) {
    const { done, value } = await bodyReader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (feed(line)) {
        bodyReader.cancel().catch(() => {});
        return reader;
      }
    }
  }
  buffer += decoder.decode();
  if (buffer) feed(buffer);
  return reader;
}

/** Final response state for a pane stream that has stopped. */
export function paneResponseState({ streamOk, awaitingChoice, aborted, sawOutput }) {
  if (awaitingChoice) return PANE_PENDING;
  if (streamOk) return PANE_OK;
  if (aborted && sawOutput) return PANE_PARTIAL;
  return PANE_FAILED;
}

/** Response state for a search-provider pane: its results are its content. */
export function searchResponseState(data) {
  const ok = data && !data.error && Array.isArray(data.results) && data.results.length > 0;
  return ok ? PANE_OK : PANE_FAILED;
}

/** Why a pane's answer can never be voted on, or '' (votable, or still answering). */
export function voteBlockedReason(state) {
  if (state === PANE_FAILED) return 'This response failed and cannot be voted for';
  if (state === PANE_PARTIAL) return 'This response was stopped before it finished and cannot be voted for';
  return '';
}

/**
 * Which vote controls are usable. Only a pane that finished successfully
 * with content (PANE_OK) is a contestant: pending, partial, failed and
 * not-yet-stamped panes can't win or share a tie. Voting needs two
 * contestants. `contestants` ignores the prompt/voted gates, so the vote
 * record can name who took part after the round is marked voted.
 */
export function voteAvailability(states, { hasPrompt = true, voted = false } = {}) {
  const list = states || [];
  const contestants = list.map(s => s === PANE_OK);
  const open = Boolean(hasPrompt) && !voted && contestants.filter(Boolean).length >= 2;
  return {
    panes: contestants.map(c => open && c),
    tie: open,
    contestants,
    reasons: list.map(voteBlockedReason),
  };
}
