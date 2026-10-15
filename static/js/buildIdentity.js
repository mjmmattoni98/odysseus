/* ── Build identity ──
   One formatter for "which build is this?", shared by the login footer and
   the Settings sidebar footer so both pages name the running build the same
   way. The payload is /api/version, whose values come from src/constants.py:
   `version` is the public application semver, `build` the preview harness
   version (HARNESS_VERSION), and `source_commit` the commit the process
   loaded. The backend reports the string "unknown" for anything it cannot
   resolve, e.g. the commit in a read-only Docker tree with no .git.

   Deliberately free of DOM access and imports so the logged-out login page
   can load it on its own. */
const BUILD_COMMIT_SHORT_LEN = 8;
const BUILD_UNKNOWN = 'unknown';
const BUILD_COMMIT_SHA = /^[0-9a-f]{7,64}$/i;
const SEPARATOR = ' · ';

function buildField(value) {
  const text = String(value ?? '').trim();
  return text && text.toLowerCase() !== BUILD_UNKNOWN ? text : '';
}

/** Turn an /api/version payload into display strings, or null when it
 *  carries nothing worth showing.
 *
 *  `label` is the one-line identity, e.g. "v1.0.3 · build 0.20.19 · 8a28c5f9".
 *  `versionLabel` and `commitLabel` are its two halves, for layouts that put
 *  the commit on its own line; `commitTitle` is the full commit for hover. */
export function formatBuildIdentity(payload) {
  const version = buildField(payload?.version);
  const build = buildField(payload?.build);
  const commit = buildField(payload?.source_commit);

  const parts = [];
  if (version) parts.push(`v${version}`);
  // The preview harness versions itself separately from the public semver, so
  // show both — but never the same number twice.
  if (build && build !== version) parts.push(`build ${build}`);

  // An ODYSSEUS_SOURCE_COMMIT override need not be a sha; leave those intact
  // and let the CSS ellipsis deal with the width.
  const commitLabel = BUILD_COMMIT_SHA.test(commit)
    ? commit.slice(0, BUILD_COMMIT_SHORT_LEN).toLowerCase()
    : commit;

  if (!parts.length && !commitLabel) return null;

  const versionLabel = parts.join(SEPARATOR);
  return {
    label: [versionLabel, commitLabel].filter(Boolean).join(SEPARATOR),
    versionLabel,
    commitLabel,
    commitTitle: commit,
  };
}
