**Odysseus webapp audit, 12 October 2026**

The main workflows work, but model eligibility and Compare error handling need attention before treating the app as fully reliable. The existing release smoke suite passed all 16 checks across 12 areas. Live browser testing found failures that the deterministic provider in that suite does not exercise.

**Environment and evidence**

- Checkout: `dev`, commit `981535245cb61c5345211c88a454fdd7dce6451f`.
- Test instance: `http://127.0.0.1:7022`, container `odysseus-qa-20261012`.
- Fresh, separate data, database, logs, and ChromaDB under `/tmp/odysseus-qa-20261012/`. The personal instance on port 7000 was preserved.
- Current sources were mounted into the QA instance. Matching SHA-256 hashes were verified for `app.py` and `static/js/compare/stream.js`.
- The existing local Docker image supplied Python dependencies. This checks current application code with those installed dependencies, not a fresh Docker image build.
- Real local Ollama and the running SearXNG service were used for live chat, search, research, and comparison.
- Browser checks used the T3 collaborative browser at 1280 × 800 and 375 × 667, with the Europe/Madrid timezone.
- Evidence: `/tmp/odysseus-qa-20261012/smoke.log`, `checks.json`, `checks.log`, `data/qa-smoke.xml`, and `qa-document.docx`.
- The original port-7000 service has different application and requirements hashes from this checkout. Its behavior should not be used to assess the current source tree until it is rebuilt.

**What worked**

| Area | Automated evidence | Browser or live evidence |
| --- | --- | --- |
| Login and settings | A preference survived a separate login | Admin login, endpoint creation, and conversation settings worked |
| Chat | Buffered reply, streamed reply, and saved history passed | Qwen answered arithmetic correctly; history and settings survived reload |
| Cancellation | Additional audit check read the persisted cancelled turn | Stop worked; the next prompt returned `17` without remaining stuck |
| Web search | Outside the deterministic smoke suite | Five sources found, three pages retrieved; sources and fallback evidence survived reload |
| Deep Research | Additional audit check read the saved report | A real one-round run completed in 133.1 seconds, and its HTML report opened |
| Compare | Two stub replies, voting, and reveal passed | Gemma answered `42`; the incompatible model failed silently in its pane, detailed below |
| Notes | Create, read, edit, and delete passed | A note created through the UI persisted and opened for editing on mobile |
| Calendar | Create, list, and delete passed | An event created through the UI appeared in the calendar and API |
| Tasks | Create, list, and pause passed | The Tasks panel loaded its built-in tasks |
| Documents | Create, version, and restore passed | Rich-text edits saved; HTML export contained the text; the downloaded DOCX was a valid ZIP containing the edited text |
| Document indexing | Upload, chunk, index, and list passed | Semantic question answering over a document was not exercised |
| Memory | Add, search, and delete passed | A fact added through the UI persisted; chat displayed a recalled-memory count |
| Gallery | Separate live API checks verified upload and persisted rotation | Image displayed; editor canvas loaded the rotated 180 × 320 image |
| Cookbook | Hardware detection, fresh-catalog guidance, and saved state passed | The hardware/catalog panel loaded; actual download and model provisioning were not tested |
| Email | Fixture inbox, opening, and mark-read passed | An unconfigured inbox showed setup guidance; real mail delivery was not tested |

Search engines returned rate limits or CAPTCHAs during the real search. The fallback provider `searxng_yep` returned results, and the chat UI correctly explained the fallback, partial page text, snippet-only sources, and failed reads.

**Fixes, in priority order**

1. **Reject incompatible models when choosing defaults and probing readiness.** Adding the local Ollama endpoint to a fresh instance automatically selected `clef-flash:latest`. Its advertised capabilities were `decision` and `vision`; Ollama returned HTTP 400 with `does not support chat`. The app's `/api/probe-selected` nevertheless returned `status: ok`, and Compare displayed "All ready!". The current filtering excludes embedding-only models but does not reject other known non-chat models. Filter by actual chat capability, preserve a suitable default, and keep an explicit unknown state when capabilities cannot be determined. Relevant code: `routes/model_routes.py:682`, `:815`, and `:2466`.

2. **Show provider stream errors in Compare and prevent voting on failed responses.** Compare Gemma 4 12B with Clef Flash, submit "What is 17 + 25? Reply with the number only." Gemma returns `42`; Clef produces an empty answer pane. The server logged `event: error` with a `status: 400` and error text, but the pane showed no error. Vote and Tie remained usable; clicking Tie revealed both models as a tie. The stream parser ignores named error events and does not handle the associated error payload, then marks a cleanly closed stream as successful. Normalize provider errors, mark the pane failed, offer a model switch or retry, and exclude failed panes from voting. Relevant code: `static/js/compare/stream.js:489`, `:703`, and `static/js/compare/vote.js`.

3. **Correct Gallery UTC timestamps.** A photo uploaded at `2026-10-11T23:59:55.397856` was inspected at 02:00:28 in Madrid. The inspector displayed 23:59:55 on the previous date and "2 hours ago", although the photo was about 33 seconds old. UTC database timestamps are serialized without an offset, and the browser parses them as local time. Apply timezone-aware serialization to creation/update timestamps and editor-draft timestamps; preserve the separate semantics of EXIF capture dates. Relevant code: `routes/gallery/gallery_helpers.py:120` and `static/js/gallery.js:1523`. The same incorrect age appeared on the saved editor project card.

4. **Make research source-quality counts consistent.** The saved research data contained three official NASA sources and one secondary source. `source_coverage.primary_or_official` was 3, but the rendered report said "0 Primary sources". Its counter includes only `source_kind == 'primary'`. Include official sources in the displayed first-party evidence count, or show separate, clearly labeled primary and official counts. Relevant code: `src/visual_report.py:306` and `:2904`.

5. **Update the composer's accessible name with its action.** During generation, the button's title changed to "Stop generation", but its `aria-label` remained "New chat". The same control also changes between sending, queueing, and starting a new chat. Update the accessible name alongside the visual state. Relevant code: `static/js/chat.js:1376` and `static/index.html:1233`. Cancellation itself worked.

6. **Refresh the Gallery inspector after a rotation.** Rotating the uploaded 320 × 180 fixture persisted a correct 180 × 320 image. The displayed inspector still said "320 x 180 · 4.3 KB". The handler updates the image URL and library but leaves the detail metadata unchanged. Reconcile the detail object and inspector from the rotation response or a refreshed image record. Relevant code: `static/js/gallery.js:1995`.

7. **Expose the public auth policy to the login page.** The logged-out login page requests `/api/auth/policy`, which returns 401. The route describes its data as public policy, but it is missing from the middleware's exemptions. Login succeeds using fallback constants; setup/signup cannot reliably load the server policy. Exempt this specific public route. Separately, avoid redundant logged-out calls to protected theme-preference endpoints. Relevant code: `app.py:270`, `routes/auth_routes.py:228`, and `static/login.html:432`.

The first two fixes deserve failure-path regression coverage. A passing happy-path comparison with a stub provider did not catch either defect.

**Improvements to existing behavior**

- Honor answer-length requests in Deep Research. The request explicitly asked for a brief summary, but the report was 1,810 words, labeled a nine-minute read, and took 133.1 seconds. Expose a short-report option at launch and let explicit length instructions override the default report target.
- Simplify the everyday workspace. Let users pin the few tools they use most and put advanced tools behind an expandable section. Use a readable default density and stronger contrast for muted text; the current default uses small monospace text extensively.
- Give mobile sheets a visible Back or Close action. Notes can be dismissed through Escape or a gesture, but the mobile header hides the desktop close/minimize control. A visible action would make navigation easier to discover.
- Explain overnight calendar ranges explicitly. A 12:00 to 11:00 range became an event ending the following day; the list showed only "12:00 – 11:00". Show the end date or an overnight marker.
- In Cookbook, distinguish task type and runtime compatibility from memory fit. The catalog contains chat, embedding, speech, vision, and time-series entries. A "PERFECT" memory-fit label alone should not imply that a model is a suitable chat model or can run through the displayed engine. Compatibility was not tested by downloading these entries.
- Keep deployment instructions and version labels consistent. The README/Compose default is port 7011 while the setup guide still tells Docker users to open port 7000. The login page showed build 0.20.19 while Settings showed application 1.0.3 and build 0.20.19. Show one clear version/build/source identity.

**Useful additions**

These prioritize the previously stated preference for ordinary chat, web research, and help making decisions.

- A decision notebook with options, constraints, weighted criteria, tradeoffs, source links, a saved choice, and a date to revisit it. It should preserve why an option won so later changes can be evaluated against the same criteria.
- Source rules for research: allowed or excluded domains, official-source-only mode, and warnings when the final report cites a page that was not actually read. The current run read Wikipedia and retained it in the source list despite the NASA-only request; the report's prose said its conclusions used the official pages.
- Saved research refreshes with a change summary: revisit a saved question, identify which evidence changed, and preserve the previous answer rather than replacing it.
- A visible local inference queue showing active and waiting jobs, which model they require, and likely memory/context pressure. The existing timing and context controls can feed this view.
- Scheduled backups with a guided restore and a verification check covering the database and file-backed workspace state. Keep this distinct from merely downloading an export.

**Limits and rerunning the checks**

This audit does not establish that every provider, operating system, or integration works. Real IMAP/SMTP, CalDAV sync, OAuth/2FA, audio capture/STT/TTS, AI image generation/editing, arbitrary MCP integrations, and Cookbook download/serve provisioning were not tested. The QA instance disabled scheduled execution and email pollers; task configuration and fixture email behavior were checked, not real scheduled delivery. RAG indexing passed, but an actual document-grounded answer was not tested.

The read-only source mount caused `/api/fonts/custom` to fail when it tried to create a directory under `static/fonts/custom`. This is a test-environment limitation and is excluded from the confirmed deployment bugs above. Later T3 screenshot calls failed and the desktop preview disconnected; later checks used live DOM/API and saved-artifact evidence. Browser-tool errors and aborted completed streams were not counted as application defects.

The audit-specific check script is [check_20261012.py](check_20261012.py). It targets only port 7022, reads the QA account from the private scratch directory, and never prints credentials. It validates this run's fixtures and saved exports; it is not a general fresh-install test harness. Its exit code is 1 while confirmed diagnostic issues remain. The short-report recommendation is recorded separately as an improvement.

```bash
python3 docs/qa/check_20261012.py
```

The smoke runner adaptation lives at `/tmp/odysseus-qa-20261012/run_smoke.py`. It binds the stub provider to the container's bridge address, because the application translates loopback provider URLs to the Docker host. The underlying repository smoke scenarios are unchanged.

```bash
docker exec odysseus-qa-20261012 python /tmp/run_smoke.py
```

The test app and isolated ChromaDB are left running for inspection. Their launch helper is `/tmp/odysseus-qa-20261012/launch.py`. No application code was changed, committed, or pushed.
