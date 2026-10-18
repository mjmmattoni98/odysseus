#!/usr/bin/env python3
"""Non-destructive checks against this run's isolated QA instance and artifacts."""
import datetime as dt
import argparse
import http.cookiejar
import json
import re
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts-dir', type=Path, default=Path('/tmp/odysseus-qa-20261012'))
OUT = parser.parse_args().artifacts_dir
BASE = 'http://127.0.0.1:7022'
rows = []
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

def request(path, body=None, authenticated=True):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={'Content-Type': 'application/json'} if data else {})
    handle = opener.open if authenticated else urllib.request.urlopen
    try:
        with handle(req, timeout=15) as response:
            raw = response.read().decode()
            try:
                value = json.loads(raw)
            except ValueError:
                value = raw
            return response.status, value
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()

def record(name, passed, evidence, kind='check'):
    rows.append({'name': name, 'result': 'PASS' if passed else 'ISSUE', 'kind': kind, 'evidence': evidence})

status, body = request('/api/auth/policy', authenticated=False)
record('Public login policy is available before login', status == 200, {'http_status': status}, 'finding')
status, body = request('/api/auth/login', json.loads((OUT / 'account.json').read_text()))
assert status == 200, 'QA login failed'
status, body = request('/api/ready')
record('Current checkout is ready', status == 200 and body.get('ready'), {'http_status': status, 'ready': body.get('ready')})

status, endpoints = request('/api/model-endpoints')
endpoints = endpoints if isinstance(endpoints, list) else endpoints.get('endpoints', [])
local = next(e for e in endpoints if '11434' in e.get('base_url', ''))
record('Chat list excludes known non-chat model', 'clef-flash:latest' not in local.get('models', []),
       {'models': local.get('models', []), 'non_chat_model': 'clef-flash:latest'}, 'finding')
status, probe = request('/api/probe-selected', {'models': [{'model': 'clef-flash:latest', 'endpoint_id': local['id']}]})
assert status == 200, 'QA probe route failed'
record('Probe rejects a model without chat capability', (probe.get('results') or [{}])[0].get('status') != 'ok',
       {'http_status': status, 'probe': probe}, 'finding')
native_request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
    data=json.dumps({'model': 'clef-flash:latest', 'messages': [{'role': 'user', 'content': 'QA capability check.'}],
                     'stream': False, 'options': {'num_predict': 1}}).encode(),
    headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(native_request, timeout=15) as response:
        native_status, native_body = response.status, response.read().decode()
except urllib.error.HTTPError as error:
    native_status, native_body = error.code, error.read().decode()
record('Model accepted by app probe can actually chat', native_status == 200,
       {'ollama_http_status': native_status, 'response': native_body[:400]}, 'finding')

status, photo = request('/api/gallery/0caed2bc-9a81-43cd-ad8a-ba8717dad211')
stamp = photo.get('created_at', '')
parsed = dt.datetime.fromisoformat(stamp) if stamp else None
record('Gallery timestamp contains its UTC offset', parsed is not None and parsed.tzinfo is not None,
       {'created_at': stamp, 'browser_timezone': 'Europe/Madrid', 'observed_ui': '2 hours ago for a newly uploaded image'}, 'finding')
record('Image rotation persisted dimensions', [photo.get('width'), photo.get('height')] == [180, 320],
       {'dimensions': [photo.get('width'), photo.get('height')]})

status, document = request('/api/document/f7dcbcdf-a38b-4f7a-9b9c-361cfcef4315')
record('Document saved browser edits', status == 200 and 'This text must survive reloading and exporting.' in document.get('current_content', ''),
       {'http_status': status, 'versions': document.get('version_count')})
with zipfile.ZipFile(OUT / 'qa-document.docx') as archive:
    xml = archive.read('word/document.xml').decode()
    record('Downloaded DOCX is valid and contains the edited text', archive.testzip() is None and 'This text must survive reloading and exporting.' in xml,
           {'bytes': (OUT / 'qa-document.docx').stat().st_size})

report = json.loads((OUT / 'data/deep_research/rp-f4202c23768b.json').read_text())
record('Live research completed and saved a report', report.get('status') == 'done' and bool(report.get('result')),
       {'stats': report.get('stats'), 'words': len(report.get('result', '').split()), 'sources': len(report.get('sources') or [])})
status, rendered = request('/api/research/report/rp-f4202c23768b')
match = re.search(r'class="evidence-value">(\d+)</span>\s*<span class="evidence-label">Primary sources', rendered)
rendered_primary = int(match.group(1)) if match else None
expected_primary = report.get('source_coverage', {}).get('primary_or_official', 0)
record('Research report preserves official-source counts', rendered_primary is not None and rendered_primary >= expected_primary,
       {'primary_or_official': report.get('source_coverage', {}).get('primary_or_official'),
        'sources': [{'url': s.get('url'), 'source_kind': s.get('source_kind')} for s in report.get('sources', [])],
        'rendered_primary_count': rendered_primary}, 'finding')
record('Brief research request produces a brief report', len(report.get('result', '').split()) < 400,
       {'prompt': report.get('query'), 'words': len(report.get('result', '').split())}, 'improvement')

status, memories = request('/api/memory')
record('Memory added in the browser persisted', status == 200 and any('QA preference: use concise answers.' == m.get('text') for m in memories.get('memory', [])),
       {'http_status': status, 'matching_records': sum('QA preference: use concise answers.' == m.get('text') for m in memories.get('memory', []))})
status, history = request('/api/history/89d177c3-4684-4fb8-b2a7-74a5b98bd735')
messages = history.get('history') or []
record('Cancelled turn is saved and a subsequent turn succeeds', status == 200
       and any(m.get('metadata', {}).get('cancelled') for m in messages)
       and any(m.get('role') == 'assistant' and m.get('content') == '17' for m in messages),
       {'http_status': status, 'cancelled_turns': sum(bool(m.get('metadata', {}).get('cancelled')) for m in messages),
        'recovery_reply': next((m.get('content') for m in reversed(messages) if m.get('role') == 'assistant'), None)})

(OUT / 'checks.json').write_text(json.dumps(rows, indent=2) + '\n')
for row in rows:
    print(f"{row['result']:5} {row['name']}: {json.dumps(row['evidence'], ensure_ascii=False)}")
raise SystemExit(1 if any(row['result'] == 'ISSUE' and row['kind'] == 'finding' for row in rows) else 0)
