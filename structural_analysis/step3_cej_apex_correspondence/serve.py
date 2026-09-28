"""Local-only review UI with atomic review persistence and revision checks."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

from landmarks import HERE, ROOT, apply_review as apply_decision, write as atomic_json, report, sha, resolve_image
from cej_readiness import refresh, review_records


def make_server(directory=HERE, port=8767):
    directory = Path(directory).resolve()
    data = json.loads((directory / 'candidate_landmark_mappings.json').read_text(encoding='utf-8'))
    lock = threading.Lock()

    def review():
        value = json.loads((directory / 'expert_landmark_review.json').read_text(encoding='utf-8'))
        if value['dataset_id'] != data['dataset_id']:
            raise ValueError('Review dataset fingerprint mismatch')
        return value

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, kind='application/json'):
            if not isinstance(content, bytes):
                content = json.dumps(content, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            try:
                if route == '/api/state':
                    with lock:
                        current = review()
                        return self.send(200, {'data': data, 'review': current, 'report': report(data, current)})
                if route.startswith('/originals/'):
                    key = Path(route).stem
                    record = next((r for r in data['records'] if r['key'] == key), None)
                    if record is None or not record.get('original_path'):
                        return self.send(404, {'error': 'Unknown original image'})
                    path = resolve_image(record['split'], record['image_id'])
                    relative = str(path.relative_to(ROOT))
                    if relative != record['original_path'] or not path.exists() or sha(path) != data['source_sha256'].get(relative):
                        return self.send(409, {'error': 'Original image changed or is unavailable; rerun availability audit'})
                    return self.send(200, path.read_bytes(), 'image/jpeg')
                if route == '/api/report':
                    with lock:
                        return self.send(200, report(data, review()))
                if route == '/api/cej-records':
                    with lock:
                        current = review()
                        return self.send(200, {'candidate_dataset_id': data['dataset_id'], 'expert_review_revision': current['revision'], 'records': review_records(data, current)})
                if route == '/expert_landmark_review.json':
                    with lock:
                        return self.send(200, review())
                target = (directory / ('index.html' if route == '/' else route.lstrip('/'))).resolve()
                allowed = target == directory / 'index.html' or (
                    target.is_relative_to(directory / 'overlays') and target.suffix in {'.png', '.jpg'})
                if not allowed or not target.is_file():
                    return self.send(404, {'error': 'Not found'})
                return self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            except ValueError as exc:
                return self.send(409, {'error': str(exc)})

        def do_POST(self):
            if self.path not in {'/api/decision', '/api/regenerate'}:
                return self.send(404, {'error': 'Not found'})
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                return self.send(403, {'error': 'Cross-origin review writes are disabled'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send(415, {'error': 'JSON required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 20000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('JSON object required')
                with lock:
                    if self.path == '/api/regenerate':
                        if directory != HERE.resolve():
                            raise ValueError('Regeneration is disabled for synthetic test datasets')
                        return self.send(200, refresh())
                    current = apply_decision(data, review(), payload)
                    atomic_json(directory / 'expert_landmark_review.json', current)
                    summary = report(data, current)
                    atomic_json(directory / 'reports' / 'summary.json', summary)
                    if directory == HERE.resolve():
                        atomic_json(directory / 'cej_review_records.json', {'candidate_dataset_id': data['dataset_id'], 'expert_review_revision': current['revision'], 'records': review_records(data, current)})
                return self.send(200, {'review': current, 'report': summary})
            except (ValueError, TypeError) as exc:
                return self.send(409, {'error': str(exc)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    server = make_server(port=args.port)
    print(f'Review UI: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()
