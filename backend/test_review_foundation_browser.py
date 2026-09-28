"""Optional browser smoke test: run after `npm.cmd run build` in frontend."""
import base64
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import time

import cv2
import numpy as np
import pytest
import uvicorn

sync_playwright = pytest.importorskip('playwright.sync_api').sync_playwright

from app import review_foundation
from app.tooth_segmentation import ROOT as PROJECT_ROOT
from main import app


def test_blind_first_browser_flow(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        monkeypatch.setattr(review_foundation, 'ROOT', work / 'records')
        pixels = np.full((320, 440), 150, dtype=np.uint8)
        image_path = work / 'stage1_case.png'
        cv2.imwrite(str(image_path), pixels)
        routine_path = work / 'routine_case.png'
        cv2.imwrite(str(routine_path), np.full((320, 440), 120, dtype=np.uint8))
        offline_path = work / 'offline_case.png'
        cv2.imwrite(str(offline_path), np.full((320, 440), 90, dtype=np.uint8))
        mask = np.zeros((320, 440, 4), dtype=np.uint8)
        mask[60:290, 140:290, :] = 255
        mask_url = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', mask)[1].tobytes()).decode('ascii')
        fake_checkpoint = work / 'checkpoint.pt'
        fake_checkpoint.write_bytes(b'browser smoke checkpoint identity')
        monkeypatch.setattr(review_foundation, 'CHECKPOINT', fake_checkpoint)

        def fake_predict(image_id, original):
            return {'image_id': image_id, 'task': 'tooth_instance_segmentation',
                    'model_status': 'available', 'model_version': 'maskrcnn_epoch8',
                    'confidence_threshold': .5, 'mask_threshold': .5,
                    'instances': [{'instance_id': 1, 'confidence': .92,
                                   'bbox': [140, 60, 290, 290], 'mask_url': mask_url}],
                    'width': original.shape[1], 'height': original.shape[0],
                    'coordinate_space': 'original_image', 'bbox_format': 'xyxy',
                    'ground_truth': None}

        monkeypatch.setattr(review_foundation.tooth_service, 'predict', fake_predict)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            api_port = listener.getsockname()[1]
        api = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=api_port, log_level='warning'))
        api_thread = threading.Thread(target=api.run, daemon=True)
        api_thread.start()
        web = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(
            SimpleHTTPRequestHandler, directory=str(PROJECT_ROOT / 'frontend' / 'dist')))
        web_thread = threading.Thread(target=web.serve_forever, daemon=True)
        web_thread.start()
        try:
            for _ in range(100):
                if api.started:
                    break
                time.sleep(.1)
            assert api.started
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                page.add_init_script(f"""(() => {{ const oldFetch = window.fetch.bind(window);
                  window.fetch = (input, init) => oldFetch(typeof input === 'string'
                    ? input.replace('http://127.0.0.1:8000', 'http://127.0.0.1:{api_port}') : input, init);
                }})()""")
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('dialog', lambda dialog: dialog.accept())
                page.goto(f'http://127.0.0.1:{web.server_address[1]}')
                assert page.get_by_label('Review mode').input_value() == 'ai_assisted'
                page.locator('input[type=file]').first.set_input_files(str(routine_path))
                page.get_by_text('AI-Assisted Review', exact=True).wait_for()
                assert page.get_by_role('button', name='Tooth segmentation').is_enabled()
                page.reload()
                page.get_by_label('Review mode').select_option('independent_evaluation')
                page.locator('input[type=file]').first.set_input_files(str(image_path))
                page.get_by_text('Independent Review', exact=True).wait_for()
                assert page.get_by_role('button', name='Tooth segmentation').is_disabled()
                assert page.get_by_role('button', name='Reveal AI after independent annotation').count() == 0
                page.get_by_label('Reviewer ID').fill('reviewer-browser')
                page.locator('.spatial-canvas canvas').first.click(position={'x': 250, 'y': 250})
                page.get_by_role('button', name='Save current').click()
                page.get_by_text('Independent annotation version 1 saved.').wait_for()
                assert page.get_by_role('button', name='Reveal AI after independent annotation').is_visible()
                page.reload()
                page.get_by_label('Review mode').select_option('independent_evaluation')
                page.locator('input[type=file]').first.set_input_files(str(image_path))
                page.get_by_text('Independent Review', exact=True).wait_for()
                assert page.get_by_role('button', name='Tooth segmentation').is_disabled()
                assert page.get_by_role('button', name='Reveal AI after independent annotation').is_visible()
                page.get_by_label('Reviewer ID').fill('reviewer-browser')
                page.get_by_role('button', name='Reveal AI after independent annotation').click()
                page.get_by_text('AI-Assisted Review', exact=True).wait_for()
                assert page.get_by_role('button', name='Tooth segmentation').is_enabled()
                assert page.get_by_text('AI-generated—awaiting expert review').count() > 0
                page.locator('.spatial-canvas canvas').first.click(position={'x': 280, 'y': 280})
                page.get_by_role('button', name='Save current').click()
                page.get_by_text('Expert correction version 1 saved.').wait_for()
                page.get_by_role('button', name='Approve', exact=True).click()
                page.get_by_text('Final decision approved saved as a new record.').wait_for()
                page.reload()
                page.get_by_label('Review mode').select_option('independent_evaluation')
                page.locator('input[type=file]').first.set_input_files(str(image_path))
                page.get_by_text('AI-Assisted Review', exact=True).wait_for()
                assert page.get_by_role('button', name='Tooth segmentation').is_enabled()
                page.route('**/api/review/sources', lambda route: route.fulfill(
                    status=503, content_type='application/json', body='{"detail":"service unavailable"}'))
                page.reload()
                page.locator('input[type=file]').first.set_input_files(str(offline_path))
                page.get_by_text('offline_case.png').wait_for()
                assert page.get_by_role('button', name='Versioned review').is_disabled()
                assert page.get_by_role('button', name='Tooth segmentation').is_disabled()
                assert page.get_by_alt_text('Original radiograph offline_case.png').is_visible()
                assert not errors, errors
                browser.close()
        finally:
            api.should_exit = True
            web.shutdown()
            api_thread.join(timeout=10)
            web_thread.join(timeout=10)
