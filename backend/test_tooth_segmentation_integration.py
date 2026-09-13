"""Run after frontend build: real checkpoint + held-out images + headless Edge.

Writes responses/screenshots/verification to artifacts/tooth-segmentation.
"""
import base64
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
import uvicorn

from main import app
from app.tooth_segmentation import ROOT, ToothSegmentationService, service
from app.tooth_instance_baseline.data import Record, ToothInstanceDataset
from app.preprocessing.transforms import apply_transforms
from app.preprocessing.config import load_config


def run():
    output = ROOT / 'artifacts/tooth-segmentation'
    output.mkdir(parents=True, exist_ok=True)
    api = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=8000, log_level='warning'))
    threading.Thread(target=api.run, daemon=True).start()
    web = ThreadingHTTPServer(('127.0.0.1', 5174), functools.partial(SimpleHTTPRequestHandler, directory=str(ROOT / 'frontend/dist')))
    threading.Thread(target=web.serve_forever, daemon=True).start()
    while not api.started:
        time.sleep(.1)
    report = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1100})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            for split, image_id, folder in [('Validation', '1002', 'Validation/Images'), ('Testing', '1', 'Images')]:
                source = ROOT / 'DenPAR Radiographs Dataset/Dataset' / folder / f'{image_id}.jpg'
                masks = tuple(sorted((ROOT / 'DenPAR Radiographs Dataset/Dataset' / split / 'Masks (Tooth-wise)' / image_id).glob('*.png')))
                hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, *masks)}
                original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                h, w = original.shape[:2]
                # Exact equality to the tensor produced by the validated training loader.
                expected, _ = ToothInstanceDataset([Record(image_id, source, masks, w, h)])[0]
                processed, _, _ = apply_transforms(original, load_config())
                assert np.array_equal(expected[0].numpy(), processed.astype(np.float32) / 255.0)
                page.goto('http://127.0.0.1:5174')
                page.locator('input[type=file]').first.set_input_files(str(source))
                with page.expect_response(lambda r: '/analysis/tooth-segmentation?' in r.url and r.status == 200, timeout=180000) as response:
                    page.get_by_role('button', name='Tooth segmentation', exact=True).click()
                result = response.value.json()
                assert result['model_status'] == 'available' and result['instances']
                assert result['ground_truth']['split'] == split
                page.get_by_test_id('segmentation-viewer').wait_for()
                assert page.get_by_test_id('prediction-mask').count() == len(result['instances'])
                for index, instance in enumerate(result['instances']):
                    assert .5 <= instance['confidence'] <= 1
                    assert page.get_by_test_id('confidence-label').nth(index).text_content() == f"{instance['instance_id']}: {instance['confidence']*100:.2f}%"
                    box = page.get_by_test_id('prediction-box').nth(index)
                    x1, y1, x2, y2 = instance['bbox']
                    assert float(box.get_attribute('x')) == x1
                    assert float(box.get_attribute('y')) == y1
                    assert abs(float(box.get_attribute('width')) - (x2-x1)) < 1e-6
                    png = cv2.imdecode(np.frombuffer(base64.b64decode(instance['mask_url'].split(',')[1]), np.uint8), cv2.IMREAD_UNCHANGED)
                    assert png.shape == (h, w, 4)
                    yy, xx = np.where(png[:, :, 3] > 0)
                    assert xx.min() >= x1-3 and xx.max() <= x2+3 and yy.min() >= y1-3 and yy.max() <= y2+3
                for viewport in (1440, 760):
                    page.set_viewport_size({'width': viewport, 'height': 1100})
                    geometry = page.get_by_test_id('segmentation-viewer').evaluate('''svg => {
                      const rect = svg.getBoundingClientRect(), matrix = svg.getScreenCTM();
                      return { width: rect.width, height: rect.height, sx: matrix.a, sy: matrix.d };
                    }''')
                    assert abs(geometry['width']/geometry['height'] - w/h) < .01
                    assert abs(geometry['sx'] - geometry['sy']) < 1e-6
                page.set_viewport_size({'width': 1440, 'height': 1100})
                page.screenshot(path=str(output / f'{split}-prediction.png'), full_page=True)
                page.get_by_label('Show bounding boxes').uncheck()
                assert page.get_by_test_id('prediction-box').count() == 0
                page.get_by_label('Show confidence labels').uncheck()
                assert page.get_by_test_id('confidence-label').count() == 0
                page.get_by_label('Show masks').uncheck()
                assert page.get_by_test_id('prediction-mask').count() == 0
                page.get_by_label('Show masks').check()
                page.get_by_label('Comparison mode').select_option('Ground Truth')
                assert page.get_by_test_id('prediction-mask').count() == 0
                assert page.get_by_test_id('truth-mask').count() == len(masks)
                page.get_by_label('Comparison mode').select_option('Overlay Comparison')
                assert page.get_by_test_id('prediction-mask').count() == len(result['instances'])
                page.screenshot(path=str(output / f'{split}-comparison.png'), full_page=True)
                assert hashes == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hashes}
                (output / f'{split}-response.json').write_text(json.dumps(result), encoding='utf-8')
                report.append(dict(split=split, image_id=image_id, instances=len(result['instances']), source_size=[w,h], preprocessing_exact=True, coordinates_and_confidence_verified=True, viewports=[1440,760], toggles_and_comparison_verified=True, source_hashes_unchanged=True))
            assert not errors, errors
            browser.close()
        with TestClient(app) as client:
            assert client.post('/analysis/tooth-segmentation?image_id=invalid', files={'file': ('bad.jpg', b'invalid')}).status_code == 422
            saved = service.model
            service.model = None
            service.attempted = True
            unavailable = client.post('/analysis/tooth-segmentation?image_id=1', files={'file': ('1.jpg', source.read_bytes())}).json()
            service.model = saved
            assert unavailable['model_status'] == 'unavailable' and unavailable['instances'] == []
        (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        api.should_exit = True
        web.shutdown()


if __name__ == '__main__':
    run()

