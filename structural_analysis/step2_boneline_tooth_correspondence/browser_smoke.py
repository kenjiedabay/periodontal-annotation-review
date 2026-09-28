"""Exercise review controls in headless Edge using isolated synthetic data."""
import json
from pathlib import Path
import shutil
import tempfile
import threading

from correspondence import HERE, empty_review
from serve import make_server
from test_correspondence import fixture
from PIL import Image
from playwright.sync_api import sync_playwright


def run():
    data = fixture()
    record = data['records'][0]
    record.update(dimensions=[100, 100], metadata={'Arch': 'Synthetic test'}, issues=[],
                  original_url='review_overlays/test/original.jpg')
    record['bone_lines'][0]['points'] = [[20, 50], [80, 50]]
    for tooth in record['teeth']:
        tooth['overlay_url'] = f'review_overlays/test/{tooth["tooth_instance_id"]}.png'
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        (folder / 'reports').mkdir()
        assets = folder / 'review_overlays' / 'test'
        assets.mkdir(parents=True)
        Image.new('RGB', (100, 100), '#444444').save(assets / 'original.jpg')
        for tooth in record['teeth']:
            Image.new('RGBA', (100, 100), (0, 150, 255, 70)).save(assets / f'{tooth["tooth_instance_id"]}.png')
        shutil.copyfile(HERE / 'index.html', folder / 'index.html')
        (folder / 'candidate_mappings.json').write_text(json.dumps(data))
        (folder / 'expert_review.json').write_text(json.dumps(empty_review(data)))
        server = make_server(folder, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        errors = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless=True)
                page = browser.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function("document.querySelector('#images').options.length===1")
                page.locator('#reviewer').fill('Synthetic UI test')
                page.locator('[data-status="confirmed"]').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 1')")
                page.reload()
                page.wait_for_function("document.querySelector('#counts').textContent.includes('confirmed: 1')")
                page.locator('#reviewer').fill('Synthetic UI test')
                page.locator('#showmask').uncheck()
                assert not page.locator('#mask').is_visible()
                page.locator('#showmask').check()
                assert page.locator('#mask').is_visible()
                page.locator('#tooth').select_option('mask2')
                page.locator('[data-status="uncertain"]').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 2')")
                page.locator('#replace').select_option('Validation_1|B0|T:mask1')
                page.locator('#tooth').select_option('region')
                page.locator('#region').fill('Synthetic interdental region')
                page.locator('#reassign').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 3')")
                current = json.loads((folder / 'expert_review.json').read_text())
                assert any(d.get('region') == 'Synthetic interdental region' and d['status'] == 'confirmed' for d in current['decisions'])
                assert any(d['tooth_instance_id'] == 'mask1' and d['status'] == 'rejected' for d in current['decisions'])
                page.locator('#bone').select_option('1')
                assert page.locator('[data-status="confirmed"]').is_disabled()
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print('PASS: browser accept, reload persistence, layer toggle, uncertain, region reassignment, invalid-line blocking; no JavaScript errors. Real expert reviews untouched.')


if __name__ == '__main__':
    run()
