"""Headless UI checks on temporary synthetic records, never real expert reviews."""
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from PIL import Image
from landmarks import HERE, ROOT, empty_review
from serve import make_server
from test_landmarks import fixture

sys.path.insert(0, str(ROOT / '.runtime'))
from playwright.sync_api import sync_playwright


def run():
    data=fixture()
    r=data['records'][0]
    r.update(dimensions=[100,100],metadata={},source_counts={'bboxes':1,'CEJ':2,'Apex':0},bboxes=[{'bbox_id':0,'bbox':[5,5,50,70],'issues':[]}],original_url='overlays/test/original.jpg')
    for t in r['teeth']:
        t['overlay_url']=f'overlays/test/{t["tooth_instance_id"]}.png'
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        (folder/'reports').mkdir()
        assets=folder/'overlays'/'test'
        assets.mkdir(parents=True)
        Image.new('RGB',(100,100),'#444444').save(assets/'original.jpg')
        for t in r['teeth']:
            Image.new('RGBA',(100,100),(0,150,255,70)).save(assets/f'{t["tooth_instance_id"]}.png')
        shutil.copyfile(HERE/'index.html',folder/'index.html')
        (folder/'candidate_landmark_mappings.json').write_text(json.dumps(data))
        (folder/'expert_landmark_review.json').write_text(json.dumps(empty_review(data)))
        server=make_server(folder,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        errors=[]
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
                page=browser.new_page()
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function("document.querySelector('#point').options.length===2")
                page.locator('#reviewer').fill('Synthetic test')
                page.locator('[data-status="confirmed"]').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 1')")
                page.locator('#point').select_option('CEJ:1')
                page.locator('[data-status="confirmed"]').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 2')")
                page.locator('#missingtype').select_option('Apex')
                page.locator('#missing').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 3')")
                page.reload()
                page.wait_for_function("document.querySelector('#counts').textContent.includes('missing apex count: 1')")
                page.locator('#reviewer').fill('Synthetic test')
                page.locator('#replace').select_option('Validation_1|CEJ:0|mask1')
                page.locator('#tooth').select_option('mask2')
                page.locator('#reassign').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 4')")
                page.locator('#showmask').uncheck()
                assert not page.locator('#mask').is_visible()
                page.locator('#showboxes').uncheck()
                assert page.locator('#geometry rect').count()==0
                page.locator('#tooth').select_option('mask1')
                page.locator('#missingtype').select_option('Apex')
                page.locator('#clearmissing').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 5')")
                page.locator('[data-status="cannot_determine"]').click()
                page.wait_for_function("document.querySelector('#message').textContent.includes('revision 6')")
                assert page.locator('#neighbors img').count() == 1
                page.locator('#showneighbors').uncheck()
                assert page.locator('#neighbors img').count() == 0
                stored=json.loads((folder/'expert_landmark_review.json').read_text())
                assert stored['revision']==6
                assert any(d['status']=='cannot_determine' for d in stored['decisions'])
                reassigned = next(d for d in stored['decisions'] if d['tooth_instance_id']=='mask2' and d['status']=='confirmed')
                assert reassigned['candidate_tooth_instance_id']=='mask1'
                assert reassigned['expert_tooth_instance_id']=='mask2'
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown();server.server_close();thread.join()
    print('PASS: browser confirms multiple CEJ, saves/clears missing apex, reloads reviews, reassigns with independent target IDs, records cannot-determine and toggles neighboring masks; no JavaScript errors. Real reviews untouched.')


if __name__=='__main__':
    run()
