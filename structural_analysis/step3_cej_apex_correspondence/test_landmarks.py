import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from landmarks import apply_review, boundary_pixels, empty_review, evidence, identity, report
from serve import make_server


def fixture():
    r = {'key': 'Validation_1', 'image_id': '1', 'issues': [], 'teeth': [{'tooth_instance_id': t, 'issues': []} for t in ['mask1','mask2']],
         'points': [{'point_id': f'CEJ:{i}', 'point_type': 'CEJ', 'coordinate': [10+i, 10], 'issues': []} for i in range(2)]}
    candidates = [{'id': identity(r['key'], p['point_id'], 'mask1'), 'key': r['key'], 'image_id': '1', 'point_id': p['point_id'],
                   'point_type': 'CEJ', 'coordinate': p['coordinate'], 'tooth_instance_id': 'mask1', 'status': 'probable', 'candidate_score': .8} for p in r['points']]
    return {'dataset_id': 'test', 'records': [r], 'candidates': candidates}


def payload(**kwargs):
    return {'key': 'Validation_1', 'point_id': 'CEJ:0', 'tooth_instance_id': 'mask1', 'status': 'confirmed',
            'reviewer': 'Test reviewer', 'revision': 0, **kwargs}


class LandmarkTests(unittest.TestCase):
    def test_geometry_ignores_bbox_order(self):
        fg = np.zeros((30,30), dtype=bool)
        fg[8:20, 8:20] = True
        boxes = [{'bbox_id': 0, 'bbox': [0,0,5,5], 'issues': []}, {'bbox_id': 1, 'bbox': [8,8,20,20], 'issues': []}]
        a = evidence([9,9], fg, boundary_pixels(fg), [8,8,20,20], [13.5,13.5], boxes)
        b = evidence([9,9], fg, boundary_pixels(fg), [8,8,20,20], [13.5,13.5], boxes[::-1])
        self.assertEqual(a['candidate_score'], b['candidate_score'])
        self.assertEqual(a['min_mask_pixel_center_distance_px'], 0)
        self.assertEqual(a['keypoint_bbox_support'], 1)
        self.assertEqual(a['image_vertical_region'], 'upper')
        outside = evidence([25,10], fg, boundary_pixels(fg), [8,8,20,20], [13.5,13.5], [])
        self.assertEqual(outside['min_mask_pixel_center_distance_px'], 6)

    def test_multiple_cej_allowed_without_automatic_confirmation(self):
        data = fixture()
        original = json.dumps(data)
        review = empty_review(data)
        self.assertEqual(report(data,review)['counts']['confirmed_correspondence_count'],0)
        review = apply_review(data,review,payload())
        review = apply_review(data,review,payload(point_id='CEJ:1',revision=1))
        counts = report(data,review)['counts']
        self.assertEqual(counts['confirmed_correspondence_count'],2)
        self.assertEqual(counts['expert_approved_multiple_point_cases'],1)
        self.assertEqual(json.dumps(data),original)

    def test_missing_does_not_fabricate_point_and_conflicts_blocked(self):
        data = fixture()
        review = apply_review(data,empty_review(data),payload(point_id=None,point_type='Apex',status='missing'))
        self.assertIsNone(review['decisions'][0]['coordinate'])
        self.assertEqual(report(data,review)['counts']['missing_apex_count'],1)
        self.assertEqual(report(data,review)['counts']['missing_CEJ_count'],0)
        self.assertEqual(len(data['records'][0]['points']),2)
        review = apply_review(data,review,payload(revision=1))
        with self.assertRaises(ValueError):
            apply_review(data,review,payload(point_id=None,point_type='CEJ',status='missing',revision=2))
        review = apply_review(data,review,payload(point_id=None,point_type='Apex',status='uncertain',revision=2))
        self.assertEqual(report(data,review)['counts']['missing_apex_count'],0)

    def test_reassign_and_reject(self):
        data=fixture()
        review=apply_review(data,empty_review(data),payload())
        review=apply_review(data,review,payload(tooth_instance_id='mask2',revision=1,reassign_from=identity('Validation_1','CEJ:0','mask1')))
        self.assertEqual(len(review['history']),3)
        self.assertEqual(report(data,review)['counts']['confirmed_correspondence_count'],1)
        self.assertEqual(review['decisions'][0]['status'],'rejected')
        for changes in [{'revision':9},{'reviewer':''},{'point_id':'unknown'},{'tooth_instance_id':'unknown'},{'status':'missing'}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                apply_review(data,empty_review(data),payload(**changes))

    def test_http_persistence_and_stale_write(self):
        data=fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'reports').mkdir()
            (folder/'candidate_landmark_mappings.json').write_text(json.dumps(data))
            (folder/'expert_landmark_review.json').write_text(json.dumps(empty_review(data)))
            server=make_server(folder,0)
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            try:
                url=f'http://127.0.0.1:{server.server_port}'
                request=Request(url+'/api/decision',json.dumps(payload()).encode(),{'Content-Type':'application/json'})
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['review']['revision'],1)
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code,409)
                self.assertEqual(json.loads((folder/'candidate_landmark_mappings.json').read_text()),data)
                self.assertEqual(json.loads((folder/'expert_landmark_review.json').read_text())['decisions'][0]['status'],'confirmed')
            finally:
                server.shutdown();server.server_close();thread.join()


if __name__=='__main__':
    unittest.main()
