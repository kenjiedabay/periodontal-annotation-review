import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from correspondence import apply_decision, cv2, empty_review, features, pair_id, report, sample_line
from serve import make_server


def fixture():
    key = 'Validation_1'
    record = {'key': key, 'image_id': '1', 'teeth': [{'tooth_instance_id': t, 'issues': []} for t in ['mask1', 'mask2']],
              'bone_lines': [{'bone_line_id': 0, 'issues': []}, {'bone_line_id': 1, 'issues': ['out_of_bounds']} ]}
    candidates = [{'id': pair_id(key, 0, t), 'key': key, 'image_id': '1', 'bone_line_id': 0,
                   'tooth_instance_id': t, 'status': 'probable', 'candidate_score': .8} for t in ['mask1', 'mask2']]
    return {'dataset_id': 'fixture', 'records': [record], 'candidates': candidates}


def payload(**kwargs):
    return {'key': 'Validation_1', 'bone_line_id': 0, 'tooth_instance_id': 'mask1',
            'status': 'confirmed', 'reviewer': 'Test reviewer', 'revision': 0, **kwargs}


class CorrespondenceTests(unittest.TestCase):
    def test_segment_interior_not_just_endpoints(self):
        foreground = np.zeros((30, 30), dtype=bool)
        foreground[10:20, 10:20] = True
        distance = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        f = features(sample_line([[0, 15], [29, 15]]), distance, [10, 10, 19, 19], [14.5, 14.5])
        self.assertEqual(f['min_mask_distance_px_approx'], 0)
        self.assertEqual(f['min_bbox_distance_px_approx'], 0)
        self.assertGreater(f['fraction_samples_in_expanded_mask'], 0)

    def test_multiplicity_and_explicit_confirmation(self):
        data = fixture()
        before = json.dumps(data)
        initial = empty_review(data)
        initial_report = report(data, initial)
        self.assertEqual(initial_report['counts']['confirmed'], 0)
        self.assertEqual(initial_report['counts']['ambiguous_images'], 1)
        accepted = apply_decision(data, initial, payload())
        self.assertEqual(report(data, accepted)['counts']['confirmed'], 1)
        self.assertEqual(report(data, accepted)['counts']['probable'], 1)
        self.assertEqual(initial['revision'], 0)
        self.assertEqual(json.dumps(data), before)

    def test_reassignment_and_history(self):
        data = fixture()
        first = apply_decision(data, empty_review(data), payload())
        second = apply_decision(data, first, payload(revision=1, tooth_instance_id=None, region='Interdental region',
                               reassign_from=pair_id('Validation_1', 0, 'mask1')))
        self.assertEqual(second['revision'], 2)
        self.assertEqual(len(second['history']), 3)
        self.assertEqual(report(data, second)['counts']['rejected'], 1)
        self.assertEqual(report(data, second)['counts']['confirmed'], 1)

    def test_reject_uncertain_unassigned_and_invalid_input(self):
        data = fixture()
        for status in ['rejected', 'uncertain', 'unassigned', 'probable']:
            result = apply_decision(data, empty_review(data), payload(status=status))
            self.assertEqual(result['decisions'][0]['status'], status)
        for change in [{'revision': 7}, {'reviewer': ''}, {'bone_line_id': 1}, {'tooth_instance_id': 'unknown'},
                       {'status': 'automatic_ground_truth'}, {'region': 'region'}, {'reassign_from': 'unknown'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply_decision(data, empty_review(data), payload(**change))

    def test_server_persistence_conflict_and_source_isolation(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'reports').mkdir()
            (folder / 'candidate_mappings.json').write_text(json.dumps(data))
            (folder / 'expert_review.json').write_text(json.dumps(empty_review(data)))
            server = make_server(folder, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            request = Request(url+'/api/decision', json.dumps(payload()).encode(), {'Content-Type': 'application/json'})
            try:
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['review']['revision'], 1)
                self.assertEqual(json.loads((folder / 'expert_review.json').read_text())['decisions'][0]['status'], 'confirmed')
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 409)
                cross = Request(url+'/api/decision', json.dumps(payload(revision=1)).encode(), {'Content-Type': 'application/json', 'Origin': 'http://unrelated.example'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(cross)
                self.assertEqual(error.exception.code, 403)
                with self.assertRaises(HTTPError):
                    urlopen(url+'/../candidate_mappings.json')
                self.assertEqual(json.loads((folder / 'candidate_mappings.json').read_text()), data)
                with urlopen(url+'/api/report') as response:
                    self.assertEqual(json.load(response)['counts']['confirmed'], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
