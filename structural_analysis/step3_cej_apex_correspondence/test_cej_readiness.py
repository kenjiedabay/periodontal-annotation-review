import unittest
from cej_readiness import readiness, review_records
from landmarks import apply_review, empty_review, identity
from test_landmarks import fixture, payload


class CEJReadinessTests(unittest.TestCase):
    def test_no_expert_decisions_do_not_inherit_geometric_status(self):
        data = fixture()
        r = data['records'][0]
        r.update(split='Validation', original_url='overlays/test/original.jpg')
        for t in r['teeth']:
            t.update(bbox=[0,0,20,20], source='mask.png', overlay_url='overlay.png')
        for c in data['candidates']:
            c['features'] = {'min_mask_pixel_center_distance_px': 0, 'normalized_mask_distance': 0, 'keypoint_bbox_support': .9}
        queue = review_records(data, empty_review(data))
        self.assertTrue(all(q['expert_status'] == 'unreviewed' for q in queue))
        result = readiness(data, empty_review(data), {'records': []})
        self.assertIn('STATUS C', result['status'])
        self.assertEqual(result['uncertain_points'], 0)
        self.assertEqual(result['unresolved_points'], 2)

    def test_reassignment_preserves_candidate_and_expert_targets(self):
        data = fixture()
        review = apply_review(data, empty_review(data), payload(tooth_instance_id='mask2', reassign_from=identity('Validation_1','CEJ:0','mask1')))
        new = next(d for d in review['decisions'] if d['status'] == 'confirmed')
        self.assertEqual(new['candidate_tooth_instance_id'], 'mask1')
        self.assertEqual(new['expert_tooth_instance_id'], 'mask2')
        self.assertEqual(new['expert_status'], 'confirmed')
        undecidable = apply_review(data, empty_review(data), payload(status='cannot_determine'))
        result = readiness(data, undecidable, {'records': []})
        self.assertEqual(result['cannot_determine_points'], 1)
        self.assertEqual(result['unresolved_points'], 2)

    def test_readiness_counts_only_manifest_targets_by_partition(self):
        data = fixture()
        record = {'key': 'Validation_1', 'image_id': '1', 'tooth_instance_id': 'mask1', 'partition': 'Validation',
                  'cej_points': [[10,10],[11,10]], 'target_provenance': [{'kind':'landmark','json_field':'CEJ_Points','source_index':i} for i in [0,1]]}
        result = readiness(data, empty_review(data), {'records':[record]})
        self.assertIn('STATUS B', result['status'])
        self.assertEqual(result['official_partitions']['Validation']['confirmed_CEJ_points'], 2)
        self.assertEqual(len(result['multiple_CEJ_per_tooth_cases']), 1)


if __name__ == '__main__':
    unittest.main()
