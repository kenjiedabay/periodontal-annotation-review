import json
from pathlib import Path
import stage2c_pilot_v2 as pilot

def identity(queue):return [(x['stable_image_id'],x['primary_selection_reason']) for x in queue['entries']]

def test_balanced_primary_group_counts_and_routine_nonclipped(tmp_path):
 result=pilot.build(tmp_path/'pilot',20260922);q=result['queue']
 assert q['primary_group_counts']==dict(pilot.QUOTAS) and q['unmet_quotas']=={}
 routine=[x for x in q['entries'] if x['primary_selection_reason']=='high_confidence_non_clipped']
 assert len(routine)==10 and all('clipped_crop' not in x['secondary_characteristics'] for x in routine)

def test_v2_reproducible_unique_and_safe(tmp_path):
 a=pilot.build(tmp_path/'a',20260922)['queue'];b=pilot.build(tmp_path/'b',20260922)['queue']
 assert identity(a)==identity(b) and len(set(x['stable_image_id'] for x in a['entries']))==40
 excluded={x['stable_image_id'] for x in pilot.load_manifests()['denpar_exclusion_manifest.json']['entries']}
 assert not excluded.intersection(x['stable_image_id'] for x in a['entries'])
 assert all(x['source_partition'] in ('Training','Validation') for x in a['entries'])

def test_pilot_v1_is_preserved():
 root=Path(__file__).resolve().parents[1]/'artifacts'
 assert (root/'denpar-stage2c-pilot-v1'/'pilot_queue.json').is_file()
 assert json.loads((root/'denpar-stage2c-pilot-v2'/'queue_summary.json').read_text())['acceptance_metrics_calculated'] is False
