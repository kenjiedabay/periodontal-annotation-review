"""Post-training handoff that never re-evaluates the exposed Test partition."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from .data import ToothInstanceDataset, collate, validate_official_split
from .model import create_model
from .train import metrics

ROOT=Path('..'); OUT=ROOT/'models/tooth_instance_maskrcnn_baseline/full_run'; REPORTS=OUT/'reports'; PLOTS=OUT/'plots'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,data): (REPORTS/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
def main():
 REPORTS.mkdir(parents=True,exist_ok=True); PLOTS.mkdir(parents=True,exist_ok=True)
 evaluation=json.loads((OUT/'evaluation_report.json').read_text())
 config=json.loads((OUT/'config/final_train_config.json').read_text())
 best_path=OUT/'best_checkpoint.pt'; latest_path=OUT/'latest_checkpoint.pt'
 best=torch.load(best_path,map_location='cpu',weights_only=False); latest=torch.load(latest_path,map_location='cpu',weights_only=False)
 # The implementation selected by Validation Dice; this remains the only selection rule.
 lock={'record_type':'retrospective_lock_record','formal_lock_created_after_first_test_evaluation':True,'no_post_test_model_or_threshold_changes_permitted':True,'checkpoint_used':str(best_path),'checkpoint_epoch':best['epoch'],'checkpoint_sha256':sha(best_path),'latest_checkpoint':str(latest_path),'latest_checkpoint_epoch':latest['epoch'],'architecture':best['architecture'],'preprocessing_version':best['preprocessing_version'],'input_geometry':[1024,1024],'amp_enabled':True,'batch_size':1,'mask_score_threshold':0.5,'matching_iou_threshold':0.5,'selection_rule_actual':'Validation Dice','validation_metric':best['metrics']['validation'],'test_exposure_status':'Test already evaluated before this formal record.'}
 write('final_model_lock.json',lock)
 write('test_exposure_record.json',{'test_partition_already_evaluated':True,'first_known_test_evaluation_source':'evaluation_report.json','mask_score_threshold':0.5,'matching_iou_threshold':0.5,'checkpoint_used':str(best_path),'checkpoint_epoch':best['epoch'],'test_used_for_training':False,'test_used_for_model_selection':False,'post_test_tuning_allowed':False,'notes':'The formal lock artifact was generated after the first Test evaluation; chronology is documented rather than represented as a pre-Test lock.'})
 test=evaluation['test_results']; per=test['per_image']; totals={k:sum(x[k] for x in per) for k in ('matched_instances','missed_instances','false_positive_instances')}
 test_report={'test_exposure':'existing evaluation only; no new Test inference performed','dataset':{'images':200,'instances':864},'configuration':{'checkpoint':str(best_path),'checkpoint_epoch':best['epoch'],'mask_score_threshold':0.5,'matching_iou_threshold':0.5},'aggregate_mean_per_image':test['aggregate'],'absolute_instance_totals_from_per_image':totals,'per_image':per}
 write('test_report.json',test_report)
 logs=evaluation['training_log']; validation={'selection_rule_actual':'Validation Dice','official_best_epoch':best['epoch'],'official_best_metric':best['metrics']['validation']['dice'],'epochs':logs,'best_by_metric':{key:max(logs,key=lambda x:x['validation'][key])['epoch'] for key in ('dice','iou','precision','recall')}}; write('validation_report.json',validation)
 # Validation-only threshold sensitivity. Predictions are computed once per image.
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,_=create_model(False,1024); model.load_state_dict(best['state_dict']); model.to(device).eval()
 dsroot=ROOT/'DenPAR Radiographs Dataset/Dataset'; raw=ROOT/'dataset/raw'; records,_=validate_official_split(dsroot,'validation',raw); loader=DataLoader(ToothInstanceDataset(records,1024,False,42),batch_size=1,shuffle=False,collate_fn=collate)
 cached=[]
 with torch.no_grad():
  for images,targets in loader:
   cached.append((model([images[0].to(device)])[0],targets[0]))
 rows=[]
 for threshold in (.3,.4,.5,.6,.7):
  values=[metrics(p,t,threshold,.5) for p,t in cached]; rows.append({'threshold':threshold,**{k:sum(x[k] for x in values)/len(values) for k in ('dice','iou','precision','recall','matched_instances','missed_instances','false_positive_instances')}})
 write('threshold_analysis_report.json',{'status':'descriptive_validation_only; official_test_threshold_remains_0.5','rows':rows})
 # Curves, based solely on existing log values.
 def plot(name,ys,labels):
  plt.figure(); [plt.plot(range(1,len(logs)+1),v,label=l) for v,l in zip(ys,labels)]; plt.xlabel('Epoch'); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/name,dpi=130); plt.close()
 plot('training_loss.png',[[sum(x['training_loss_components'].values()) for x in logs]],['training loss'])
 plot('validation_dice.png',[[x['validation']['dice'] for x in logs]],['Dice']); plot('validation_iou.png',[[x['validation']['iou'] for x in logs]],['IoU']); plot('validation_precision_recall.png',[[x['validation']['precision'] for x in logs],[x['validation']['recall'] for x in logs]],['precision','recall']); plot('learning_curve_summary.png',[[x['validation']['dice'] for x in logs],[x['validation']['iou'] for x in logs]],['Dice','IoU'])
 plt.figure(); plt.plot([x['threshold'] for x in rows],[x['dice'] for x in rows],label='Dice'); plt.plot([x['threshold'] for x in rows],[x['precision'] for x in rows],label='precision'); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/'threshold_analysis.png',dpi=130); plt.close()
 ranked=sorted(per,key=lambda x:x['dice']); error={'lowest_dice':ranked[:10],'highest_dice':ranked[-10:],'median_example':ranked[len(ranked)//2],'highest_false_positives':sorted(per,key=lambda x:x['false_positive_instances'],reverse=True)[:10],'images_with_missed_instances':[x for x in per if x['missed_instances']>0],'limitations':'Quantitative categories derive from stored matching only; merged/split/contrast/edge causes require expert qualitative review and no disease inference is made.'}; write('error_analysis.json',error)
 md=f'''# Mask R-CNN Tooth-Instance Segmentation Baseline\n\n## Dataset\n\nTraining: 650 images / 2,882 tooth instances. Validation: 150 / 655. Testing: 200 / 864. Official partitions were retained; case identifiers were unavailable, so patient-level independence cannot be independently verified.\n\n## Preprocessing\n\nGrayscale conversion, aspect-ratio-preserving resize, centered zero padding to 1024×1024, percentile normalization, CLAHE, and nearest-neighbor mask geometry.\n\n## Model and training\n\nMask R-CNN ResNet-50 FPN, COCO-pretrained initialization, all layers fine-tuned, AMP enabled, batch size 1, seed 42, learning rate 0.0001, 10 epochs. Checkpoint selection actually used Validation Dice; best epoch: {best['epoch']}.\n\n## Validation\n\nBest-checkpoint Dice {best['metrics']['validation']['dice']:.6f}; IoU {best['metrics']['validation']['iou']:.6f}.\n\n## Test\n\nExisting threshold-0.5 Test evaluation: Dice {test['aggregate']['dice']:.6f}; IoU {test['aggregate']['iou']:.6f}; precision {test['aggregate']['precision']:.6f}; recall {test['aggregate']['recall']:.6f}.\n\n## Threshold sensitivity\n\nThe Test partition had already been evaluated at score threshold 0.5 before the formal lock artifact was generated. Subsequent Validation threshold analysis was descriptive only and was not used to modify or re-evaluate the Test baseline.\n\n## Limitations\n\nThis model performs tooth-instance segmentation only. Tooth masks do not identify periodontal disease, diseased teeth, severity, or progression. Patient/case identifiers were unavailable. Formal locking occurred after first Test evaluation and is transparently documented. Test was not used for further tuning. This is a research prototype, not clinical validation.\n'''; (REPORTS/'final_model_report.md').write_text(md,encoding='utf-8')
 important=[OUT/'config/final_train_config.json',best_path,latest_path,OUT/'evaluation_report.json',REPORTS/'validation_report.json',REPORTS/'threshold_analysis_report.json',REPORTS/'final_model_lock.json',REPORTS/'test_exposure_record.json',REPORTS/'test_report.json',REPORTS/'error_analysis.json',REPORTS/'final_model_report.md']; write('baseline_handoff_manifest.json',{'files':[{'path':str(x),'exists':x.exists(),'sha256':sha(x) if x.exists() else None} for x in important],'plots':[str(x) for x in PLOTS.glob('*.png')],'qualitative_outputs':str(OUT/'qualitative_predictions')})
 print(json.dumps({'best_epoch':best['epoch'],'test_dice':test['aggregate']['dice'],'validation_threshold_rows':rows},indent=2))
if __name__=='__main__': main()
