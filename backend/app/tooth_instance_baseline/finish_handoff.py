from __future__ import annotations
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('../models/tooth_instance_maskrcnn_baseline/full_run'); R=O/'reports'; P=O/'plots'
def dump(n,x):(R/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 P.mkdir(exist_ok=True); e=json.loads((O/'evaluation_report.json').read_text()); v=json.loads((R/'validation_report.json').read_text()); t=json.loads((R/'test_report.json').read_text()); logs=v['epochs']; per=t['per_image']
 def g(name, series):
  plt.figure();
  for y,label in series: plt.plot(range(1,len(logs)+1),y,label=label)
  plt.xlabel('Epoch');plt.legend();plt.tight_layout();plt.savefig(P/name,dpi=120);plt.close()
 g('training_loss.png',[([sum(x['training_loss_components'].values()) for x in logs],'training loss')]);g('validation_dice.png',[([x['validation']['dice'] for x in logs],'Dice')]);g('validation_iou.png',[([x['validation']['iou'] for x in logs],'IoU')]);g('validation_precision_recall.png',[([x['validation']['precision'] for x in logs],'precision'),([x['validation']['recall'] for x in logs],'recall')]);g('learning_curve_summary.png',[([x['validation']['dice'] for x in logs],'Dice'),([x['validation']['iou'] for x in logs],'IoU')])
 q=json.loads((R/'threshold_analysis_report.json').read_text())['rows'];plt.figure();plt.plot([x['threshold'] for x in q],[x['dice'] for x in q]);plt.tight_layout();plt.savefig(P/'threshold_analysis.png',dpi=120);plt.close()
 s=sorted(per,key=lambda x:x['dice']);dump('error_analysis.json',{'lowest_dice':s[:10],'highest_dice':s[-10:],'median_example':s[len(s)//2],'highest_false_positives':sorted(per,key=lambda x:x['false_positive_instances'],reverse=True)[:10],'images_with_missed_instances':[x for x in per if x['missed_instances']>0],'note':'Stored quantitative metrics only; structural cause categories require qualitative expert review. No disease inference.'})
 lock=json.loads((R/'final_model_lock.json').read_text());a=t['aggregate_mean_per_image'];md=f'''# Mask R-CNN Tooth-Instance Segmentation Baseline\n\n## Dataset\n\nTraining: 650 images / 2,882 tooth instances. Validation: 150 / 655. Testing: 200 / 864. Official partitions were retained; patient/case identifiers were unavailable.\n\n## Preprocessing\n\nGrayscale, aspect-ratio-preserving resize, centred zero-padding to 1024×1024, percentile normalization, CLAHE, nearest-neighbor masks.\n\n## Model and Training\n\nMask R-CNN ResNet-50 FPN; COCO-pretrained, all layers fine-tuned, AMP, batch size 1, seed 42, LR 0.0001, 10 epochs. Actual selection criterion: Validation Dice; best epoch {lock['checkpoint_epoch']}.\n\n## Validation\n\nDice {lock['validation_metric']['dice']:.6f}; IoU {lock['validation_metric']['iou']:.6f}.\n\n## Test\n\nExisting threshold-0.5 Test result: Dice {a['dice']:.6f}; IoU {a['iou']:.6f}; precision {a['precision']:.6f}; recall {a['recall']:.6f}.\n\n## Threshold Sensitivity\n\nThe Test partition had already been evaluated at 0.5 before the formal lock. Subsequent Validation-only analysis is descriptive and was not used to modify or re-evaluate Test.\n\n## Limitations\n\nTooth-instance segmentation only; no periodontal disease, diseased-tooth, severity, or progression prediction. Patient-level independence could not be verified. Formal locking followed initial Test exposure, transparently documented. Research prototype, not clinical validation.\n''';(R/'final_model_report.md').write_text(md,encoding='utf-8')
 files=[O/'config/final_train_config.json',O/'best_checkpoint.pt',O/'latest_checkpoint.pt',O/'evaluation_report.json',R/'validation_report.json',R/'threshold_analysis_report.json',R/'final_model_lock.json',R/'test_exposure_record.json',R/'test_report.json',R/'error_analysis.json',R/'final_model_report.md'];dump('baseline_handoff_manifest.json',{'files':[{'path':str(x),'exists':x.exists(),'sha256':h(x) if x.exists() else None} for x in files],'plots':[str(x) for x in P.glob('*.png')],'qualitative_outputs':str(O/'qualitative_predictions')})
if __name__=='__main__':main()
