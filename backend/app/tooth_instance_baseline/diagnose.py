"""Preflight and smoke diagnostics; does not run the full experiment."""
from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import torch, torchvision
from torch.utils.data import DataLoader
from .data import ToothInstanceDataset, collate, validate_official_split
from .model import create_model
from .train import move_targets, seed_everything

def main() -> None:
    root=Path('../DenPAR Radiographs Dataset/Dataset').resolve(); raw=Path('../dataset/raw').resolve(); out=Path('../models/tooth_instance_maskrcnn_baseline').resolve(); logs=out/'logs'; reports=out/'reports'; smoke=out/'smoke_test'
    for path in (logs,reports,smoke): path.mkdir(parents=True,exist_ok=True)
    start=time.time(); diagnostics={'command':sys.argv,'cwd':str(Path.cwd()),'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_available':torch.cuda.is_available(),'dataset_paths':{'training_images':str(raw),'training_masks':str(root/'Training/Masks (Tooth-wise)'),'validation_images':str(root/'Validation/Images'),'validation_masks':str(root/'Validation/Masks (Tooth-wise)'),'testing_images':str(root/'Images'),'testing_masks':str(root/'Testing/Masks (Tooth-wise)')}}
    records, validation={},{}
    for split in ('train','validation','test'): records[split],validation[split]=validate_official_split(root,split,raw)
    qa=[]; dataset=ToothInstanceDataset(records['train'],1024,False,42)
    for index in range(10):
        image,target=dataset[index]; qa.append({'image_id':target['metadata']['image_id'],'image_shape':list(image.shape),'mask_shape':list(target['masks'].shape),'nonempty_masks':bool(target['masks'].any(dim=(1,2)).all()),'valid_boxes':bool((target['boxes'][:,2]>target['boxes'][:,0]).all() and (target['boxes'][:,3]>target['boxes'][:,1]).all()),'preprocessing':target['metadata']['preprocessing']})
    loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate); images,targets=next(iter(loader)); diagnostics['first_batch']={'image_shape':list(images[0].shape),'image_dtype':str(images[0].dtype),'range':[float(images[0].min()),float(images[0].max())],'mask_shape':list(targets[0]['masks'].shape),'box_shape':list(targets[0]['boxes'].shape),'label_shape':list(targets[0]['labels'].shape)}
    preflight={'training':validation['train'],'validation':validation['validation'],'testing':validation['test'],'loader_qa':qa,'gates':{'train_650':len(records['train'])==650,'instances_2882':validation['train']['total_instances']==2882,'no_invalid':not validation['train']['invalid_masks'],'qa_passed':all(item['nonempty_masks'] and item['valid_boxes'] and item['image_shape']==[3,1024,1024] for item in qa)}}
    (reports/'refreshed_preflight_report.json').write_text(json.dumps(preflight,indent=2),encoding='utf-8')
    seed_everything(42); model,metadata=create_model(False,1024); model.train(); optimizer=torch.optim.SGD(model.parameters(),lr=0.0001,momentum=.9)
    losses=model([images[0]],move_targets(list(targets),torch.device('cpu'))); total=sum(losses.values()); finite=bool(torch.isfinite(total));
    if finite: optimizer.zero_grad(); total.backward(); optimizer.step()
    checkpoint=smoke/'smoke_latest.pth'; torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'losses':{k:float(v.detach()) for k,v in losses.items()}},checkpoint); restored,_=create_model(False,1024); restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
    diagnostics['model']={**metadata,'losses':{k:float(v.detach()) for k,v in losses.items()},'total_loss':float(total.detach()),'finite_losses':finite,'checkpoint_roundtrip':True,'elapsed_seconds':time.time()-start}
    diagnostics['exit_code']=0; (logs/'training_startup_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8'); (logs/'training_startup_stdout.log').write_text('Preflight and CPU smoke step completed.\n',encoding='utf-8'); (logs/'training_startup_stderr.log').write_text('',encoding='utf-8')
    print(json.dumps({'gates':preflight['gates'],'smoke':diagnostics['model']},indent=2))
if __name__=='__main__': main()
