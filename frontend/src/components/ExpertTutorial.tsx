import {useEffect,useRef,useState} from 'react';
const pages=[
 ['Purpose','Assess current radiographic alveolar bone loss. This is not a definitive periodontitis diagnosis.'],
 ['Two review phases','Phase A is independent and blinded. Phase B begins only after Phase A is locked and reveals a separate model snapshot.'],
 ['Orientation and FDI','Confirm arch, patient side, display orientation, flip status, and FDI identity before assigning surfaces.'],
 ['Mesial and distal','Mesial faces the dental midline; distal faces away. Image-left is not automatically mesial.'],
 ['CEJ landmark','Place the point at the visible cementoenamel junction for the selected tooth surface.'],
 ['Bone crest','Place the point where the alveolar bone level intersects the selected tooth surface.'],
 ['Apex reference','Select the root apex used for RBL. Multi-rooted teeth require your manual selection.'],
 ['Visibility and grading','Use Not visible when a landmark cannot be located reliably. Mark a surface ungradable when a defensible measurement is impossible.'],
 ['Warnings','A blocking warning must be resolved or the surface marked ungradable. Other warnings request careful review.'],
 ['Saving and locking','Save drafts, pause, and resume freely. Locking Phase A is irreversible and is required before model reveal.'],
 ['After Phase A','Compare the immutable model snapshot in Phase B. Assisted corrections never overwrite Phase A or model output.'],
];
export default function ExpertTutorial({open,onClose,onFinish}:{open:boolean;onClose:()=>void;onFinish:()=>void}){const[index,setIndex]=useState(0),ref=useRef<HTMLButtonElement>(null);useEffect(()=>{if(open){setIndex(0);setTimeout(()=>ref.current?.focus(),0)}},[open]);if(!open)return null;return <div className="tutorial-backdrop" role="presentation"><section className="tutorial-dialog" role="dialog" aria-modal="true" aria-labelledby="tutorial-title"><span className="eyebrow">EXPERT TUTORIAL · {index+1} OF {pages.length}</span><div className="tutorial-diagram" aria-hidden="true">{index===3?'MIDLINE  ←  M | TOOTH | D  →':index>=4&&index<=6?'CEJ  ● ─── ●  BONE  ─── ●  APEX':'PHASE A  →  LOCK  →  PHASE B'}</div><h2 id="tutorial-title">{pages[index][0]}</h2><p>{pages[index][1]}</p><div className="tutorial-actions"><button onClick={onClose}>Skip</button><button disabled={!index} onClick={()=>setIndex(index-1)}>Back</button>{index<pages.length-1?<button ref={ref} className="primary-action" onClick={()=>setIndex(index+1)}>Next</button>:<button ref={ref} className="primary-action" onClick={onFinish}>Finish tutorial</button>}</div></section></div>}
