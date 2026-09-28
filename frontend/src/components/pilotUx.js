export const PILOT_STEPS = [
  ['orientation','Verify image orientation'],['identity','Confirm tooth identity'],['surfaces','Confirm mesial/distal'],
  ['cej','Place CEJ landmarks'],['bone','Place bone crest'],['apex','Select apex'],['review','Review RBL and warnings'],
];
export function tutorialKey(expert='anonymous'){ return `surface-pilot-tutorial-v1:${expert||'anonymous'}`; }
export function tutorialComplete(storage,expert){ return storage.getItem(tutorialKey(expert))==='complete'; }
export function finishTutorial(storage,expert){ storage.setItem(tutorialKey(expert),'complete'); }
export function stepComplete(step,orientation,teeth){
 if(step===0)return orientation.status==='verified';
 if(!teeth.length)return false;
 if(step===1)return teeth.every(t=>t.tooth_verification_status==='confirmed'&&/^([1-4][1-8])$/.test(t.expert_fdi||''));
 if(step===2)return teeth.every(t=>t.surfaces.every(s=>s.verification_status==='ungradable'||(s.pixel_side&&s.adjacent_fdi!==undefined)));
 const key=step===3?'cej':step===4?'bone_crest':step===5?'root_apex':null;
 if(key)return teeth.every(t=>t.surfaces.every(s=>s.verification_status==='ungradable'||s[key].verification_status==='confirmed'||s[key].state==='not_visible'));
 return step===6;
}
export function nextIncomplete(statuses){const index=statuses.findIndex(x=>!x.phase_a_locked);return index<0?statuses.findIndex(x=>!x.phase_b):index;}
export function caseLabel(state){if(state?.phase_b)return'Phase B completed';if(state?.phase_a_locked)return'Phase A completed';if(state?.draft)return'In progress';return'Not started';}
