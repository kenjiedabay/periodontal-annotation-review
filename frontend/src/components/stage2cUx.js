export function findingId(f){return f.source_annotation_id||f.source_polyline_id||''}
export function findingName(f){return f.landmark_type==='cej'?'CEJ landmark':f.landmark_type==='apex'?'Root apex landmark':'Bone line'}
export function reviewedIds(actions){return new Set(actions.map(a=>a.source_annotation_id).filter(Boolean))}
export function nextUnreviewed(findings,actions,after=-1){const done=reviewedIds(actions);for(let n=1;n<=findings.length;n++){const i=(after+n)%findings.length;if(!done.has(findingId(findings[i])))return i}return -1}
export function caseStatus(findings,actions){if(!actions.length)return'Not started';const done=reviewedIds(actions).size;if(done>=findings.length)return'Complete';if(actions.some(a=>a.review_state==='needs_second_review'))return'Needs second review';return'In progress'}
