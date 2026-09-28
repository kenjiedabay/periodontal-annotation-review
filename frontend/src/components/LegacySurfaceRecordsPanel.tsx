import {useEffect,useState} from 'react';

const API=(import.meta.env.VITE_API_URL as string|undefined)?.replace(/\/$/,'')||'http://localhost:8000';

export default function LegacySurfaceRecordsPanel({imageId,onBack}:{imageId:string;onBack:()=>void}){
 const[data,setData]=useState<any>(null),[error,setError]=useState('');
 useEffect(()=>{let active=true;(async()=>{for(const partition of ['Training','Validation']){const r=await fetch(`${API}/api/surface-verification/${partition}/${encodeURIComponent(imageId)}`);if(r.ok){const body=await r.json();if(active)setData({...body,partition});return}}throw new Error('Historical surface records are unavailable for this image.')})().catch(e=>{if(active)setError(String(e.message||e))});return()=>{active=false}},[imageId]);
 return <section className="review-foundation-card legacy-surface-records">
  <span className="eyebrow">ADVANCED / LEGACY</span><h2>Historical surface records</h2>
  <p className="notice">Read-only archive. New tooth and surface work must be completed in Unified Anatomical Review.</p>
  {error&&<div className="save-error" role="alert">{error}</div>}
  {!data&&!error&&<p>Loading historical records…</p>}
  {data&&<><p><strong>{data.history_count}</strong> saved standalone version(s) for {data.partition} image {imageId}.</p>{data.latest?<details><summary>View latest archived record</summary><pre className="pilot-model-json">{JSON.stringify(data.latest,null,2)}</pre></details>:<p>No standalone surface record exists for this image.</p>}</>}
  <button onClick={onBack}>Back to Image Review</button>
 </section>;
}
