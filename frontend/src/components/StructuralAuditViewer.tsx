import { useEffect, useState } from 'react';
import { getStructuralAudit } from '../lib/annotations';
import type { StructuralAuditRecord } from '../types';

interface Props { imageId: string; imageSrc: string; onBack: () => void; }
type ReviewTab = 'bone' | 'annotations' | 'correspondence';
type Status = 'confirmed' | 'uncertain' | 'missing' | 'not_applicable';
type Layers = { boneLines: boolean; toothMasks: boolean; cocoBoxes: boolean; keypointBoxes: boolean; cej: boolean; apex: boolean; radiographMask: boolean; };
const initialLayers: Layers = { boneLines: true, toothMasks: false, cocoBoxes: false, keypointBoxes: false, cej: false, apex: false, radiographMask: false };

export default function StructuralAuditViewer({ imageId, imageSrc, onBack }: Props) {
  const [tab, setTab] = useState<ReviewTab>('bone');
  const [record, setRecord] = useState<StructuralAuditRecord | null>(null);
  const [layers, setLayers] = useState<Layers>(initialLayers);
  const [error, setError] = useState<string | null>(null);
  const [statuses, setStatuses] = useState<Record<string, Status>>({});
  const [selected, setSelected] = useState<Record<string, string>>({});
  const [bonePrediction, setBonePrediction] = useState<'loading' | 'available' | 'missing' | 'backend_outdated' | 'backend_error'>('loading');
  const [showSourceLines, setShowSourceLines] = useState(true);
  const [showPrediction, setShowPrediction] = useState(true);
  const [boneReview, setBoneReview] = useState({ outcome: 'unreviewed', notes: '' });
  const [boneReviewSaved, setBoneReviewSaved] = useState(false);
  useEffect(() => { let active = true; setRecord(null); setError(null); setTab('bone'); void getStructuralAudit(imageId).then((value) => { if (active) setRecord(value); }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Structural audit is unavailable.'); }); return () => { active = false; }; }, [imageId]);
  const dimensions = record?.image;
  const keypoints = record?.keypoints;
  const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
  const sourceImageUrl = `${apiUrl}/structural-audit/${encodeURIComponent(imageId)}/image`;
  const predictionUrl = `${apiUrl}/structural-audit/${encodeURIComponent(imageId)}/bone-line-prediction`;
  useEffect(() => {
    const controller = new AbortController();
    setBonePrediction('loading');
    void fetch(predictionUrl, { signal: controller.signal }).then(async (response) => {
      if (controller.signal.aborted) return;
      if (response.ok) { setBonePrediction('available'); return; }
      if (response.status === 404) {
        const body = await response.json().catch(() => ({})) as { detail?: string };
        setBonePrediction(body.detail === 'Not Found' ? 'backend_outdated' : 'missing');
      } else setBonePrediction('backend_error');
    }).catch(() => { if (!controller.signal.aborted) setBonePrediction('backend_error'); });
    return () => controller.abort();
  }, [predictionUrl]);
  useEffect(() => {
    try {
      const saved = localStorage.getItem(`bone-line-review:${imageId}`);
      setBoneReview(saved ? JSON.parse(saved) as { outcome: string; notes: string } : { outcome: 'unreviewed', notes: '' });
    } catch { setBoneReview({ outcome: 'unreviewed', notes: '' }); }
    setBoneReviewSaved(false);
  }, [imageId]);
  const saveBoneReview = () => {
    localStorage.setItem(`bone-line-review:${imageId}`, JSON.stringify({ image_id: imageId, model: 'bone_line_unet_baseline_epoch_7', ...boneReview, saved_at: new Date().toISOString() }));
    setBoneReviewSaved(true);
  };
  const exportBoneReviews = () => {
    const rows: unknown[] = [];
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (!key?.startsWith('bone-line-review:')) continue;
      try {
        const value = localStorage.getItem(key);
        if (value) rows.push(JSON.parse(value));
      } catch { /* Ignore malformed local entries. */ }
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify({ task: 'bone_line_visual_review', reviews: rows }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'bone-line-reviews.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const selectedMask = selected.tooth_masks;
  const selectedCoco = selected.coco_instances;
  const selectedBox = selected.keypoint_bboxes;
  const selectedCej = selected.cej_points;
  const selectedApex = selected.apex_points;
  const selectItem = (kind: string, value: string) => setSelected((current) => ({ ...current, [kind]: value }));
  if (record?.available === false) {
    return <section className="audit-viewer-page">
      <div className="audit-viewer-heading"><div><span className="eyebrow">DENPAR CORRESPONDENCE REVIEW</span><h2>Structural annotation inspection</h2><p>{imageId} · source coordinates preserved</p></div><button className="secondary-btn" onClick={onBack}>Back</button></div>
      <div className="audit-warning" role="status"><strong>Structural annotations unavailable.</strong> {record.reason} The uploaded radiograph remains available for tooth segmentation and expert review.</div>
    </section>;
  }
  return <section className="audit-viewer-page">
    <div className="audit-viewer-heading"><div><span className="eyebrow">DENPAR CORRESPONDENCE REVIEW</span><h2>Structural annotation inspection</h2><p>{imageId} · source coordinates preserved</p></div><button className="secondary-btn" onClick={onBack}>Back</button></div>
    {error && <div className="save-error" role="alert">{error}</div>}
    {!record && !error && <div className="audit-loading" role="status"><img src={imageSrc} alt={`Uploaded radiograph ${imageId}`} /><span>Loading source annotations…</span></div>}
    {record && <div className="audit-warning">This review shows source geometry and an experimental bone-line prediction. It does not establish disease, severity, or tooth correspondence.</div>}
    {record && <div className="audit-summary-grid"><div><strong>{record.bone_lines?.Bone_Lines.length ?? 0}</strong><span>bone lines</span></div><div><strong>{record.tooth_masks.length}</strong><span>tooth masks</span></div><div><strong>{keypoints?.CEJ_Points.length ?? 0}</strong><span>CEJ points</span></div><div><strong>{keypoints?.Apex_Points.length ?? 0}</strong><span>apex points</span></div></div>}
    {record && <nav className="audit-section-tabs" aria-label="Structural review sections">{([['bone', 'Bone model review'], ['annotations', 'Source annotations'], ['correspondence', 'Correspondence']] as const).map(([key, label]) => <button key={key} type="button" className={tab === key ? 'active' : ''} aria-current={tab === key ? 'page' : undefined} onClick={() => setTab(key)}>{label}</button>)}</nav>}
    {record && tab === 'annotations' && <div className="audit-viewer-grid"><div className="audit-canvas-frame"><div className="audit-canvas" style={{ aspectRatio: dimensions ? `${dimensions.width} / ${dimensions.height}` : '1 / 1' }}><img src={sourceImageUrl} alt={`Original DenPAR radiograph ${imageId}`} />
      <div className="audit-mask-layer">{layers.radiographMask && record?.radiograph_mask && <img src={`${apiUrl}${record.radiograph_mask.url}`} alt="Radiograph-wise tooth mask" />}{layers.toothMasks && record?.tooth_masks.filter((mask) => !selectedMask || mask.filename === selectedMask).map((mask) => <img key={mask.filename} src={`${apiUrl}${mask.url}`} alt={`Tooth-wise ${mask.filename}`} />)}</div>
      {dimensions && <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} preserveAspectRatio="none">
        {layers.boneLines && record?.bone_lines?.Bone_Lines.map((line, index) => <polyline key={`bone-${index}`} points={line.map((point) => point.join(',')).join(' ')} fill="none" stroke="#37e5a1" strokeWidth="4" vectorEffect="non-scaling-stroke" />)}
        {layers.cocoBoxes && record?.coco_annotations.map((item) => { const [x, y, width, height] = item.bbox; return <rect key={`coco-${item.id}`} className={selectedCoco && selectedCoco !== String(item.id) ? 'audit-box muted' : 'audit-box'} x={x} y={y} width={width} height={height} />; })}
        {layers.keypointBoxes && keypoints?.bboxes.map((box, index) => <rect key={`kp-${index}`} className={selectedBox && selectedBox !== String(index) ? 'audit-keypoint-box muted' : 'audit-keypoint-box'} x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} />)}
        {layers.cej && keypoints?.CEJ_Points.map((point, index) => <circle key={`cej-${index}`} className={selectedCej && selectedCej !== String(index) ? 'audit-cej muted' : 'audit-cej'} cx={point[0]} cy={point[1]} r={Math.max(dimensions.width, dimensions.height) * 0.006} />)}
        {layers.apex && keypoints?.Apex_Points.map((point, index) => <circle key={`apex-${index}`} className={selectedApex && selectedApex !== String(index) ? 'audit-apex muted' : 'audit-apex'} cx={point[0]} cy={point[1]} r={Math.max(dimensions.width, dimensions.height) * 0.006} />)}
      </svg>}</div></div>
      <aside className="audit-controls"><span className="eyebrow">LAYERS</span>{([['boneLines', 'Bone lines'], ['toothMasks', 'Tooth-wise masks'], ['cocoBoxes', 'COCO tooth boxes'], ['keypointBoxes', 'Key-point boxes'], ['cej', 'CEJ points'], ['apex', 'Apex points'], ['radiographMask', 'Radiograph mask']] as const).map(([key, label]) => <button key={key} className={`audit-layer-toggle ${layers[key] ? 'active' : ''}`} onClick={() => setLayers((current) => ({ ...current, [key]: !current[key] }))}><i />{label}</button>)}
      {record && <div className="audit-stats"><strong>Source counts</strong><span>Masks: {record.tooth_masks.length}</span><span>COCO: {record.coco_annotations.length}</span><span>Key-point boxes: {keypoints?.bboxes.length || 0}</span><span>CEJ: {keypoints?.CEJ_Points.length || 0}</span><span>Apex: {keypoints?.Apex_Points.length || 0}</span></div>}
      {record && <div className="audit-item-selectors"><strong>Isolate an item</strong>{Object.entries(record.correspondence.review_items).map(([kind, items]) => <label key={kind}>{kind.replace(/_/g, ' ')}<select value={selected[kind] || ''} onChange={(event) => selectItem(kind, event.target.value)}><option value="">Show all</option>{items.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}</select></label>)}</div>}</aside></div>}
    {record && tab === 'bone' && <section className="correspondence-review"><div className="structural-analysis-heading"><div><span className="eyebrow">BONE-LINE MODEL REVIEW</span><h3>Source annotation and model prediction</h3></div><span className="pill amber">VALIDATION ONLY</span></div>
      <p>Both views use the original DenPAR image. Green lines are the source polylines; red areas are the experimental model prediction. A line is not a disease label or a confirmed tooth match.</p>
      <div className="bone-review-controls"><label><input type="checkbox" checked={showSourceLines} onChange={(event) => setShowSourceLines(event.target.checked)} /> Source lines</label><label><input type="checkbox" checked={showPrediction} onChange={(event) => setShowPrediction(event.target.checked)} /> Model prediction</label></div>
      <div className="bone-review-grid">
        <div><strong>Source annotation · {record.bone_lines?.Num_of_Bone_Lines ?? 0} lines</strong><svg viewBox={`0 0 ${record.image.width} ${record.image.height}`} role="img" aria-label="Original radiograph with source bone lines"><image href={sourceImageUrl} width={record.image.width} height={record.image.height} />{showSourceLines && record.bone_lines?.Bone_Lines.map((line, index) => <polyline key={index} points={line.map((point) => point.join(',')).join(' ')} fill="none" stroke="#38e485" strokeWidth="4" vectorEffect="non-scaling-stroke" />)}</svg></div>
        <div><strong>Model prediction · epoch 7, threshold 0.7</strong><svg viewBox={`0 0 ${record.image.width} ${record.image.height}`} role="img" aria-label="Original radiograph with predicted bone-line mask"><image href={sourceImageUrl} width={record.image.width} height={record.image.height} />{showPrediction && bonePrediction === 'available' && <image href={predictionUrl} width={record.image.width} height={record.image.height} />}</svg>{bonePrediction === 'loading' && <small role="status">Checking for an exported prediction…</small>}{bonePrediction === 'missing' && <small role="status">No exported prediction for image {imageId}. This can happen when its annotation geometry was excluded or the Validation export has not been generated.</small>}{bonePrediction === 'backend_outdated' && <small role="status">The backend does not have the bone-line prediction route. Restart the updated backend.</small>}{bonePrediction === 'backend_error' && <small role="status">Could not reach the bone-line prediction endpoint. Check that the backend is running and reload this page.</small>}</div>
      </div>
      <div className="bone-review-notes"><label>Prediction review <select value={boneReview.outcome} onChange={(event) => { setBoneReview((current) => ({ ...current, outcome: event.target.value })); setBoneReviewSaved(false); }}><option value="unreviewed">Unreviewed</option><option value="matches">Matches source line</option><option value="missed">Missed line</option><option value="overpredicted">Broad or false-positive prediction</option><option value="annotation_issue">Source annotation issue</option></select></label><label>Notes <textarea value={boneReview.notes} onChange={(event) => { setBoneReview((current) => ({ ...current, notes: event.target.value })); setBoneReviewSaved(false); }} placeholder="Describe the line or region to check" /></label><div className="bone-review-actions"><button className="secondary-btn" onClick={saveBoneReview}>Save review in this browser</button><button className="secondary-btn" onClick={exportBoneReviews}>Export all reviews (JSON)</button></div>{boneReviewSaved && <small role="status">Saved locally for image {imageId}. This does not change the source annotation.</small>}</div>
    </section>}
    {record && tab === 'correspondence' && <section className="correspondence-review"><div className="structural-analysis-heading"><div><span className="eyebrow">CORRESPONDENCE STATUS</span><h3>Uncertain candidates are not ground truth</h3></div><span className="pill amber">{record.correspondence.status.toUpperCase()}</span></div>
      {record.correspondence.relationships.map((item) => <div className="correspondence-row" key={item.name}><div><strong>{item.name}</strong><small>{item.evidence}</small>{item.unresolved_reason && <small className="correspondence-reason">{item.unresolved_reason}</small>}</div><select value={statuses[item.name] || item.status} onChange={(event) => setStatuses((current) => ({ ...current, [item.name]: event.target.value as Status }))}><option value="confirmed">Confirmed</option><option value="uncertain">Uncertain</option><option value="missing">Missing</option><option value="not_applicable">Not applicable</option></select></div>)}
      {(record.correspondence.missing_landmarks.length > 0 || record.correspondence.suspicious_mappings.length > 0) && <div className="correspondence-findings">{record.correspondence.missing_landmarks.map((item) => <span key={`missing-${item}`}>Missing landmark: {item}</span>)}{record.correspondence.suspicious_mappings.map((item) => <span key={item}>Suspicious: {item}</span>)}</div>}
    </section>}
  </section>;
}
