import { useEffect, useState } from 'react';
import { getStructuralAudit } from '../lib/annotations';
import type { StructuralAuditRecord } from '../types';

interface Props { imageId: string; imageSrc: string; onBack: () => void; }
type Status = 'confirmed' | 'uncertain' | 'missing' | 'not_applicable';
type Layers = { toothMasks: boolean; cocoBoxes: boolean; keypointBoxes: boolean; cej: boolean; apex: boolean; radiographMask: boolean; };
const initialLayers: Layers = { toothMasks: true, cocoBoxes: true, keypointBoxes: true, cej: true, apex: true, radiographMask: false };

export default function StructuralAuditViewer({ imageId, imageSrc, onBack }: Props) {
  const [record, setRecord] = useState<StructuralAuditRecord | null>(null);
  const [layers, setLayers] = useState<Layers>(initialLayers);
  const [error, setError] = useState<string | null>(null);
  const [statuses, setStatuses] = useState<Record<string, Status>>({});
  const [selected, setSelected] = useState<Record<string, string>>({});
  useEffect(() => { let active = true; void getStructuralAudit(imageId).then((value) => { if (active) setRecord(value); }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Structural audit is unavailable.'); }); return () => { active = false; }; }, [imageId]);
  const dimensions = record?.image;
  const keypoints = record?.keypoints;
  const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
  const selectedMask = selected.tooth_masks;
  const selectedCoco = selected.coco_instances;
  const selectedBox = selected.keypoint_bboxes;
  const selectedCej = selected.cej_points;
  const selectedApex = selected.apex_points;
  const selectItem = (kind: string, value: string) => setSelected((current) => ({ ...current, [kind]: value }));
  return <section className="audit-viewer-page">
    <div className="audit-viewer-heading"><div><span className="eyebrow">DENPAR CORRESPONDENCE REVIEW</span><h2>Structural annotation inspection</h2><p>{imageId} · source coordinates preserved</p></div><button className="secondary-btn" onClick={onBack}>Back</button></div>
    {error && <div className="save-error" role="alert">{error}</div>}
    {record && <div className="audit-warning">This review exposes source evidence and candidates only. It creates no landmark pairing and makes no clinical interpretation.</div>}
    <div className="audit-viewer-grid"><div className="audit-canvas-frame"><div className="audit-canvas" style={{ aspectRatio: dimensions ? `${dimensions.width} / ${dimensions.height}` : '1 / 1' }}><img src={imageSrc} alt={`Original DenPAR radiograph ${imageId}`} />
      <div className="audit-mask-layer">{layers.radiographMask && record?.radiograph_mask && <img src={`${apiUrl}${record.radiograph_mask.url}`} alt="Radiograph-wise tooth mask" />}{layers.toothMasks && record?.tooth_masks.filter((mask) => !selectedMask || mask.filename === selectedMask).map((mask) => <img key={mask.filename} src={`${apiUrl}${mask.url}`} alt={`Tooth-wise ${mask.filename}`} />)}</div>
      {dimensions && <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} preserveAspectRatio="none">
        {layers.cocoBoxes && record?.coco_annotations.map((item) => { const [x, y, width, height] = item.bbox; return <rect key={`coco-${item.id}`} className={selectedCoco && selectedCoco !== String(item.id) ? 'audit-box muted' : 'audit-box'} x={x} y={y} width={width} height={height} />; })}
        {layers.keypointBoxes && keypoints?.bboxes.map((box, index) => <rect key={`kp-${index}`} className={selectedBox && selectedBox !== String(index) ? 'audit-keypoint-box muted' : 'audit-keypoint-box'} x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} />)}
        {layers.cej && keypoints?.CEJ_Points.map((point, index) => <circle key={`cej-${index}`} className={selectedCej && selectedCej !== String(index) ? 'audit-cej muted' : 'audit-cej'} cx={point[0]} cy={point[1]} r={Math.max(dimensions.width, dimensions.height) * 0.006} />)}
        {layers.apex && keypoints?.Apex_Points.map((point, index) => <circle key={`apex-${index}`} className={selectedApex && selectedApex !== String(index) ? 'audit-apex muted' : 'audit-apex'} cx={point[0]} cy={point[1]} r={Math.max(dimensions.width, dimensions.height) * 0.006} />)}
      </svg>}</div></div>
      <aside className="audit-controls"><span className="eyebrow">LAYERS</span>{([['toothMasks', 'Tooth-wise masks'], ['cocoBoxes', 'COCO tooth boxes'], ['keypointBoxes', 'Key-point boxes'], ['cej', 'CEJ points'], ['apex', 'Apex points'], ['radiographMask', 'Radiograph mask']] as const).map(([key, label]) => <button key={key} className={`audit-layer-toggle ${layers[key] ? 'active' : ''}`} onClick={() => setLayers((current) => ({ ...current, [key]: !current[key] }))}><i />{label}</button>)}
      {record && <div className="audit-stats"><strong>Source counts</strong><span>Masks: {record.tooth_masks.length}</span><span>COCO: {record.coco_annotations.length}</span><span>Key-point boxes: {keypoints?.bboxes.length || 0}</span><span>CEJ: {keypoints?.CEJ_Points.length || 0}</span><span>Apex: {keypoints?.Apex_Points.length || 0}</span></div>}</aside></div>
    {record && <section className="correspondence-review"><div className="structural-analysis-heading"><div><span className="eyebrow">CORRESPONDENCE STATUS</span><h3>Uncertain candidates are not ground truth</h3></div><span className="pill amber">{record.correspondence.status.toUpperCase()}</span></div>
      {record.correspondence.relationships.map((item) => <div className="correspondence-row" key={item.name}><div><strong>{item.name}</strong><small>{item.evidence}</small>{item.unresolved_reason && <small className="correspondence-reason">{item.unresolved_reason}</small>}</div><select value={statuses[item.name] || item.status} onChange={(event) => setStatuses((current) => ({ ...current, [item.name]: event.target.value as Status }))}><option value="confirmed">Confirmed</option><option value="uncertain">Uncertain</option><option value="missing">Missing</option><option value="not_applicable">Not applicable</option></select></div>)}
      <div className="correspondence-inspector"><strong>Expert inspection candidates</strong><small>Select an item to isolate it on the image. The selections are local review state; they do not write to DenPAR or certify a correspondence.</small>{Object.entries(record.correspondence.review_items).map(([kind, items]) => <label key={kind}>{kind.replace(/_/g, ' ')}<select value={selected[kind] || ''} onChange={(event) => selectItem(kind, event.target.value)}><option value="">All / no selection</option>{items.map((item) => <option key={item.id} value={item.id}>{item.id}{item.bbox_candidates ? ` — key-point box candidates: ${item.bbox_candidates.join(', ') || 'none'}` : ''}</option>)}</select></label>)}</div>
      {(record.correspondence.missing_landmarks.length > 0 || record.correspondence.suspicious_mappings.length > 0) && <div className="correspondence-findings">{record.correspondence.missing_landmarks.map((item) => <span key={`missing-${item}`}>Missing landmark: {item}</span>)}{record.correspondence.suspicious_mappings.map((item) => <span key={item}>Suspicious: {item}</span>)}</div>}
    </section>}
  </section>;
}
