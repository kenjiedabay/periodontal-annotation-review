import { Fragment, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Circle, Line, Text } from 'react-konva';
import type Konva from 'konva';
import { getToothSegmentation } from '../../lib/annotations';
import type { StructuralAuditRecord, ToothInstance } from '../../types';
import type { ReviewRecord } from '../../lib/reviewFoundation';
import { clamp, fit, screenToImage, type Finding, type Point, type SpatialRecord, type Status, type Tool } from './model';
import './annotation.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const tools: Tool[] = ['select', 'point', 'polyline', 'polygon', 'brush', 'eraser', 'pan'];
const findings: Finding[] = ['cej', 'apex', 'bone_level', 'bone_loss_region', 'other'];
const statuses: Status[] = ['draft', 'confirmed', 'uncertain', 'rejected', 'cannot_determine'];
const colors: Record<Finding, string> = { cej: '#53e6be', apex: '#ffa65e', bone_level: '#ffe277', bone_loss_region: '#f86f87', other: '#ad8bff' };
const preferred: Record<Finding, Tool> = { cej: 'point', apex: 'point', bone_level: 'polyline', bone_loss_region: 'brush', other: 'polygon' };
function useImage(src: string | null) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => { if (!src) { setImage(null); return; } const img = new Image(); img.onload = () => setImage(img); img.src = src; return () => { img.onload = null; }; }, [src]);
  return image;
}
function BinaryOverlay({ src, width, height, opacity }: { src: string; width: number; height: number; opacity: number }) {
  const [image, setImage] = useState<HTMLCanvasElement | null>(null);
  useEffect(() => { const source = new Image(); source.crossOrigin = 'anonymous'; source.onload = () => { const canvas = maskCanvas(width, height); const ctx = canvas.getContext('2d')!; ctx.drawImage(source, 0, 0, width, height); const pixels = ctx.getImageData(0, 0, width, height); for (let i = 0; i < pixels.data.length; i += 4) { const value = pixels.data[i]; pixels.data[i] = 248; pixels.data[i + 1] = 111; pixels.data[i + 2] = 135; pixels.data[i + 3] = value; } ctx.putImageData(pixels, 0, 0); setImage(canvas); }; source.src = src; return () => { source.onload = null; }; }, [src, width, height]);
  return image ? <KonvaImage image={image} width={width} height={height} opacity={opacity} listening={false} /> : null;
}
function ToothOverlay({ src, width, height, opacity }: { src: string; width: number; height: number; opacity: number }) { const image = useImage(src); return image ? <KonvaImage image={image} width={width} height={height} opacity={opacity} listening={false} /> : null; }
function maskCanvas(width: number, height: number) { const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height; return canvas; }
function pointsFlat(points: Point[]) { return points.flatMap(p => [p.x, p.y]); }

interface VersionedSession {
  phase: 'independent' | 'correction';
  record: ReviewRecord | null;
  rawPrediction: ReviewRecord | null;
  onSave: (items: SpatialRecord[]) => Promise<ReviewRecord>;
}

export default function AnnotationCanvas({ imageId, imageSrc, imageFile, width, height, onBack, versioned }: { imageId: string; imageSrc: string; imageFile: File; width: number; height: number; onBack: () => void; versioned?: VersionedSession }) {
  const [size, setSize] = useState({ width: 800, height: 600 });
  const holder = useRef<HTMLDivElement>(null);
  const stage = useRef<Konva.Stage>(null);
  const mask = useRef<HTMLCanvasElement>(maskCanvas(width, height));
  const [maskVersion, setMaskVersion] = useState(0);
  const [records, setRecords] = useState<SpatialRecord[]>([]);
  const [instances, setInstances] = useState<ToothInstance[]>([]);
  const [dataset, setDataset] = useState<StructuralAuditRecord | null>(null);
  const [selectedTooth, setSelectedTooth] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hiddenIds, setHiddenIds] = useState<string[]>([]);
  const [tool, setTool] = useState<Tool>('point');
  const [finding, setFinding] = useState<Finding>('cej');
  const [status, setStatus] = useState<Status>('draft');
  const [comment, setComment] = useState('');
  const [brushSize, setBrushSize] = useState(16);
  const [draft, setDraft] = useState<Point[]>([]);
  const [finished, setFinished] = useState(false);
  const [redoPoints, setRedoPoints] = useState<Point[][]>([]);
  const [maskHistory, setMaskHistory] = useState<string[]>([]);
  const [maskRedo, setMaskRedo] = useState<string[]>([]);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [visible, setVisible] = useState({ radiograph: true, masks: true, boxes: true, expert: true, model: versioned?.phase !== 'independent', dataset: false, cej: true, apex: true, bone_level: true, bone_loss_region: true, other: true });
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const [drawing, setDrawing] = useState(false);
  const lastPaint = useRef<Point | null>(null);
  const original = useImage(imageSrc);
  const maskUrl = maskVersion ? mask.current.toDataURL('image/png') : null;
  useEffect(() => { const observer = new ResizeObserver(() => { if (holder.current) setSize({ width: holder.current.clientWidth, height: Math.max(420, holder.current.clientHeight) }); }); if (holder.current) observer.observe(holder.current); return () => observer.disconnect(); }, []);
  useEffect(() => { mask.current = maskCanvas(width, height); setMaskVersion(0); setDraft([]); setDirty(false); setSelectedId(null); setSelectedTooth(null); setView({ scale: 1, x: 0, y: 0 }); }, [imageId, width, height]);
  useEffect(() => {
    const controller = new AbortController();
    if (versioned) {
      setRecords((versioned.record?.items || []).map(item => ({ id: item.item_id, image_id: imageId,
        tooth_instance_id: item.tooth_instance_id, finding_type: item.finding_type,
        annotation_tool: item.annotation_tool, points: item.points, mask_data_url: item.mask_data_url,
        status: item.status, expert_comment: item.expert_comment, created_at: versioned.record!.created_at,
        updated_at: versioned.record!.created_at })));
      setInstances(versioned.phase === 'correction' ? versioned.rawPrediction?.raw_model_output?.instances || [] : []);
      setVisible(current => ({ ...current, model: versioned.phase === 'correction', dataset: false }));
      if (versioned.phase === 'correction') fetch(`${API}/structural-audit/${encodeURIComponent(imageId)}`, { signal: controller.signal })
        .then(r => r.json()).then(setDataset).catch(() => { if (!controller.signal.aborted) setDataset(null); });
      else setDataset(null);
    } else {
      fetch(`${API}/api/annotations/${encodeURIComponent(imageId)}`, { signal: controller.signal })
        .then(r => { if (!r.ok) throw Error('Could not load annotations'); return r.json(); }).then(setRecords)
        .catch(e => { if (!controller.signal.aborted) setError(String(e)); });
      getToothSegmentation(imageId, imageFile, controller.signal).then(r => setInstances(r.instances))
        .catch(() => { if (!controller.signal.aborted) setInstances([]); });
      fetch(`${API}/structural-audit/${encodeURIComponent(imageId)}`, { signal: controller.signal })
        .then(r => r.json()).then(setDataset).catch(() => { if (!controller.signal.aborted) setDataset(null); });
    }
    return () => controller.abort();
  }, [imageId, imageFile, versioned?.phase, versioned?.record?.record_id, versioned?.rawPrediction?.record_id]);
  useEffect(() => { const listener = (event: BeforeUnloadEvent) => { if (dirty) event.preventDefault(); }; window.addEventListener('beforeunload', listener); return () => window.removeEventListener('beforeunload', listener); }, [dirty]);
  const fitted = fit(width, height, size.width, size.height);
  const scale = fitted.scale * view.scale;
  const offset = { x: (size.width - fitted.width) / 2 + view.x, y: (size.height - fitted.height) / 2 + view.y };
  const active = records.find(r => r.id === selectedId);
  function pointer(): Point | null { const p = stage.current?.getPointerPosition(); return p ? clamp(screenToImage(p, offset, scale), width, height) : null; }
  function paint(p: Point, previous?: Point) { const ctx = mask.current.getContext('2d')!; ctx.strokeStyle = tool === 'eraser' ? '#000' : '#fff'; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = brushSize; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.beginPath(); if (previous) { ctx.moveTo(previous.x, previous.y); ctx.lineTo(p.x, p.y); ctx.stroke(); } else { ctx.arc(p.x, p.y, brushSize / 2, 0, Math.PI * 2); ctx.fill(); } setMaskVersion(v => v + 1); setDirty(true); }
  function down(_e: Konva.KonvaEventObject<PointerEvent>) { if (tool === 'pan' || tool === 'select') return; const p = pointer(); if (!p) return; if (tool === 'point') { setDraft([p]); setDirty(true); } else if (tool === 'polyline' || tool === 'polygon') { if (finished) return; if (tool === 'polygon' && draft.length >= 3 && Math.hypot(p.x - draft[0].x, p.y - draft[0].y) < 12 / scale) { setFinished(true); return; } setDraft(d => [...d, p]); setRedoPoints([]); setDirty(true); } else { setMaskHistory(items => [...items, mask.current.toDataURL('image/png')]); setMaskRedo([]); setDrawing(true); lastPaint.current = p; paint(p); } }
  function move() { if (!drawing) return; const p = pointer(); if (!p) return; paint(p, lastPaint.current ?? undefined); lastPaint.current = p; }
  function finish() { setDrawing(false); lastPaint.current = null; }
  function clear() { setDraft([]); setFinished(false); setRedoPoints([]); setMaskHistory([]); setMaskRedo([]); mask.current = maskCanvas(width, height); setMaskVersion(v => v + 1); setDirty(false); setSelectedId(null); }
  function restoreMask(src: string) { const image = new Image(); image.onload = () => { const canvas = maskCanvas(width, height); canvas.getContext('2d')!.drawImage(image, 0, 0); mask.current = canvas; setMaskVersion(v => v + 1); setDirty(true); }; image.src = src; }
  function undo() { if (tool === 'brush' || tool === 'eraser') { const previous = maskHistory[maskHistory.length - 1]; if (!previous) return; setMaskRedo(items => [...items, mask.current.toDataURL('image/png')]); setMaskHistory(items => items.slice(0, -1)); restoreMask(previous); } else { setRedoPoints(items => [...items, draft]); setDraft(d => d.slice(0, -1)); setFinished(false); setDirty(true); } }
  function redo() { if (tool === 'brush' || tool === 'eraser') { const next = maskRedo[maskRedo.length - 1]; if (!next) return; setMaskHistory(items => [...items, mask.current.toDataURL('image/png')]); setMaskRedo(items => items.slice(0, -1)); restoreMask(next); } else { const next = redoPoints[redoPoints.length - 1]; if (next) { setDraft(next); setRedoPoints(items => items.slice(0, -1)); setDirty(true); } } }
  async function save() {
    if (!dirty) return; const annotationTool = tool === 'eraser' ? 'brush' : tool;
    if (annotationTool === 'pan' || annotationTool === 'select' || (annotationTool !== 'brush' && draft.length < (annotationTool === 'point' ? 1 : annotationTool === 'polyline' ? 2 : 3)) || ((annotationTool === 'polyline' || annotationTool === 'polygon') && !finished)) { setError('Finish the drawing before saving.'); return; }
    setSaving(true); setError('');
    try {
      if (versioned) {
        const id = active?.id || crypto.randomUUID();
        const item: SpatialRecord = {
          id, image_id: imageId, tooth_instance_id: selectedTooth, finding_type: finding,
          annotation_tool: annotationTool, points: annotationTool === 'brush' ? [] : draft,
          mask_data_url: annotationTool === 'brush' ? mask.current.toDataURL('image/png') : null,
          status, expert_comment: comment, created_at: active?.created_at || new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        const next = [...records.filter(r => r.id !== id), item];
        await versioned.onSave(next);
        setRecords(next); clear();
        return;
      }
      const payload = { image_id: imageId, tooth_instance_id: selectedTooth, finding_type: finding, annotation_tool: annotationTool, points: annotationTool === 'brush' ? [] : draft, status, expert_comment: comment };
      const response = await fetch(active ? `${API}/api/annotations/${active.id}` : `${API}/api/annotations`, { method: active ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!response.ok) throw Error(`Save failed (${response.status})`);
      let saved: SpatialRecord = await response.json();
      if (annotationTool === 'brush' && maskVersion) { const blob = await new Promise<Blob>((resolve, reject) => mask.current.toBlob(b => b ? resolve(b) : reject(Error('PNG export failed')), 'image/png')); const form = new FormData(); form.append('file', blob, 'mask.png'); const uploaded = await fetch(`${API}/api/annotations/${saved.id}/mask`, { method: 'POST', body: form }); if (!uploaded.ok) throw Error(`Mask upload failed (${uploaded.status})`); saved = { ...saved, ...(await uploaded.json()) }; }
      setRecords(items => [...items.filter(r => r.id !== saved.id), saved]); clear();
    } catch (e) { setError(String(e)); } finally { setSaving(false); }
  }
  async function remove(id: string) {
    if (versioned) {
      try { const next = records.filter(r => r.id !== id); await versioned.onSave(next); setRecords(next); if (selectedId === id) clear(); }
      catch (e) { setError(String(e)); }
      return;
    }
    const response = await fetch(`${API}/api/annotations/${id}`, { method: 'DELETE' }); if (!response.ok) { setError('Delete failed'); return; } setRecords(items => items.filter(r => r.id !== id)); if (selectedId === id) clear();
  }
  function select(record: SpatialRecord) { if (dirty && !window.confirm('Discard unsaved changes?')) return; clear(); setSelectedId(record.id); setSelectedTooth(record.tooth_instance_id); setFinding(record.finding_type); setTool(record.annotation_tool); setStatus(record.status); setComment(record.expert_comment); setDraft(record.points); setFinished(true); if (record.annotation_tool === 'brush' && (record.mask_path || record.mask_data_url)) { const image = new Image(); image.crossOrigin = 'anonymous'; image.onload = () => { mask.current.getContext('2d')!.drawImage(image, 0, 0, width, height); setMaskVersion(v => v + 1); }; image.src = record.mask_data_url || `${API}/api/annotations/${record.id}/mask`; } }
  function beginNew() { if (dirty && !window.confirm('Discard unsaved changes?')) return; clear(); setComment(''); setStatus('draft'); }
  function changeFinding(value: Finding) { setFinding(value); setTool(preferred[value]); beginNew(); }
  return <section className="spatial-page"><div className="spatial-heading"><div><span className="eyebrow">{versioned?.phase === 'independent' ? 'INDEPENDENT REVIEW' : versioned?.phase === 'correction' ? 'AI-ASSISTED REVIEW' : 'EXPERT SPATIAL ANNOTATION'}</span><h2>{imageId}</h2><p>{versioned ? `Version ${versioned.record?.version || 0} · ${versioned.record?.review_status || 'draft'} · original-image pixel coordinates` : 'Research annotation interface. Annotations represent expert-reviewed research ground truth and are not standalone clinical diagnoses.'}</p>{versioned?.phase === 'correction' && <p>AI-generated—awaiting expert review</p>}</div><button onClick={() => { if (!dirty || window.confirm('Discard unsaved changes?')) onBack(); }}>Back to review</button></div>
    <div className="spatial-layout"><aside className="spatial-side"><h3>Drawing tools</h3><label>Finding<select value={finding} onChange={e => changeFinding(e.target.value as Finding)}>{findings.map(f => <option key={f} value={f}>{f.replace(/_/g, ' ')}</option>)}</select></label><div className="spatial-tools">{tools.map(t => <button key={t} className={tool === t ? 'active' : ''} onClick={() => setTool(t)}>{t}</button>)}</div><label>Brush size: {brushSize} px<input type="range" min="2" max="100" value={brushSize} onChange={e => setBrushSize(Number(e.target.value))} /></label><button onClick={undo} disabled={tool === 'brush' || tool === 'eraser' ? !maskHistory.length : !draft.length}>Undo</button><button onClick={redo} disabled={tool === 'brush' || tool === 'eraser' ? !maskRedo.length : !redoPoints.length}>Redo</button><button onClick={clear}>Clear current drawing</button><button onClick={() => setTool('eraser')}>Erase</button><button onClick={() => { if (draft.length >= (tool === 'polyline' ? 2 : 3)) setFinished(true); }}>Finish line / polygon</button><h3>Overlays</h3>{Object.entries(visible).map(([key, value]) => <label key={key}><input type="checkbox" checked={value} onChange={() => setVisible(v => ({ ...v, [key]: !value }))} /> {key}</label>)}<h3>View</h3><div className="spatial-tools"><button onClick={() => setView(v => ({ ...v, scale: Math.min(8, v.scale * 1.25) }))}>Zoom +</button><button onClick={() => setView(v => ({ ...v, scale: Math.max(.5, v.scale / 1.25) }))}>Zoom −</button><button onClick={() => setView({ scale: 1, x: 0, y: 0 })}>Fit</button></div></aside>
    <div ref={holder} className="spatial-canvas"><Stage ref={stage} width={size.width} height={size.height} onPointerDown={down} onPointerMove={move} onPointerUp={finish} draggable={tool === 'pan'} onDragEnd={e => { if (e.target === stage.current) { setView(v => ({ ...v, x: v.x + e.target.x(), y: v.y + e.target.y() })); e.target.position({ x: 0, y: 0 }); } }}><Layer x={offset.x} y={offset.y} scaleX={scale} scaleY={scale}>{visible.radiograph && original && <KonvaImage image={original} width={width} height={height} listening={false} />}{visible.model && instances.map(item => <Fragment key={`model-${item.instance_id}`}>{visible.masks && <ToothOverlay key={`mask-${item.instance_id}`} src={item.mask_url} width={width} height={height} opacity={selectedTooth === item.instance_id ? .55 : .24} />}{visible.boxes && <Rect key={`box-${item.instance_id}`} x={item.bbox[0]} y={item.bbox[1]} width={item.bbox[2]-item.bbox[0]} height={item.bbox[3]-item.bbox[1]} stroke={selectedTooth === item.instance_id ? '#fff' : '#42ef93'} fill="rgba(0,0,0,0.001)" strokeWidth={2 / scale} onClick={() => { if (tool === 'select') setSelectedTooth(item.instance_id); }} />}</Fragment>)}{visible.dataset && dataset?.keypoints?.CEJ_Points.map((point, i) => <Circle key={`dataset-cej-${i}`} x={point[0]} y={point[1]} radius={4 / scale} fill="#ed69da" listening={false} />)}{visible.dataset && dataset?.keypoints?.Apex_Points.map((point, i) => <Circle key={`dataset-apex-${i}`} x={point[0]} y={point[1]} radius={4 / scale} fill="#ed69da" listening={false} />)}{visible.dataset && dataset?.bone_lines?.Bone_Lines.map((line, i) => <Line key={`dataset-line-${i}`} points={line.flat()} stroke="#ed69da" strokeWidth={2 / scale} listening={false} />)}{visible.expert && records.filter(r => visible[r.finding_type] && !hiddenIds.includes(r.id)).map(r => <Fragment key={`expert-${r.id}`}>{r.annotation_tool === 'brush' && (r.mask_path || r.mask_data_url) && <BinaryOverlay key={r.id} src={r.mask_data_url || `${API}/api/annotations/${r.id}/mask?v=${r.updated_at}`} width={width} height={height} opacity={.55} />}{r.annotation_tool === 'point' ? r.points.map((p, j) => <Circle key={`${r.id}-${j}`} x={p.x} y={p.y} radius={5 / scale} fill={colors[r.finding_type]} onClick={() => { if (tool === 'select') select(r); }} />) : r.points.length > 1 && <Line key={r.id} points={pointsFlat(r.points)} closed={r.annotation_tool === 'polygon'} stroke={colors[r.finding_type]} fill={r.annotation_tool === 'polygon' ? `${colors[r.finding_type]}33` : undefined} strokeWidth={3 / scale} onClick={() => { if (tool === 'select') select(r); }} />}</Fragment>)}{maskUrl && <BinaryOverlay src={maskUrl} width={width} height={height} opacity={.55} />}{draft.length > 1 && tool !== 'brush' && <Line points={pointsFlat(draft)} stroke={colors[finding]} closed={tool === 'polygon'} strokeWidth={3 / scale} listening={false} />}{draft.map((p, i) => tool !== 'brush' && <Circle key={i} x={p.x} y={p.y} radius={5 / scale} fill={colors[finding]} draggable onDragEnd={e => { const p = clamp({ x: e.target.x(), y: e.target.y() }, width, height); setDraft(d => d.map((item, j) => j === i ? p : item)); setDirty(true); }} onDblClick={() => { setDraft(d => d.filter((_, j) => j !== i)); setDirty(true); }} />)}<Text x={8} y={height - 22} text="MODEL OUTPUT: green  |  EXPERT ANNOTATION: finding colors" fontSize={14} fill="white" listening={false} /></Layer></Stage></div>
    <aside className="spatial-side"><h3>Tooth instance</h3><small>Use Select to click a model box, or choose an instance below.</small><select value={selectedTooth ?? ''} onChange={e => setSelectedTooth(e.target.value ? Number(e.target.value) : null)}><option value="">Whole image / unassigned</option>{instances.map(i => <option key={i.instance_id} value={i.instance_id}>Instance {i.instance_id}</option>)}</select><p>Image ID: {imageId}</p><h3>Expert annotations</h3><button onClick={beginNew}>New annotation</button><div className="spatial-list">{records.map(r => <div key={r.id} className={selectedId === r.id ? 'selected' : ''}><button onClick={() => select(r)}>{r.finding_type.replace(/_/g, ' ')} · {r.annotation_tool} · {r.status} · tooth {r.tooth_instance_id ?? 'unassigned'}</button><button onClick={() => setHiddenIds(ids => ids.includes(r.id) ? ids.filter(id => id !== r.id) : [...ids, r.id])}>{hiddenIds.includes(r.id) ? 'Show' : 'Hide'}</button><button onClick={() => void remove(r.id)}>Delete</button></div>)}</div><label>Status<select value={status} onChange={e => { setStatus(e.target.value as Status); setDirty(true); }}>{statuses.map(s => <option key={s}>{s}</option>)}</select></label><label>Expert comment<textarea value={comment} onChange={e => { setComment(e.target.value); setDirty(true); }} /></label>{dirty && <strong className="spatial-dirty">Unsaved changes</strong>}{error && <p role="alert">{error}</p>}<button className="primary-btn" disabled={saving || !dirty} onClick={() => void save()}>{saving ? 'Saving…' : 'Save current'}</button><button disabled={saving || !dirty} onClick={() => void save()}>Save all</button><small>Only confirmed records are eligible for later ground-truth use.</small></aside></div></section>;
}
