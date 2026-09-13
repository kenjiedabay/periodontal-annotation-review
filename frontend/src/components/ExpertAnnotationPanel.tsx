import { useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react';
import { saveAnnotation } from '../lib/annotations';
import { annotationLabels } from '../lib/annotationConfig';
import type { AnnotationDraft, AnnotationMethod, AnnotationPayload, Point } from '../types';

type InteractionMode = 'annotate' | 'pan';

interface ExpertAnnotationPanelProps {
  imageId: string;
  imageName: string;
  imageSrc: string | null;
  annotation: AnnotationDraft;
  onChange: (patch: Partial<AnnotationDraft>) => void;
  onSaved: (savedAt: string) => void;
  onContinue: () => void;
  findings?: readonly string[];
  teeth?: readonly string[];
  canPrevious?: boolean;
  canNext?: boolean;
  imageIndex?: number;
  imageCount?: number;
  onPrevious?: () => void;
  onNext?: () => void;
}

function normalizedPoint(event: ReactPointerEvent<SVGSVGElement>, element: SVGSVGElement): Point {
  const bounds = element.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
    y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
  };
}

function distance(left: Point, right: Point): number {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

export default function ExpertAnnotationPanel({
  imageId,
  imageName,
  imageSrc,
  annotation,
  onChange,
  onSaved,
  onContinue,
  findings = annotationLabels.findings,
  teeth = annotationLabels.fdiTeeth,
  canPrevious = false,
  canNext = false,
  imageIndex = 0,
  imageCount = 1,
  onPrevious,
  onNext,
}: ExpertAnnotationPanelProps) {
  const [zoom, setZoom] = useState(1);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('annotate');
  const [isDrawing, setIsDrawing] = useState(false);
  const [boxStart, setBoxStart] = useState<Point | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const dragStart = useRef<{ x: number; y: number; pan: Point } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  function resetViewer() {
    setZoom(1);
    setBrightness(100);
    setContrast(100);
    setPan({ x: 0, y: 0 });
  }

  function updateSpatial(spatial: AnnotationDraft['spatial']) {
    onChange({ spatial, region: spatial.method });
  }

  function handlePointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    if (!svgRef.current || interactionMode === 'pan') {
      dragStart.current = { x: event.clientX, y: event.clientY, pan };
      return;
    }
    const point = normalizedPoint(event, svgRef.current);
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsDrawing(true);
    if (annotation.region === 'Bounding box') {
      setBoxStart(point);
      updateSpatial({ method: annotation.region, boundingBox: { x: point.x, y: point.y, width: 0, height: 0 }, polygon: [] });
    } else {
      updateSpatial({ method: annotation.region, boundingBox: null, polygon: [point] });
    }
  }

  function handlePointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (interactionMode === 'pan' && dragStart.current) {
      setPan({ x: dragStart.current.pan.x + event.clientX - dragStart.current.x, y: dragStart.current.pan.y + event.clientY - dragStart.current.y });
      return;
    }
    if (!isDrawing || !svgRef.current) return;
    const point = normalizedPoint(event, svgRef.current);
    if (annotation.region === 'Bounding box' && boxStart) {
      updateSpatial({ method: annotation.region, boundingBox: { x: Math.min(boxStart.x, point.x), y: Math.min(boxStart.y, point.y), width: Math.abs(point.x - boxStart.x), height: Math.abs(point.y - boxStart.y) }, polygon: [] });
    } else {
      const previous = annotation.spatial.polygon[annotation.spatial.polygon.length - 1];
      if (!previous || distance(previous, point) > 0.008) updateSpatial({ method: annotation.region, boundingBox: null, polygon: [...annotation.spatial.polygon, point] });
    }
  }

  function finishPointer() {
    setIsDrawing(false);
    setBoxStart(null);
    dragStart.current = null;
  }

  function chooseMethod(method: AnnotationMethod) {
    onChange({ region: method, spatial: { method, boundingBox: null, polygon: [] } });
  }

  function toggleFinding(finding: string) {
    const next = annotation.findings.includes(finding) ? annotation.findings.filter((item) => item !== finding) : [...annotation.findings, finding];
    onChange({ findings: next });
  }

  async function handleSave() {
    const payload: AnnotationPayload = {
      image_id: imageId,
      disease_status: annotation.status.toLowerCase() as AnnotationPayload['disease_status'],
      affected_teeth: annotation.teeth,
      severity: annotation.severity.toLowerCase().replace(/ /g, '_') as AnnotationPayload['severity'],
      findings: annotation.findings,
      regions: annotation.spatial.boundingBox ? [{ type: 'bounding_box', coordinates: [annotation.spatial.boundingBox.x, annotation.spatial.boundingBox.y, annotation.spatial.boundingBox.width, annotation.spatial.boundingBox.height] }] : annotation.spatial.polygon.length ? [{ type: annotation.spatial.method === 'Polygon / freehand region' ? 'freehand' : 'polygon', coordinates: annotation.spatial.polygon.map((point) => [point.x, point.y]) }] : [],
      expert_comment: annotation.notes,
      expert_validated: false,
      validation_status: 'pending',
      validation_timestamp: undefined,
    };
    setSaving(true);
    setSaveError(null);
    try {
      const response = await saveAnnotation(payload);
      onSaved(response.saved_at || new Date().toISOString());
      onContinue();
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'The annotation could not be saved.');
    } finally {
      setSaving(false);
    }
  }

  const imageStyle: CSSProperties = {
    filter: `brightness(${brightness}%) contrast(${contrast}%)`,
  };
  const viewportStyle: CSSProperties = { transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` };
  const box = annotation.spatial.boundingBox;
  const polygonPoints = annotation.spatial.polygon.map((point) => `${point.x},${point.y}`).join(' ');

  return <>
    <div className="expert-ground-truth-label">EXPERT ANNOTATION — GROUND TRUTH</div>
    <div className="annotation-workbench">
      <section className="annotation-viewer-panel">
        <div className="annotation-viewer-toolbar">
          <button onClick={() => setZoom((value) => Math.max(0.5, value - 0.1))} aria-label="Zoom out">−</button>
          <strong>{Math.round(zoom * 100)}%</strong>
          <button onClick={() => setZoom((value) => Math.min(4, value + 0.1))} aria-label="Zoom in">+</button>
          <button onClick={resetViewer}>Reset</button>
          <span className="tool-spacer" />
          <button className={interactionMode === 'annotate' ? 'selected' : ''} onClick={() => setInteractionMode('annotate')}>Annotate</button>
          <button className={interactionMode === 'pan' ? 'selected' : ''} onClick={() => setInteractionMode('pan')}>Pan</button>
          <button onClick={onPrevious} disabled={!canPrevious} aria-label="Previous image">←</button>
          <span className="image-counter">{imageIndex + 1} / {imageCount}</span>
          <button onClick={onNext} disabled={!canNext} aria-label="Next image">→</button>
        </div>
        <div className="annotation-viewport" style={{ cursor: interactionMode === 'pan' ? 'grab' : 'crosshair' }}>
          <div className="annotation-canvas" style={viewportStyle}>
            {imageSrc ? <img src={imageSrc} style={imageStyle} alt={`Original radiograph ${imageName}`} /> : <div className="empty-viewer"><b>No radiograph loaded</b><span>Return to Image intake to add a source image.</span></div>}
            <svg ref={svgRef} className="annotation-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={finishPointer} onPointerCancel={finishPointer}>
              {box && box.width > 0 && box.height > 0 && <rect className="annotation-box-shape" x={box.x} y={box.y} width={box.width} height={box.height} />}
              {annotation.spatial.polygon.length > 1 && <polyline className="annotation-polygon-shape" points={polygonPoints} />}
            </svg>
          </div>
        </div>
        <div className="annotation-viewer-footer">
          <label>Brightness <input type="range" min="50" max="150" value={brightness} onChange={(event) => setBrightness(Number(event.target.value))} /></label>
          <label>Contrast <input type="range" min="50" max="160" value={contrast} onChange={(event) => setContrast(Number(event.target.value))} /></label>
          <span className="preserved-note">Original radiograph preserved</span>
        </div>
      </section>
      <aside className="expert-form-panel">
        <div className="annotation-form-heading"><span className="eyebrow">RESEARCH LABELS</span><span className="pill teal">Configurable</span></div>
        <div className="form-section"><span className="eyebrow">DISEASE STATUS</span><div className="segmented">{annotationLabels.diseaseStatuses.map((status) => <button key={status} className={annotation.status === status ? 'selected' : ''} onClick={() => onChange({ status })}>{status}</button>)}</div></div>
        <div className="form-section"><span className="eyebrow">AFFECTED TOOTH · FDI · MULTI-SELECT</span><div className="tooth-grid">{teeth.map((tooth) => <button key={tooth} className={annotation.teeth.includes(tooth) ? 'selected' : ''} onClick={() => onChange({ teeth: annotation.teeth.includes(tooth) ? annotation.teeth.filter((item) => item !== tooth) : [...annotation.teeth, tooth] })}>{tooth}</button>)}</div></div>
        <div className="form-section"><span className="eyebrow">SEVERITY CATEGORY</span><select value={annotation.severity} onChange={(event) => onChange({ severity: event.target.value as AnnotationDraft['severity'] })}>{annotationLabels.severities.map((severity) => <option key={severity}>{severity}</option>)}</select></div>
        <div className="form-section"><span className="eyebrow">SPATIAL ANNOTATION METHOD</span><div className="method-switch"><button className={annotation.region === 'Bounding box' ? 'selected' : ''} onClick={() => chooseMethod('Bounding box')}>Bounding box</button><button className={annotation.region === 'Polygon / freehand region' ? 'selected' : ''} onClick={() => chooseMethod('Polygon / freehand region')}>Polygon / freehand</button></div><small className="field-help">Use Annotate mode to draw on the radiograph. The method can be changed after expert guidance.</small></div>
        <div className="form-section"><span className="eyebrow">RADIOGRAPHIC FINDINGS</span><div className="finding-list">{findings.map((finding) => <button key={finding} className={annotation.findings.includes(finding) ? 'selected' : ''} onClick={() => toggleFinding(finding)}><i>✓</i>{finding}</button>)}</div><textarea value={annotation.notes} onChange={(event) => onChange({ notes: event.target.value })} placeholder="Free-text expert observation" rows={3} /></div>
        {saveError && <div className="save-error" role="alert">{saveError}</div>}
        <button className="primary-btn full" onClick={handleSave} disabled={saving || !imageSrc}>{saving ? 'Saving annotation…' : 'Save annotation to backend'} <span>→</span></button>
      </aside>
    </div>
  </>;
}
