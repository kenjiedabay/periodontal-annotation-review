import { useState, type CSSProperties } from 'react';
import { saveAnnotation } from '../lib/annotations';
import StructuralAnalysisPanel from './StructuralAnalysisPanel';
import type { AnnotationDraft, AnnotationPayload, ValidationStatus } from '../types';

interface ExpertValidationPanelProps {
  imageId: string;
  imageName: string;
  originalSrc: string;
  processedSrc?: string;
  annotation: AnnotationDraft;
  onChange: (patch: Partial<AnnotationDraft>) => void;
  onEdit: () => void;
  onSaved: (timestamp: string, status: ValidationStatus) => void;
}

const statusToApi: Record<ValidationStatus, AnnotationPayload['validation_status']> = {
  Pending: 'pending',
  Validated: 'validated',
  'Needs Revision': 'needs_revision',
  Uncertain: 'uncertain',
};

function toPayload(imageId: string, annotation: AnnotationDraft, status: ValidationStatus): AnnotationPayload {
  return {
    image_id: imageId,
    disease_status: annotation.status.toLowerCase() as AnnotationPayload['disease_status'],
    affected_teeth: annotation.teeth,
    severity: annotation.severity.toLowerCase().replace(/ /g, '_') as AnnotationPayload['severity'],
    findings: annotation.findings,
    regions: annotation.spatial.boundingBox ? [{ type: 'bounding_box', coordinates: [annotation.spatial.boundingBox.x, annotation.spatial.boundingBox.y, annotation.spatial.boundingBox.width, annotation.spatial.boundingBox.height] }] : annotation.spatial.polygon.length ? [{ type: annotation.spatial.method === 'Polygon / freehand region' ? 'freehand' : 'polygon', coordinates: annotation.spatial.polygon.map((point) => [point.x, point.y]) }] : [],
    expert_comment: annotation.notes,
    expert_validated: status === 'Validated',
    validation_status: statusToApi[status],
    validation_timestamp: new Date().toISOString(),
  };
}

export default function ExpertValidationPanel({ imageId, imageName, originalSrc, processedSrc, annotation, onChange, onEdit, onSaved }: ExpertValidationPanelProps) {
  const [decision, setDecision] = useState<ValidationStatus>(annotation.validationStatus || 'Pending');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const overlayStyle: CSSProperties = { backgroundImage: `url(${originalSrc})` };
  const region = annotation.spatial.boundingBox;
  const polygon = annotation.spatial.polygon.map((point) => `${point.x},${point.y}`).join(' ');

  function chooseDecision(status: ValidationStatus) {
    setDecision(status);
    onChange({ validationStatus: status, validated: status === 'Validated' });
  }

  async function saveValidation() {
    setSaving(true);
    setError(null);
    const timestamp = new Date().toISOString();
    try {
      await saveAnnotation(toPayload(imageId, annotation, decision));
      onChange({ validationStatus: decision, validated: decision === 'Validated', validationTimestamp: timestamp, savedAt: timestamp });
      onSaved(timestamp, decision);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Validation could not be saved.');
    } finally {
      setSaving(false);
    }
  }

  return <section className="validation-module">
    <div className="validation-method-banner">GROUND TRUTH IS ESTABLISHED THROUGH DENTAL EXPERT VALIDATION.</div>
    <div className="validation-columns">
      <section className="validation-image-column"><div className="validation-column-heading"><span className="eyebrow">LEFT / SOURCE</span><strong>Original radiograph</strong></div><div className="validation-image-frame"><img src={originalSrc} alt={`Original radiograph ${imageName}`} /></div><small>Image ID: {imageId}</small></section>
      <section className="validation-image-column"><div className="validation-column-heading"><span className="eyebrow">CENTER / REVIEW</span><strong>Annotation overlay</strong></div><div className="validation-image-frame overlay-frame" style={overlayStyle}><svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Annotation region overlay">{region && region.width > 0 && <rect className="annotation-box-shape" x={region.x} y={region.y} width={region.width} height={region.height} />}{annotation.spatial.polygon.length > 1 && <polyline className="annotation-polygon-shape" points={polygon} />}</svg></div><small>{processedSrc ? 'Processed display available for comparison.' : 'Processed display copy not attached.'}</small></section>
      <aside className="validation-assessment"><div className="validation-column-heading"><span className="eyebrow">RIGHT / EXPERT REVIEW</span><strong>Expert assessment</strong></div><div className="validation-ground-truth-label">EXPERT VALIDATION · GROUND TRUTH</div><dl className="validation-details"><div><dt>Image ID</dt><dd>{imageId}</dd></div><div><dt>Disease status</dt><dd>{annotation.status}</dd></div><div><dt>Affected tooth / region</dt><dd>{annotation.teeth.length ? annotation.teeth.join(', ') : 'Region not assigned'}</dd></div><div><dt>Severity</dt><dd>{annotation.severity}</dd></div><div><dt>Radiographic findings</dt><dd>{annotation.findings.length ? annotation.findings.join(', ') : 'None selected'}</dd></div></dl><label className="validation-comment"><span>Expert comments</span><textarea value={annotation.notes} onChange={(event) => onChange({ notes: event.target.value })} rows={5} placeholder="Record the reasoning for this validation decision." /></label><div className="validation-status-section"><span className="eyebrow">REVIEW STATUS</span><div className="validation-status-grid">{(['Pending', 'Validated', 'Needs Revision', 'Uncertain'] as ValidationStatus[]).map((status) => <button key={status} className={decision === status ? `status-selected status-${status.toLowerCase().replace(' ', '-')}` : ''} onClick={() => chooseDecision(status)}>{status}</button>)}</div></div>{error && <div className="save-error" role="alert">{error}</div>}<div className="validation-actions"><button className="text-btn" onClick={onEdit}>Edit Annotation</button><button className="secondary-btn" onClick={() => chooseDecision('Uncertain')}>Mark Uncertain</button><button className="primary-btn" onClick={() => chooseDecision('Validated')}>Confirm Annotation</button><button className="primary-btn full" onClick={() => void saveValidation()} disabled={saving}>{saving ? 'Saving validation…' : 'Save Validation'}</button></div></aside>
    </div>
    <StructuralAnalysisPanel imageId={imageId} imageSrc={originalSrc} annotation={annotation} />
  </section>;
}
