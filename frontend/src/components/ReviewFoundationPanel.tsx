import { useCallback, useEffect, useState } from 'react';
import AnnotationCanvas from './annotation/AnnotationCanvas';
import type { SpatialRecord } from './annotation/model';
import { AI_STATUS, createPrediction, getHistory, revealAi, saveApproval, saveVersion,
  type ReviewHistory, type ReviewMode, type ReviewRecord, type ReviewState } from '../lib/reviewFoundation';

interface Props {
  imageId: string; imageSrc: string; imageFile: File; width: number; height: number;
  mode: ReviewMode; onBack: () => void; onState: (state: ReviewState) => void;
}

function latest(records: ReviewRecord[], kind: ReviewRecord['record_type']): ReviewRecord | null {
  return records.filter(record => record.record_type === kind)
    .sort((left, right) => right.version - left.version)[0] || null;
}

export default function ReviewFoundationPanel({ imageId, imageSrc, imageFile, width, height, mode, onBack, onState }: Props) {
  const [history, setHistory] = useState<ReviewHistory | null>(null);
  const [reviewer, setReviewer] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const refresh = useCallback(async () => {
    const current = await getHistory(imageId);
    setHistory(current); onState(current.review_state);
    return current;
  }, [imageId, onState]);

  useEffect(() => { let active = true; getHistory(imageId).then(current => {
    if (active) { setHistory(current); onState(current.review_state); }
  }).catch(reason => { if (active) setError(String(reason)); }); return () => { active = false; }; }, [imageId, onState]);

  const records = history?.records || [];
  const source = latest(records, 'source_record');
  const raw = latest(records, 'raw_ai_prediction');
  const independent = latest(records, 'independent_expert_annotation');
  const correction = latest(records, 'expert_correction');
  const state = history?.review_state;
  const blind = mode === 'independent_evaluation' && !state?.revealed;
  const phase = blind ? 'independent' : 'correction';
  const working = phase === 'independent' ? independent : correction || independent;

  async function run(action: () => Promise<unknown>, success: string) {
    setBusy(true); setError('');
    try { await action(); await refresh(); setNotice(success); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }

  async function saveItems(items: SpatialRecord[]): Promise<ReviewRecord> {
    if (!reviewer.trim()) throw new Error('Enter a reviewer ID before saving.');
    if (!source) throw new Error('Source record is missing.');
    const payload = {
      created_by: reviewer.trim(), parent_record_id: phase === 'independent'
        ? independent?.record_id || source.record_id : correction?.record_id || raw?.record_id,
      raw_prediction_record_id: phase === 'correction' ? raw?.record_id : null,
      independent_annotation_record_id: phase === 'correction' ? independent?.record_id : null,
      items: items.map(item => ({ item_id: item.id, tooth_instance_id: item.tooth_instance_id,
        finding_type: item.finding_type, annotation_tool: item.annotation_tool, points: item.points,
        mask_data_url: item.mask_data_url || null, status: item.status, expert_comment: item.expert_comment })),
      teeth: [], findings: [], notes: '', review_status: 'draft',
      uncertainty_reasons: items.length ? [] : ['all_annotations_removed_or_not_measurable'],
    };
    const saved = await saveVersion(imageId, phase === 'independent' ? 'independent-annotations' : 'corrections', payload);
    await refresh(); setNotice(`${phase === 'independent' ? 'Independent annotation' : 'Expert correction'} version ${saved.version} saved.`);
    return saved;
  }

  async function showAi() {
    if (!reviewer.trim()) { setError('Enter a reviewer ID before revealing AI.'); return; }
    await run(async () => {
      await revealAi(imageId, reviewer.trim());
      const current = await getHistory(imageId);
      if (!latest(current.records, 'raw_ai_prediction')) await createPrediction(imageId);
    }, 'AI output revealed. Independent annotation remains unchanged.');
  }

  async function prepareRoutinePrediction() {
    await run(async () => { if (!raw) await createPrediction(imageId); }, 'Raw Mask R-CNN prediction saved.');
  }

  async function approve(decision: 'approved' | 'rejected' | 'uncertain' | 'needs_revision') {
    const target = correction || independent;
    if (!target) { setError('Save an annotation before recording a decision.'); return; }
    if (!reviewer.trim()) { setError('Enter a reviewer ID before recording a decision.'); return; }
    if ((decision === 'approved' || decision === 'rejected') && !window.confirm(`Record final decision: ${decision}?`)) return;
    await run(() => saveApproval(imageId, { created_by: reviewer.trim(), approved_record_id: target.record_id,
      decision, notes: '', uncertainty_reasons: decision === 'uncertain' ? ['expert_marked_uncertain'] : [] }),
      `Final decision ${decision} saved as a new record.`);
  }

  return <div>
    <section className="review-foundation-card" style={{ marginBottom: 16 }}>
      <strong>{blind ? 'Independent Review' : 'AI-Assisted Review'}</strong>
      <span>AI-assisted radiographic assessment · provisional radiographic finding</span>
      <span>{mode === 'independent_evaluation' ? 'Designated evaluation image' : 'Routine non-blinded image'}</span>
      <label>Reviewer ID <input value={reviewer} onChange={event => setReviewer(event.target.value)} placeholder="Enter reviewer ID" /></label>
      {blind ? <p>Only the original image and your independent annotations are available until an annotation is saved and AI is revealed.</p>
        : <p>{AI_STATUS}. These are provisional radiographic findings, not a periodontal diagnosis.</p>}
      {state?.can_reveal && blind && <button className="secondary-btn" disabled={busy} onClick={() => void showAi()}>Reveal AI after independent annotation</button>}
      {mode === 'ai_assisted' && !raw && <button className="secondary-btn" disabled={busy} onClick={() => void prepareRoutinePrediction()}>Save raw Mask R-CNN prediction</button>}
      {working && <p>Annotation version {working.version} · review status: {working.review_status}</p>}
      {working?.uncertainty_reasons.length ? <p>Uncertainty: {working.uncertainty_reasons.join(', ')}</p> : null}
      {notice && <p role="status">{notice}</p>}{error && <p role="alert">{error}</p>}
    </section>
    {source && (phase === 'independent' || raw) && <AnnotationCanvas key={`${imageId}-${phase}`} imageId={imageId} imageSrc={imageSrc} imageFile={imageFile}
      width={width} height={height} onBack={onBack} versioned={{ phase, record: working, rawPrediction: raw, onSave: saveItems }} />}
    {!blind && <section className="review-foundation-card" style={{ marginTop: 16 }}><h3>Final review decision</h3>
      <div className="casebar-actions">
        <button disabled={busy} onClick={() => void approve('approved')}>Approve</button>
        <button disabled={busy} onClick={() => void approve('rejected')}>Reject</button>
        <button disabled={busy} onClick={() => void approve('uncertain')}>Uncertain</button>
        <button disabled={busy} onClick={() => void approve('needs_revision')}>Needs revision</button>
      </div></section>}
    <section className="review-foundation-card" style={{ marginTop: 16 }}><h3>Record history</h3>
      {records.map(record => <div key={record.record_id}>{record.record_type} · v{record.version} · {record.review_status} · {record.created_at} · {record.created_by} · {record.record_id}{record.uncertainty_reasons.length ? ` · uncertainty: ${record.uncertainty_reasons.join(', ')}` : ''}</div>)}
    </section>
  </div>;
}
