import { useEffect, useState } from 'react';
import { getStructuralAnalysis } from '../lib/annotations';
import type { AnnotationDraft, StructuralAnalysisResponse } from '../types';

interface StructuralAnalysisPanelProps {
  imageId: string;
  imageSrc: string;
  annotation: AnnotationDraft;
}

export default function StructuralAnalysisPanel({ imageId, imageSrc, annotation }: StructuralAnalysisPanelProps) {
  const [response, setResponse] = useState<StructuralAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const region = annotation.spatial.boundingBox;
  const polygon = annotation.spatial.polygon.map((point) => `${point.x},${point.y}`).join(' ');

  useEffect(() => {
    let active = true;
    void getStructuralAnalysis(imageId).then((value) => { if (active) setResponse(value); }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Structural analysis is unavailable.'); });
    return () => { active = false; };
  }, [imageId]);

  const groundTruthAvailable = annotation.validationStatus === 'Validated' || annotation.validated;
  return <section className="structural-analysis-module">
    <div className="structural-analysis-heading"><div><span className="eyebrow">STRUCTURAL ANALYSIS / RESEARCH VIEW</span><h3>Expert findings and model boundary</h3></div><span className="pill amber">No clinical conclusion</span></div>
    <div className="structural-flow">
      <section className="structural-step"><span className="structural-step-number">01</span><div className="structural-step-content"><strong>CURRENT RADIOGRAPH</strong><div className="structural-image"><img src={imageSrc} alt={`Current radiograph ${imageId}`} /></div></div></section>
      <div className="structural-arrow">↓</div>
      <section className="structural-step"><span className="structural-step-number">02</span><div className="structural-step-content"><strong>AFFECTED REGION</strong><div className="structural-region-view"><img src={imageSrc} alt="Radiograph with expert region overlay" /><svg viewBox="0 0 1 1" preserveAspectRatio="none">{region && region.width > 0 && <rect className="annotation-box-shape" x={region.x} y={region.y} width={region.width} height={region.height} />}{annotation.spatial.polygon.length > 1 && <polyline className="annotation-polygon-shape" points={polygon} />}</svg></div><small>{annotation.spatial.method}: expert-drawn geometry</small></div></section>
      <div className="structural-arrow">↓</div>
      <section className="structural-step ground-truth-step"><span className="structural-step-number">03</span><div className="structural-step-content"><strong>EXPERT-VALIDATED FINDINGS</strong>{groundTruthAvailable ? <div className="structural-tags">{annotation.findings.length ? annotation.findings.map((finding) => <span key={finding}>{finding}</span>) : <span className="empty-tag">No finding selected</span>}</div> : <p className="structural-muted">Findings are not ground truth until expert validation is saved.</p>}</div></section>
      <div className="structural-arrow">↓</div>
      <section className="structural-step model-step"><span className="structural-step-number">04</span><div className="structural-step-content"><strong>MODEL ANALYSIS</strong><div className="model-boundary"><span className="model-status">NOT AVAILABLE</span><p>No structural model has been trained. This panel intentionally generates no prediction or unsupported medical conclusion.</p>{response?.preliminary_result && <small>Preliminary result: {response.preliminary_result}</small>}{response?.model_prediction && <small>Model prediction: {response.model_prediction}</small>}{error && <small>{error}</small>}</div></div></section>
    </div>
  </section>;
}
