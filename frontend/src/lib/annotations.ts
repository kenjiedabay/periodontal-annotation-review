import type { AnnotationPayload, ModelAnalysisResponse, StructuralAnalysisResponse, StructuralAuditRecord } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export async function getToothSegmentation(imageId: string, file: File, signal?: AbortSignal): Promise<import('../types').ToothSegmentationResponse> {
  const body = new FormData();
  body.append('file', file);
  const response = await fetch(`${API_URL}/analysis/tooth-segmentation?image_id=${encodeURIComponent(imageId)}`, { method: 'POST', body, signal });
  if (!response.ok) throw new Error(`Tooth segmentation failed (${response.status}).`);
  return response.json();
}

export async function getPerioKptPreview(imageId: string, file: File, signal?: AbortSignal): Promise<import('../types').PerioKptPreviewResponse> {
  const body = new FormData(); body.append('file', file);
  const response = await fetch(`${API_URL}/analysis/perio-kpt-preview?image_id=${encodeURIComponent(imageId)}`, {method:'POST', body, signal});
  if (!response.ok) throw new Error(`Perio-KPT preview failed (${response.status}).`);
  return response.json();
}

export async function getUnifiedModelReview(imageId: string, file: File, signal?: AbortSignal): Promise<import('../types').UnifiedModelReviewResponse> {
  const body = new FormData(); body.append('file', file);
  const response = await fetch(`${API_URL}/analysis/unified-model-review?image_id=${encodeURIComponent(imageId)}`, {method:'POST', body, signal});
  if (!response.ok) throw new Error(`Unified model review failed (${response.status}).`);
  return response.json();
}

export async function saveAnnotation(payload: AnnotationPayload): Promise<AnnotationPayload & { saved_at?: string }> {
  const response = await fetch(`${API_URL}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (response.status === 409) {
    const replacement = await fetch(`${API_URL}/annotations/${encodeURIComponent(payload.image_id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!replacement.ok) {
      throw new Error(`Backend rejected the annotation update (${replacement.status}).`);
    }
    return (await replacement.json()) as AnnotationPayload & { saved_at?: string };
  }
  if (!response.ok) {
    throw new Error(`Backend rejected the annotation (${response.status}).`);
  }
  return (await response.json()) as AnnotationPayload & { saved_at?: string };
}

export async function getStructuralAnalysis(imageId: string): Promise<StructuralAnalysisResponse> {
  const response = await fetch(`${API_URL}/structural-analysis/${encodeURIComponent(imageId)}`);
  if (!response.ok) {
    throw new Error(`Structural analysis endpoint returned ${response.status}.`);
  }
  return (await response.json()) as StructuralAnalysisResponse;
}

export async function getModelAnalysis(imageId: string): Promise<ModelAnalysisResponse> {
  const response = await fetch(`${API_URL}/analysis/${encodeURIComponent(imageId)}`);
  if (!response.ok) {
    throw new Error(`Model analysis endpoint returned ${response.status}.`);
  }
  return (await response.json()) as ModelAnalysisResponse;
}

export async function getStructuralAudit(imageId: string): Promise<StructuralAuditRecord> {
  const response = await fetch(`${API_URL}/structural-audit/${encodeURIComponent(imageId)}`);
  if (!response.ok) throw new Error(`Structural audit endpoint returned ${response.status}.`);
  return (await response.json()) as StructuralAuditRecord;
}
