import type { ToothSegmentationResponse } from '../types';
import type { SpatialRecord } from '../components/annotation/model';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
export const AI_STATUS = 'AI-generated—awaiting expert review';
export type ReviewMode = 'independent_evaluation' | 'ai_assisted';
export type ReviewKind = 'source_record' | 'raw_ai_prediction' | 'independent_expert_annotation' | 'expert_correction' | 'final_approval';

export interface ReviewRecord {
  schema_version: number; record_id: string; record_type: ReviewKind; version: number;
  image_id: string; image_hash: string; source_dataset: string; source_partition: string;
  original_width: number; original_height: number;
  coordinate_space: 'original_image_pixels'; coordinate_origin: 'top_left';
  created_at: string; created_by: string; parent_record_id: string | null;
  model_name: string | null; model_version: string | null; review_status: string;
  uncertainty_reasons: string[]; review_mode?: ReviewMode;
  raw_model_output?: ToothSegmentationResponse;
  items?: Array<{ item_id: string; tooth_instance_id: number | null; finding_type: SpatialRecord['finding_type']; annotation_tool: SpatialRecord['annotation_tool']; points: SpatialRecord['points']; mask_data_url?: string | null; status: SpatialRecord['status']; expert_comment: string }>;
  raw_prediction_record_id?: string; independent_annotation_record_id?: string;
  approved_record_id?: string; decision?: string;
}

export interface ReviewState {
  image_id: string; mode: ReviewMode; independent_saved: boolean;
  revealed: boolean; can_reveal: boolean; revealed_at: string | null; revealed_by: string | null;
}
export interface ReviewHistory { image_id: string; review_state: ReviewState; records: ReviewRecord[] }

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Review request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function registerSource(imageId: string, file: File, mode: ReviewMode, reviewer: string): Promise<ReviewRecord> {
  const form = new FormData();
  form.append('image_id', imageId); form.append('review_mode', mode);
  form.append('created_by', reviewer); form.append('file', file);
  return json<ReviewRecord>(await fetch(`${API}/api/review/sources`, { method: 'POST', body: form }));
}
export async function getHistory(imageId: string): Promise<ReviewHistory> {
  return json<ReviewHistory>(await fetch(`${API}/api/review/sources/${encodeURIComponent(imageId)}/history`));
}
export async function revealAi(imageId: string, reviewer: string): Promise<ReviewState> {
  const form = new FormData(); form.append('created_by', reviewer);
  return json<ReviewState>(await fetch(`${API}/api/review/sources/${encodeURIComponent(imageId)}/reveal`, { method: 'POST', body: form }));
}
export async function createPrediction(imageId: string): Promise<ReviewRecord | { record_id: string }> {
  const form = new FormData(); form.append('created_by', 'Mask R-CNN service');
  return json<ReviewRecord | { record_id: string }>(await fetch(`${API}/api/review/sources/${encodeURIComponent(imageId)}/predictions`, { method: 'POST', body: form }));
}
export async function saveVersion(imageId: string, kind: 'independent-annotations' | 'corrections', payload: unknown): Promise<ReviewRecord> {
  return json<ReviewRecord>(await fetch(`${API}/api/review/sources/${encodeURIComponent(imageId)}/${kind}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }));
}
export async function saveApproval(imageId: string, payload: unknown): Promise<ReviewRecord> {
  return json<ReviewRecord>(await fetch(`${API}/api/review/sources/${encodeURIComponent(imageId)}/approvals`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }));
}
