export type Tool = 'select' | 'point' | 'polyline' | 'polygon' | 'brush' | 'eraser' | 'pan';
export type Finding = 'cej' | 'apex' | 'bone_level' | 'bone_loss_region' | 'other';
export type Status = 'draft' | 'confirmed' | 'uncertain' | 'rejected' | 'cannot_determine';
export type Point = { x: number; y: number };
export interface SpatialRecord {
  id: string; image_id: string; tooth_instance_id: number | null;
  finding_type: Finding; annotation_tool: Exclude<Tool, 'pan' | 'eraser' | 'select'>;
  points: Point[]; mask_path?: string | null; mask_data_url?: string | null; status: Status;
  expert_comment: string; created_at: string; updated_at: string;
}
export function fit(w: number, h: number, maxW: number, maxH: number) {
  const scale = Math.min(maxW / w, maxH / h);
  return { width: w * scale, height: h * scale, scale };
}
export function screenToImage(point: Point, stage: Point, scale: number): Point {
  return { x: (point.x - stage.x) / scale, y: (point.y - stage.y) / scale };
}
export function imageToScreen(point: Point, stage: Point, scale: number): Point {
  return { x: point.x * scale + stage.x, y: point.y * scale + stage.y };
}
export function clamp(point: Point, width: number, height: number): Point {
  return { x: Math.max(0, Math.min(width, point.x)), y: Math.max(0, Math.min(height, point.y)) };
}
