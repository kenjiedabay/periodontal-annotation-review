export interface Point {
  x: number;
  y: number;
}

export interface ToothInstance {
  instance_id: number;
  confidence: number;
  /** Original-image pixels, [left, top, right, bottom]. */
  bbox: [number, number, number, number];
  /** PNG data URL: opaque foreground, transparent background. */
  mask_url: string;
}

export interface ToothSegmentationResponse {
  image_id: string;
  task: 'tooth_instance_segmentation';
  model_status: 'available' | 'unavailable';
  model_version: string;
  confidence_threshold: number;
  model_error?: string;
  width: number;
  height: number;
  instances: ToothInstance[];
  ground_truth: { split: string; mask_urls: string[]; source: string } | null;
}

export type LandmarkKey = 'cej' | 'bone' | 'apex';

export type ToothStatus = 'confirmed' | 'corrected' | 'annotated_from_scratch';

export type SeverityBucket = 'none' | 'mild' | 'moderate' | 'severe' | 'unknown';

/** One tooth side currently being reviewed in the viewer (working, in-memory state). */
export interface Tooth {
  /** Stable identity for React keys / state updates. Independent of the editable label below. */
  uid: string;
  /** Editable human label, e.g. "36-distal". */
  id: string;
  cej: Point;
  bone: Point;
  apex: Point;
  /** True if these coordinates came from a pasted model-prediction JSON rather than "+ Add tooth". */
  fromPrediction: boolean;
  /** True once the dentist has dragged any point away from its loaded position. */
  wasEdited: boolean;
  notes: string;
}

/** One row of the append-only, persisted validated dataset. */
export interface ValidatedRecord {
  id: string;
  imageName: string;
  toothId: string;
  cej: Point;
  bone: Point;
  apex: Point;
  severityPct: number | null;
  severityBucket: SeverityBucket;
  status: ToothStatus;
  notes: string;
  timestamp: number;
}

/** Shape expected when pasting a model's prediction output into the "Review" tab. */
export interface PredictionToothInput {
  tooth_id?: string;
  cej: [number, number];
  bone_intersection: [number, number];
  apex: [number, number];
}

export interface PredictionPayload {
  image_name?: string;
  teeth: PredictionToothInput[];
}

export type DiseaseStatus = 'Present' | 'Absent' | 'Uncertain';
export type AnnotationSeverity = 'Mild' | 'Moderate' | 'Severe' | 'Cannot determine';
export type AnnotationMethod = 'Bounding box' | 'Polygon / freehand region';
export type ValidationStatus = 'Pending' | 'Validated' | 'Needs Revision' | 'Uncertain';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SpatialAnnotation {
  method: AnnotationMethod;
  boundingBox: BoundingBox | null;
  polygon: Point[];
}

export interface AnnotationDraft {
  status: DiseaseStatus;
  teeth: string[];
  severity: AnnotationSeverity;
  findings: string[];
  notes: string;
  region: AnnotationMethod;
  spatial: SpatialAnnotation;
  validated: boolean;
  savedAt?: string;
  validationStatus: ValidationStatus;
  validationTimestamp?: string;
}

export interface AnnotationPayload {
  image_id: string;
  disease_status: Lowercase<DiseaseStatus>;
  affected_teeth: string[];
  severity: 'mild' | 'moderate' | 'severe' | 'cannot_determine';
  findings: string[];
  regions: Array<{ type: 'bounding_box' | 'polygon' | 'freehand'; coordinates: number[] | number[][] }>;
  expert_comment: string;
  expert_validated: boolean;
  validation_status: 'pending' | 'validated' | 'needs_revision' | 'uncertain';
  validation_timestamp?: string;
}

export interface StructuralAnalysisResponse {
  image_id: string;
  ground_truth: { expert_validated: boolean; findings: string[]; regions: AnnotationPayload['regions'] } | null;
  preliminary_result: string | null;
  model_prediction: string | null;
  analysis_status: 'ground_truth_only' | 'model_unavailable';
}

export interface ModelAnalysisResponse {
  image_id: string;
  model_status: 'available' | 'unavailable';
  disease_prediction: { label: string | null; confidence: number | null };
  region_prediction: { status: string | null; confidence: number | null };
  severity_prediction: { label: string | null; confidence: number | null };
  structural_findings: string[];
  progression: { status: 'unsupported' | 'unavailable'; reason: string };
  research_only: boolean;
}

export interface StructuralAuditRecord {
  available: boolean;
  reason?: string;
  image: { image_id: string; filename: string; width: number; height: number };
  tooth_masks: Array<{ filename: string; url: string; width: number; height: number; nonzero_pixels: number }>;
  radiograph_mask: { url: string; width: number; height: number; nonzero_pixels: number } | null;
  coco_annotations: Array<{ id: number; bbox: number[]; segmentation: number[][]; image_id: number; category_id: number }>;
  keypoints: { Image_id: string; bboxes: number[][]; CEJ_Points: number[][]; Apex_Points: number[][] } | null;
  bone_lines: { Image_id: string; Num_of_Bone_Lines: number; Bone_Lines: number[][][] } | null;
  characteristics: Record<string, string> | null;
  issues: Array<{ image_id: string; category: string; message: string; source: string }>;
  correspondence: { image_id: string; status: string; relationships: Array<{ name: string; status: 'confirmed' | 'uncertain' | 'missing' | 'not_applicable'; evidence: string; unresolved_reason: string }>; missing_landmarks: string[]; suspicious_mappings: string[]; review_items: Record<string, Array<{ id: string; status: 'confirmed' | 'uncertain' | 'missing' | 'not_applicable'; bbox_xyxy?: number[] | null; point?: number[]; bbox_candidates?: string[] }> > };
  source_preserved: boolean;
}
