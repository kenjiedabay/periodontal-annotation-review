import type { Point, SeverityBucket } from '../types';

export function euclid(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * A radiographic landmark ratio used by this prototype. Its interpretation,
 * thresholds, and any clinical use must be validated by a dental professional.
 */
export function computeSeverity(cej: Point, bone: Point, apex: Point): number | null {
  const rootDist = euclid(cej, apex);
  if (!rootDist) return null;
  return (euclid(cej, bone) / rootDist) * 100;
}

/**
 * One published cutoff scheme (crestal-only = mild, 10-33% = moderate,
 * 33%+ = severe). Other sources use 15% as the mild/moderate line instead —
 * treat this as a starting point, not a universal constant, and document
 * whichever scheme you settle on for your project.
 */
export function bucketFor(pct: number | null): SeverityBucket {
  if (pct === null || Number.isNaN(pct)) return 'unknown';
  if (pct <= 0) return 'none';
  if (pct < 15) return 'mild';
  if (pct < 33) return 'moderate';
  return 'severe';
}

interface BucketStyle {
  /** Literal Tailwind classes (not built from template strings) so the JIT scanner can find them. */
  fill: string;
  stroke: string;
  bg: string;
  label: string;
}

export const BUCKET_STYLES: Record<SeverityBucket, BucketStyle> = {
  none: { fill: 'fill-slate-400', stroke: 'stroke-slate-400', bg: 'bg-slate-400', label: 'None' },
  mild: { fill: 'fill-emerald-500', stroke: 'stroke-emerald-500', bg: 'bg-emerald-500', label: 'Mild' },
  moderate: { fill: 'fill-amber-500', stroke: 'stroke-amber-500', bg: 'bg-amber-500', label: 'Moderate' },
  severe: { fill: 'fill-rose-600', stroke: 'stroke-rose-600', bg: 'bg-rose-600', label: 'Severe' },
  unknown: { fill: 'fill-slate-400', stroke: 'stroke-slate-400', bg: 'bg-slate-400', label: 'Unknown' },
};

/** Fixed per-landmark-type colors (independent of computed severity). */
export const MARKER_STYLES = {
  cej: { fill: 'fill-emerald-500', dot: 'bg-emerald-500' },
  bone: { fill: 'fill-blue-500', dot: 'bg-blue-500' },
  apex: { fill: 'fill-red-500', dot: 'bg-red-500' },
} as const;
