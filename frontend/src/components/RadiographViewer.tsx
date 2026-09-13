import { useCallback, useEffect, useRef, useState } from 'react';
import type { LandmarkKey, Point, Tooth } from '../types';
import { BUCKET_STYLES, MARKER_STYLES, bucketFor, computeSeverity } from '../lib/severity';

interface OverlayRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface RadiographViewerProps {
  imageSrc: string | null;
  naturalSize: { w: number; h: number } | null;
  teeth: Tooth[];
  onImageLoaded: (naturalWidth: number, naturalHeight: number) => void;
  onPointChange: (uid: string, key: LandmarkKey, point: Point) => void;
}

export default function RadiographViewer({
  imageSrc,
  naturalSize,
  teeth,
  onImageLoaded,
  onPointChange,
}: RadiographViewerProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [overlay, setOverlay] = useState<OverlayRect>({ left: 0, top: 0, width: 0, height: 0 });

  const syncOverlay = useCallback(() => {
    if (!wrapRef.current || !imgRef.current) return;
    const wrapRect = wrapRef.current.getBoundingClientRect();
    const imgRect = imgRef.current.getBoundingClientRect();
    setOverlay({
      left: imgRect.left - wrapRect.left,
      top: imgRect.top - wrapRect.top,
      width: imgRect.width,
      height: imgRect.height,
    });
  }, []);

  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver(syncOverlay);
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [syncOverlay]);

  useEffect(() => {
    syncOverlay();
  }, [imageSrc, naturalSize, syncOverlay]);

  function handleImgLoad() {
    if (!imgRef.current) return;
    onImageLoaded(imgRef.current.naturalWidth, imgRef.current.naturalHeight);
    requestAnimationFrame(syncOverlay);
  }

  function clientToSvgPoint(clientX: number, clientY: number): Point {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const p = pt.matrixTransform(ctm.inverse());
    return { x: p.x, y: p.y };
  }

  function startDrag(e: React.PointerEvent<SVGCircleElement>, tooth: Tooth, key: LandmarkKey) {
    e.preventDefault();
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    target.classList.add('cursor-grabbing');

    const onMove = (ev: PointerEvent) => {
      if (!naturalSize) return;
      const p = clientToSvgPoint(ev.clientX, ev.clientY);
      onPointChange(tooth.uid, key, {
        x: Math.max(0, Math.min(naturalSize.w, p.x)),
        y: Math.max(0, Math.min(naturalSize.h, p.y)),
      });
    };
    const onUp = () => {
      target.releasePointerCapture(e.pointerId);
      target.classList.remove('cursor-grabbing');
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
    };
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
  }

  const markerRadius = naturalSize ? Math.max(8, naturalSize.w * 0.014) : 8;
  const fontSize = naturalSize ? Math.max(11, naturalSize.w * 0.018) : 11;

  return (
    <div
      ref={wrapRef}
      className="relative w-full flex-1 flex items-center justify-center overflow-hidden bg-slate-950"
    >
      {!imageSrc && (
        <div className="absolute inset-0 flex items-center justify-center px-10 text-center text-sm text-slate-400">
          Load a radiograph to begin reviewing predicted landmarks.
        </div>
      )}

      {imageSrc && (
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Radiograph"
          onLoad={handleImgLoad}
          className="block h-auto max-h-full w-auto max-w-full"
        />
      )}

      {imageSrc && naturalSize && (
        <svg
          ref={svgRef}
          viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`}
          className="pointer-events-none absolute"
          style={{ left: overlay.left, top: overlay.top, width: overlay.width, height: overlay.height }}
        >
          {teeth.map((t) => {
            const pct = computeSeverity(t.cej, t.bone, t.apex);
            const bucket = bucketFor(pct);
            const styles = BUCKET_STYLES[bucket];
            const label = pct === null || Number.isNaN(pct) ? '—' : `${Math.round(pct)}%`;
            const labelX = t.bone.x + markerRadius * 1.6;
            const labelY = t.bone.y + markerRadius * 0.35;
            const labelW = label.length * fontSize * 0.62 + 10;

            return (
              <g key={t.uid}>
                <line
                  x1={t.cej.x}
                  y1={t.cej.y}
                  x2={t.apex.x}
                  y2={t.apex.y}
                  stroke="#8B93A0"
                  strokeWidth={Math.max(1, naturalSize.w * 0.0015)}
                  opacity={0.55}
                />
                <line
                  x1={t.cej.x}
                  y1={t.cej.y}
                  x2={t.bone.x}
                  y2={t.bone.y}
                  className={styles.stroke}
                  strokeWidth={Math.max(2, naturalSize.w * 0.006)}
                  strokeLinecap="round"
                  opacity={0.9}
                />
                <rect
                  x={labelX - 5}
                  y={labelY - fontSize}
                  width={labelW}
                  height={fontSize * 1.5}
                  rx={4}
                  fill="rgba(11,13,16,0.72)"
                />
                <text x={labelX} y={labelY} fontSize={fontSize} fill="#fff" fontFamily="ui-monospace, monospace">
                  {label}
                </text>
                <circle
                  cx={t.apex.x}
                  cy={t.apex.y}
                  r={markerRadius}
                  className={`${MARKER_STYLES.apex.fill} pointer-events-auto cursor-grab stroke-slate-950 stroke-2`}
                  onPointerDown={(e) => startDrag(e, t, 'apex')}
                />
                <circle
                  cx={t.bone.x}
                  cy={t.bone.y}
                  r={markerRadius}
                  className={`${MARKER_STYLES.bone.fill} pointer-events-auto cursor-grab stroke-slate-950 stroke-2`}
                  onPointerDown={(e) => startDrag(e, t, 'bone')}
                />
                <circle
                  cx={t.cej.x}
                  cy={t.cej.y}
                  r={markerRadius}
                  className={`${MARKER_STYLES.cej.fill} pointer-events-auto cursor-grab stroke-slate-950 stroke-2`}
                  onPointerDown={(e) => startDrag(e, t, 'cej')}
                />
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
