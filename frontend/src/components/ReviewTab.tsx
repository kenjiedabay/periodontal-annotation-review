import { useState } from 'react';
import type { Tooth } from '../types';
import ToothCard from './ToothCard';
import SimulationPanel from './SimulationPanel';

export interface PredictionNotice {
  text: string;
  level: 'error' | 'warn';
}

interface ReviewTabProps {
  teeth: Tooth[];
  loadDisabled: boolean;
  predictionNotice: PredictionNotice | null;
  onLoadPredictions: (raw: string) => void;
  onUpdateTooth: (uid: string, patch: Partial<Pick<Tooth, 'id' | 'notes'>>) => void;
  onRemoveTooth: (uid: string) => void;
  onSaveTooth: (uid: string) => Promise<void>;
  imageName: string | null;
}

const PLACEHOLDER =
  '{"teeth":[{"tooth_id":"36-distal","cej":[412,180],"bone_intersection":[420,240],"apex":[430,520]}]}';

export default function ReviewTab({
  teeth,
  loadDisabled,
  predictionNotice,
  onLoadPredictions,
  onUpdateTooth,
  onRemoveTooth,
  onSaveTooth,
  imageName,
}: ReviewTabProps) {
  const [raw, setRaw] = useState('');

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <p className="mb-2.5 text-[12.5px] leading-relaxed text-slate-500">
        Paste the model&apos;s predicted points for this image as JSON, then correct any landmark by
        dragging it. Or skip this and use &quot;+ Add tooth&quot; to annotate from scratch.
      </p>

      <textarea
        className="min-h-[76px] w-full resize-y rounded-md border border-slate-200 p-2 font-mono text-[11.5px] text-slate-900"
        placeholder={PLACEHOLDER}
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
      />

      {predictionNotice && (
        <div
          className={`mt-1.5 text-xs ${
            predictionNotice.level === 'error' ? 'text-rose-600' : 'text-amber-600'
          }`}
        >
          {predictionNotice.text}
        </div>
      )}

      <button
        type="button"
        disabled={loadDisabled}
        className="mt-2 w-full rounded-md bg-teal-700 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
        onClick={() => onLoadPredictions(raw)}
      >
        Load into viewer
      </button>

      <div className="mt-4 flex flex-col gap-2.5">
        {teeth.map((t) => (
          <ToothCard
            key={t.uid}
            tooth={t}
            onUpdate={(patch) => onUpdateTooth(t.uid, patch)}
            onRemove={() => onRemoveTooth(t.uid)}
            onSave={() => onSaveTooth(t.uid)}
          />
        ))}
      </div>
      <SimulationPanel teeth={teeth} imageName={imageName} />
    </div>
  );
}
