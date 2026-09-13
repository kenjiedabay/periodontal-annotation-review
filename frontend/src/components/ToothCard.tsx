import { useState } from 'react';
import type { Tooth } from '../types';
import { BUCKET_STYLES, bucketFor, computeSeverity } from '../lib/severity';

interface ToothCardProps {
  tooth: Tooth;
  onUpdate: (patch: Partial<Pick<Tooth, 'id' | 'notes'>>) => void;
  onRemove: () => void;
  onSave: () => Promise<void>;
}

export default function ToothCard({ tooth, onUpdate, onRemove, onSave }: ToothCardProps) {
  const [savedMsg, setSavedMsg] = useState<{ text: string; isError: boolean } | null>(null);

  const pct = computeSeverity(tooth.cej, tooth.bone, tooth.apex);
  const bucket = bucketFor(pct);
  const styles = BUCKET_STYLES[bucket];
  const origin = tooth.fromPrediction
    ? tooth.wasEdited
      ? 'Corrected from model prediction'
      : 'Matches model prediction'
    : 'Annotated from scratch';

  async function handleSave() {
    try {
      await onSave();
      setSavedMsg({ text: 'Saved', isError: false });
    } catch {
      setSavedMsg({ text: "Couldn't save — try again", isError: true });
    }
    setTimeout(() => setSavedMsg(null), 2600);
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center gap-2">
        <input
          className="flex-1 border-b border-transparent bg-transparent py-0.5 text-sm font-semibold text-slate-900 focus:border-teal-700 focus:outline-none"
          value={tooth.id}
          onChange={(e) => onUpdate({ id: e.target.value })}
        />
        <button
          type="button"
          className="px-1 text-xs text-slate-400 hover:text-rose-600"
          onClick={onRemove}
        >
          Remove
        </button>
      </div>

      <div className="mb-2 flex items-baseline gap-2">
        <span className="font-mono text-lg font-semibold text-slate-900">
          {pct === null || Number.isNaN(pct) ? '—' : `${pct.toFixed(1)}%`}
        </span>
        <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold text-white ${styles.bg}`}>
          {styles.label}
        </span>
      </div>

      <textarea
        className="min-h-[44px] w-full resize-y rounded-md border border-slate-200 p-2 text-xs text-slate-900"
        placeholder="Notes (optional) — e.g. apex unclear due to overlap"
        value={tooth.notes}
        onChange={(e) => onUpdate({ notes: e.target.value })}
      />

      <div className="mt-1.5 text-[11.5px] text-slate-500">{origin}</div>

      <div className="mt-2 flex items-center gap-2">
        <button
          type="button"
          className="rounded-md bg-teal-700 px-3 py-1.5 text-sm font-semibold text-white hover:brightness-105"
          onClick={handleSave}
        >
          Save to validated set
        </button>
        {savedMsg && (
          <span className={`text-xs ${savedMsg.isError ? 'text-rose-600' : 'text-emerald-600'}`}>
            {savedMsg.text}
          </span>
        )}
      </div>
    </div>
  );
}
