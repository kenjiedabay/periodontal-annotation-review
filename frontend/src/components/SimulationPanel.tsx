import { useMemo, useState } from 'react';
import type { Tooth } from '../types';
import { bucketFor, BUCKET_STYLES, computeSeverity } from '../lib/severity';

interface SimulationPanelProps {
  teeth: Tooth[];
  imageName: string | null;
}

/**
 * A conversation aid for expert review. It is intentionally not a predictive
 * model: every assumption is visible and adjustable by the dentist.
 */
export default function SimulationPanel({ teeth, imageName }: SimulationPanelProps) {
  const [years, setYears] = useState(3);
  const [untreatedChange, setUntreatedChange] = useState(4);
  const [controlledChange, setControlledChange] = useState(0.5);
  const [selectedUid, setSelectedUid] = useState<string | null>(null);

  const toothSummaries = useMemo(() => teeth.map((tooth) => ({
    tooth,
    baseline: computeSeverity(tooth.cej, tooth.bone, tooth.apex),
  })), [teeth]);

  const selected = useMemo(() => {
    const chosen = toothSummaries.find(({ tooth }) => tooth.uid === selectedUid);
    if (chosen) return chosen;
    return toothSummaries.reduce((highest, current) => {
      if (!highest || (current.baseline ?? -1) > (highest.baseline ?? -1)) return current;
      return highest;
    }, toothSummaries[0]);
  }, [selectedUid, toothSummaries]);

  const projection = useMemo(() => {
    const baseline = selected?.baseline;
    if (baseline === null || baseline === undefined) return null;
    return Array.from({ length: years + 1 }, (_, year) => ({
      year,
      untreated: Math.min(100, baseline + untreatedChange * year),
      controlled: Math.max(0, Math.min(100, baseline + controlledChange * year)),
    }));
  }, [controlledChange, selected, untreatedChange, years]);

  const format = (value: number | null | undefined) =>
    value === null || value === undefined ? '—' : `${value.toFixed(1)}%`;
  const bucket = bucketFor(selected?.baseline ?? null);
  const bucketStyle = BUCKET_STYLES[bucket];

  function selectTooth(uid: string) {
    setSelectedUid(uid || null);
  }

  function resetScenario() {
    setYears(3);
    setUntreatedChange(4);
    setControlledChange(0.5);
  }

  const lastProjection = projection?.[projection.length - 1];

  return (
    <section className="mt-5 overflow-hidden rounded-xl border border-slate-200 bg-slate-950 shadow-sm">
      <div className="border-b border-white/10 px-4 py-4 text-white">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-teal-300">Clinical discussion tool</p>
            <h2 className="mt-1 text-base font-bold">Periodontal outlook simulation</h2>
            <p className="mt-1 max-w-lg text-xs leading-relaxed text-slate-300">
              Explore how the reviewed landmark ratio could change under two transparent scenarios.
            </p>
          </div>
          <span className="shrink-0 rounded-full border border-amber-300/30 bg-amber-300/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-amber-200">
            discussion only
          </span>
        </div>
      </div>

      {!teeth.length ? (
        <div className="bg-slate-50 p-4 text-xs text-slate-600">
          Load or add landmark points first. The simulation will use a reviewed tooth as its baseline.
        </div>
      ) : (
        <div className="bg-slate-50 p-4">
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
            <div className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">Selected baseline</p>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="font-mono text-2xl font-bold tracking-tight text-slate-950">{format(selected?.baseline)}</span>
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold text-white ${bucketStyle.bg}`}>{bucketStyle.label}</span>
                  </div>
                </div>
                <label className="text-right text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  Review tooth
                  <select
                    className="mt-1 block min-w-[150px] rounded-md border border-slate-200 bg-white px-2 py-1.5 text-left text-xs font-semibold normal-case tracking-normal text-slate-800 outline-none focus:border-teal-600"
                    value={selected?.tooth.uid ?? ''}
                    onChange={(event) => selectTooth(event.target.value)}
                  >
                    {toothSummaries.map(({ tooth, baseline }) => <option key={tooth.uid} value={tooth.uid}>{tooth.id} ({format(baseline)})</option>)}
                  </select>
                </label>
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                Based on {selected?.tooth.id ?? 'the selected tooth'}{imageName ? ` in ${imageName}` : ''}. Confirm the landmark placement and clinical relevance before discussing this view.
              </p>
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">
                <span>Time horizon</span><span className="font-mono text-sm tracking-normal text-slate-900">{years} years</span>
              </div>
              <input className="mt-3 block w-full accent-teal-700" type="range" min="1" max="5" value={years} onChange={(event) => setYears(Number(event.target.value))} />
              <div className="mt-1 flex justify-between text-[10px] text-slate-400"><span>1 yr</span><span>5 yrs</span></div>
            </div>
          </div>

          <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-xs font-bold text-slate-900">Illustrative trajectory</h3>
                <p className="mt-0.5 text-[11px] text-slate-500">Higher percentage indicates more loss along the measured root axis.</p>
              </div>
              <button type="button" onClick={resetScenario} className="text-[11px] font-semibold text-teal-700 hover:text-teal-900">Reset assumptions</button>
            </div>
            <div className="mt-3 space-y-2">
              {projection?.map((point) => <div key={point.year} className="grid grid-cols-[34px_minmax(0,1fr)_minmax(0,1fr)] items-center gap-2 text-[10px]">
                <span className="font-mono text-slate-500">{point.year === 0 ? 'Now' : `Y${point.year}`}</span>
                <div className="relative h-6 overflow-hidden rounded bg-rose-50"><div className="h-full rounded bg-rose-400/80 transition-all" style={{ width: `${point.untreated}%` }} /><span className="absolute inset-y-0 right-1 flex items-center font-mono font-bold text-slate-700">{format(point.untreated)}</span></div>
                <div className="relative h-6 overflow-hidden rounded bg-teal-50"><div className="h-full rounded bg-teal-500/80 transition-all" style={{ width: `${point.controlled}%` }} /><span className="absolute inset-y-0 right-1 flex items-center font-mono font-bold text-slate-700">{format(point.controlled)}</span></div>
              </div>)}
              <div className="grid grid-cols-[34px_minmax(0,1fr)_minmax(0,1fr)] gap-2 pt-1 text-[10px] font-semibold text-slate-500"><span /> <span className="text-rose-700">Possible untreated</span><span className="text-teal-700">Possible controlled</span></div>
            </div>
          </div>

          <div className="mt-3 grid grid-cols-2 gap-3">
            <ScenarioControl label="Untreated assumption" value={untreatedChange} onChange={setUntreatedChange} tone="rose" />
            <ScenarioControl label="Controlled assumption" value={controlledChange} onChange={setControlledChange} tone="teal" />
          </div>

          <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-[11px] text-amber-950">
            <span>At year {years}: <b>{format(lastProjection?.untreated)}</b> untreated vs <b>{format(lastProjection?.controlled)}</b> controlled.</span>
            <span className="hidden text-right text-amber-800 sm:block">User-adjusted rates, not a prognosis.</span>
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">This is an educational what-if visual. Rates are not learned from longitudinal patient data and do not indicate regeneration, prognosis, or a treatment recommendation.</p>
        </div>
      )}
    </section>
  );
}

function ScenarioControl({ label, value, onChange, tone }: { label: string; value: number; onChange: (value: number) => void; tone: 'rose' | 'teal' }) {
  const accent = tone === 'rose' ? 'accent-rose-600' : 'accent-teal-700';
  return <label className="rounded-lg border border-slate-200 bg-white p-3 text-[11px] font-semibold text-slate-700">
    <span className="flex items-center justify-between gap-2"><span>{label}</span><span className="font-mono text-slate-950">{value.toFixed(1)}% / yr</span></span>
    <input className={`mt-2 block w-full ${accent}`} type="range" min="0" max="10" step="0.5" value={value} onChange={(event) => onChange(Number(event.target.value))} />
  </label>;
}
