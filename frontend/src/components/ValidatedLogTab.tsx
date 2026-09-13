import type { ValidatedRecord } from '../types';
import { BUCKET_STYLES } from '../lib/severity';
import { downloadBlob } from '../lib/download';

interface ValidatedLogTabProps {
  records: ValidatedRecord[];
}

function csvField(v: unknown): string {
  return `"${String(v ?? '').replace(/"/g, '""')}"`;
}

export default function ValidatedLogTab({ records }: ValidatedLogTabProps) {
  function exportJson() {
    downloadBlob(
      new Blob([JSON.stringify(records, null, 2)], { type: 'application/json' }),
      'validated_annotations.json',
    );
  }

  function exportCsv() {
    const header = [
      'image_name', 'tooth_id', 'cej_x', 'cej_y', 'bone_x', 'bone_y', 'apex_x', 'apex_y',
      'severity_pct', 'severity_bucket', 'status', 'notes', 'timestamp_iso',
    ];
    const rows = records.map((r) =>
      [
        csvField(r.imageName),
        csvField(r.toothId),
        r.cej.x, r.cej.y, r.bone.x, r.bone.y, r.apex.x, r.apex.y,
        r.severityPct != null ? r.severityPct.toFixed(2) : '',
        csvField(r.severityBucket),
        csvField(r.status),
        csvField(r.notes),
        csvField(new Date(r.timestamp).toISOString()),
      ].join(','),
    );
    downloadBlob(
      new Blob([[header.join(','), ...rows].join('\n')], { type: 'text/csv' }),
      'validated_annotations.csv',
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-y-auto p-4">
      <div className="flex flex-1 flex-col gap-2">
        {records.length === 0 && (
          <div className="py-5 text-center text-sm text-slate-500">No validated records yet.</div>
        )}
        {records.map((r) => {
          const styles = BUCKET_STYLES[r.severityBucket];
          return (
            <div key={r.id} className="rounded-md border border-slate-200 bg-white p-2.5 text-[12.5px]">
              <div className="flex items-start justify-between gap-2">
                <span className="font-semibold text-slate-900">
                  {r.imageName} — {r.toothId}
                </span>
                <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold text-white ${styles.bg}`}>
                  {styles.label}
                </span>
              </div>
              <div className="mt-0.5 text-[11.5px] text-slate-500">
                {r.severityPct != null ? `${r.severityPct.toFixed(1)}% · ` : ''}
                {r.status.replace(/_/g, ' ')} · {new Date(r.timestamp).toLocaleString()}
              </div>
              {r.notes && <div className="mt-0.5 text-[11.5px] text-slate-500">{r.notes}</div>}
            </div>
          );
        })}
      </div>

      <div className="mt-3.5 flex gap-2">
        <button
          type="button"
          className="flex-1 rounded-md border border-slate-200 py-2 text-sm font-semibold text-slate-900 hover:border-slate-400"
          onClick={exportJson}
        >
          Export JSON
        </button>
        <button
          type="button"
          className="flex-1 rounded-md border border-slate-200 py-2 text-sm font-semibold text-slate-900 hover:border-slate-400"
          onClick={exportCsv}
        >
          Export CSV
        </button>
      </div>
    </div>
  );
}
