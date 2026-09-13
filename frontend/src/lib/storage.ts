import type { ValidatedRecord } from '../types';

const STORAGE_KEY = 'periodontal_validated_records_v1';

/**
 * This is a standalone app, not a Claude.ai artifact, so it doesn't have
 * access to the artifact-only `window.storage` API used by the earlier
 * chat prototype. Browser localStorage is the equivalent for a
 * no-backend-yet build: it persists across reloads on this machine, but
 * it is per-browser and not shared between the dentist's computer and
 * yours, and it does not survive clearing site data.
 *
 * When you're ready to feed the "main model training" pipeline, swap the
 * two functions below for real API calls (e.g. POST /api/records,
 * GET /api/records) against whatever backend/database ingests the
 * validated set. Nothing else in the app needs to change — every
 * component only depends on the `records` array and the `saveRecord`
 * callback in App.tsx, not on localStorage directly.
 */

export function loadRecords(): ValidatedRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ValidatedRecord[]) : [];
  } catch (err) {
    console.error('Could not load validated records', err);
    return [];
  }
}

export function persistRecords(records: ValidatedRecord[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch (err) {
    console.error('Could not persist validated records', err);
    throw new Error('Browser storage is unavailable.');
  }
}
