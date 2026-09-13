import type { PredictionPayload, PredictionToothInput, Tooth } from '../types';

export interface ParsedPredictions {
  teeth: Tooth[];
  warning?: string;
}

function isValidToothInput(item: unknown): item is PredictionToothInput {
  if (!item || typeof item !== 'object') return false;
  const t = item as Record<string, unknown>;
  return Array.isArray(t.cej) && Array.isArray(t.bone_intersection) && Array.isArray(t.apex);
}

/**
 * Parses the JSON pasted into the "Review" tab into working Tooth records.
 * Throws a plain Error with a message meant to be shown directly in the UI.
 */
export function parsePredictions(raw: string, currentImageName: string | null): ParsedPredictions {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("That's not valid JSON — check for a missing bracket or comma.");
  }

  const payload = parsed as Partial<PredictionPayload>;
  const list: unknown[] | null = Array.isArray(payload.teeth)
    ? payload.teeth
    : Array.isArray(parsed)
      ? (parsed as unknown[])
      : null;

  if (!list) {
    throw new Error('Expected a "teeth" array — see the placeholder text for the shape.');
  }

  const teeth: Tooth[] = list.map((item, idx) => {
    if (!isValidToothInput(item)) {
      throw new Error('Each tooth needs cej, bone_intersection, and apex as [x, y] pairs.');
    }
    return {
      uid: crypto.randomUUID(),
      id: item.tooth_id || `Tooth ${idx + 1}`,
      cej: { x: item.cej[0], y: item.cej[1] },
      bone: { x: item.bone_intersection[0], y: item.bone_intersection[1] },
      apex: { x: item.apex[0], y: item.apex[1] },
      fromPrediction: true,
      wasEdited: false,
      notes: '',
    };
  });

  const imageName = (parsed as Partial<PredictionPayload>).image_name;
  const warning =
    imageName && imageName !== currentImageName
      ? `Note: these predictions were labeled for "${imageName}", but "${currentImageName ?? 'no image'}" is open now.`
      : undefined;

  return { teeth, warning };
}
