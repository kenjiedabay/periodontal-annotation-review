# Periodontal annotation review

A TypeScript + React + Tailwind rebuild of the dentist-facing review tool: load a
periapical radiograph, overlay a model's predicted CEJ / bone-level / apex
landmarks per tooth, let the dentist drag any point to correct it, and save the
corrected (or confirmed) result into a validated dataset you can export for
training the main model.

## Setup

```bash
npm install
npm run dev
```

Then open the printed local URL. `npm run build` type-checks with `tsc -b` and
produces a production bundle in `dist/`.

## Using it

1. **Load radiograph** — pick an image file from disk. Nothing leaves the
   browser; the image is only read locally via `FileReader`.
2. **Paste model predictions** (Review tab) as JSON in this shape, then click
   **Load into viewer**:
   ```json
   {
     "image_name": "2.jpg",
     "teeth": [
       { "tooth_id": "36-distal", "cej": [412, 180], "bone_intersection": [420, 240], "apex": [430, 520] }
     ]
   }
   ```
   `image_name` is optional; if it doesn't match the currently loaded file, a
   warning is shown (not blocked) so a mismatched prediction/image pair
   doesn't slip through silently.
3. Or skip pasting anything and use **+ Add tooth** to place three points from
   scratch and drag them into position — useful for validating the dataset's
   own ground truth, not just a model's predictions.
4. Drag any of the three points (green = CEJ, blue = bone level, red = apex).
   Severity — computed as `distance(CEJ, bone) / distance(CEJ, apex) × 100`,
   the standard % of root length — updates live in both the on-image label and
   the side panel.
5. **Save to validated set** persists that tooth's record. **Validated log**
   lists everything saved so far and exports it as JSON or CSV.

## Where the data actually lives right now

`src/lib/storage.ts` uses browser `localStorage`. That's per-browser, not
shared between machines, and not durable against clearing site data — fine
for a single reviewing session, not a real multi-user deployment.

When you're ready to feed this into your training pipeline for real, replace
`loadRecords` / `persistRecords` in that one file with calls to a real
backend (e.g. `POST /api/records`, `GET /api/records`). Nothing else in the
app needs to change: every component only depends on the `records` array and
the `saveTooth` callback in `App.tsx`, not on `localStorage` directly.

## Severity bucket cutoffs

`src/lib/severity.ts` uses one published scheme (crestal-only = mild,
10–33% = moderate, 33%+ = severe) as a placeholder. Other sources use 15% as
the mild/moderate line instead — pick and document whichever scheme your
project is validating against.

## Project structure

```
src/
├── App.tsx                     top-level state + layout
├── types.ts                    Tooth / ValidatedRecord / prediction shapes
├── lib/
│   ├── severity.ts              severity formula + bucket styling
│   ├── predictions.ts           parses/validates pasted prediction JSON
│   ├── storage.ts               persistence (see above)
│   └── download.ts              JSON/CSV export helper
└── components/
    ├── RadiographViewer.tsx      image + draggable SVG landmark overlay
    ├── ReviewTab.tsx              prediction input + tooth card list
    ├── ToothCard.tsx              one tooth's severity, notes, save button
    └── ValidatedLogTab.tsx        accumulated log + exports
```
