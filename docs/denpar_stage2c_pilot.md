# DenPAR Stage 2C dental expert pilot

This pilot reviews anatomical associations only. It does not request or produce bone-loss percentages, severity, disease pattern, treatment, or diagnosis.

## Start on Windows

Open two PowerShell windows at the project root.

Backend:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/?pilot=stage2c`.

## Expert instructions

1. Enter your reviewer ID. Keep the same ID throughout your independent review.
2. Select a stable tooth instance. Review its mask, full radiograph context, and crop with surrounding bone.
3. Select a colored CEJ or apex point, or a green bone polyline.
4. Inspect its raw confidence, ambiguity state, candidate teeth, and uncertainty reason.
5. Confirm, reject, reassign, delete, or mark the proposal uncertain. Enter a reason before rejection, uncertainty, or second review.
6. Use **Add missing CEJ** or **Add visible apex** for a missing visible point. Coordinates are original-image pixels.
7. Use neutral apex records when roots overlap. Record `uncertain_root_assignment` or the reason in the note.
8. Keep geometric `image_left` and `image_right`. Do not enter mesial or distal unless tooth identity, arch, and display orientation are verified outside this pilot.
9. For a bone line, record whether it belongs to one tooth surface, an interproximal pair, or remains uncertain.
10. Move to the next queued image. Every saved action is a new immutable version; raw Stage 2B proposals remain unchanged.

The current configuration has one expert. No overlap subset or inter-rater statistic is created until a second expert is available.
