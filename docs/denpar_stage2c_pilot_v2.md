# DenPAR Stage 2C pilot v2

Pilot v2 is a balanced anatomical-association review queue. Pilot v1 and all prior artifacts remain preserved. This workflow does not collect severity, disease pattern, treatment, diagnosis, or bone-loss percentage.

## Start on Windows

```powershell
cd C:\Users\ADMIN\Downloads\periodontal-annotation-review\backend
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

In a second PowerShell window:

```powershell
cd C:\Users\ADMIN\Downloads\periodontal-annotation-review\frontend
npm.cmd run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/?pilot=stage2c-v2`.

## Point editing

1. Enter the reviewer ID and select a stable tooth instance.
2. Drag a CEJ or apex marker in the full image or tooth crop. The outlined marker is the immutable raw position; cyan is the unsaved position.
3. Choose **Save movement** to write a new expert version, or **Cancel movement** to discard it.
4. For exact placement, choose **Precise coordinates** after selecting a point.
5. Choose **Add CEJ** or **Add apex**, then click the full image or crop.
6. Reassign the selected proposal by choosing another stable tooth and selecting **Reassign**.
7. Deletion always requests confirmation. The raw Stage 2B proposal remains stored.
8. Enter a reason before rejection, uncertainty, or second review.

Coordinates are saved in original-image pixels with top-left origin. Anatomical mesial/distal and named root labels remain unset unless identity and orientation are independently verified.
