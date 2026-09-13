# Predictive Modeling of Localized Periodontal Disease

Research prototype for oral assessment and treatment planning from anonymized periapical dental radiographs.

> Research use only. This project is not a medical device, does not provide a diagnosis, and does not replace review by a qualified dental professional.

## Structure

- `frontend/` React + TypeScript + Vite dashboard with Tailwind CSS.
- `backend/` FastAPI service with OpenCV-based image handling.
- `dataset/` staged data directories for raw, cleaned, processed, and JSON annotations.
- `models/` local model artifacts; no trained model is included in this prototype.
- `docs/` research notes and design documentation.

The existing `DenPAR Radiographs Dataset/` directory is kept as a local research reference. Do not publish patient-identifiable information or real radiographs to a public repository.

## Quick start

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

The API exposes `GET /health`; the frontend is available at the Vite URL shown in the terminal.

## Configuration

Copy `frontend/.env.example` to `frontend/.env` and `backend/.env.example` to `backend/.env` when local overrides are needed. No patient identifiers belong in environment files, JSON fixtures, logs, or filenames.

## Scope

This initial structure provides the dashboard and a basic image-processing API. Predictive modeling and PyTorch training/inference are intentionally deferred.

## Model planning

Architecture recommendations grounded in the current prepared dataset and annotation availability are documented in [docs/deep_learning_architecture_recommendation.md](docs/deep_learning_architecture_recommendation.md). The current dataset contains 650 processed images but no expert-validated annotations, so the report recommends data collection and label-quality gates before supervised training.
