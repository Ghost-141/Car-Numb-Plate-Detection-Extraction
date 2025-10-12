from __future__ import annotations

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.services.detect import process_video


app = FastAPI(title="Car Number Plate Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/processed")
for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=1)
jobs: Dict[str, Dict[str, object]] = {}
jobs_lock = threading.Lock()


def update_progress(job_id: str, progress: int) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["progress"] = progress


def run_processing_job(job_id: str, upload_path: Path) -> None:
    try:
        with jobs_lock:
            jobs[job_id].update({"status": "processing", "progress": 0})

        outputs = process_video(upload_path, OUTPUT_DIR, lambda pct: update_progress(job_id, pct))

        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "completed",
                    "progress": 100,
                    "output_video": str(outputs["video"]),
                    "output_csv": str(outputs["csv"]),
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        with jobs_lock:
            jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/process")
async def process_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    job_id = uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0}

    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(executor.submit, run_processing_job, job_id, upload_path)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> Dict[str, object]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/jobs/{job_id}/download")
async def download_video(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if job is None or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Processed video not available.")

    return FileResponse(path=job["output_video"], filename=Path(job["output_video"]).name)


@app.get("/jobs/{job_id}/csv")
async def download_csv(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if job is None or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="CSV results not available.")

    return FileResponse(path=job["output_csv"], filename=Path(job["output_csv"]).name)
