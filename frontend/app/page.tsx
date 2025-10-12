"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";

type JobState = {
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  error?: string;
  output_video?: string;
  output_csv?: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const features = [
  {
    icon: "AI",
    title: "Deep Learning Backbone",
    description: "Powered by YOLO vehicle detection and OCR-driven plate recognition for enterprise-grade accuracy.",
  },
  {
    icon: "RT",
    title: "Real-Time Tracking",
    description: "Deep Sort tracking keeps targets locked while results stream frame by frame.",
  },
  {
    icon: "CSV",
    title: "Instant Exports",
    description: "Download the annotated video alongside structured CSV logs of every detected plate.",
  },
  {
    icon: "AP",
    title: "API Ready",
    description: "FastAPI endpoints make it easy to integrate automation into your current workflows.",
  },
];

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [uploading, setUploading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    const interval = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`);
        if (!response.ok) {
          return;
        }
        const data: JobState = await response.json();
        setJob(data);
        if (data.status === "completed" || data.status === "failed") {
          window.clearInterval(interval);
        }
      } catch (_) {
        /* swallow polling errors */
      }
    }, 1000);

    return () => window.clearInterval(interval);
  }, [jobId]);

  const handleStartProcessing = async () => {
    if (!file) {
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/process`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        setJob({ status: "failed", progress: 0, error: "Upload failed." });
        return;
      }

      const data: { job_id: string } = await response.json();
      setJobId(data.job_id);
      setJob({ status: "queued", progress: 0 });
    } catch (error) {
      setJob({
        status: "failed",
        progress: 0,
        error: error instanceof Error ? error.message : "Upload error.",
      });
    } finally {
      setUploading(false);
    }
  };

  useEffect(
    () => () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    },
    [previewUrl],
  );

  const isJobRunning = job?.status === "queued" || job?.status === "processing";

  const openFilePicker = () => {
    if (uploading || isJobRunning) {
      return;
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  };

  const handleFileSelected = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (nextFile) {
      const objectUrl = URL.createObjectURL(nextFile);
      setPreviewUrl(objectUrl);
      setPreviewError(false);
    } else {
      setPreviewUrl(null);
      setPreviewError(false);
    }

    setFile(nextFile);
    setJob(null);
    setJobId(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const statusLabel = (() => {
    if (!job) {
      return "Awaiting upload";
    }
    if (job.status === "queued") {
      return "Queued - Preparing analysis";
    }
    if (job.status === "processing") {
      return "Processing frames";
    }
    if (job.status === "completed") {
      return "Your file is ready.";
    }
    return "Failed - Please try again";
  })();

  return (
    <main className="page">
      <section className="hero">
        <div className="hero-body">
          <h1 className="hero-title">Automatic License Plate Detection</h1>
          <p className="hero-sub">
            Streamline enforcement and analytics with a modern pipeline that locks onto vehicles, recognises plates, and
            serves polished outputs in record time.
          </p>
          <div className="hero-stats">
            <div className="hero-stat">
              <strong>99%</strong>
              <span>Detection confidence with YOLO</span>
            </div>
            <div className="hero-stat">
              <strong>60s</strong>
              <span>Per minute of processed footage</span>
            </div>
            <div className="hero-stat">
              <strong>API</strong>
              <span>FastAPI powered automation ready</span>
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <div className="hero-illustration">
            <div className="hero-radar" />
            <div className="hero-radar hero-radar--two" />
            <div className="hero-scan" />
            <div className="hero-plate">
              <span className="hero-plate-code">
                <span>ABC</span>
                <span>2045</span>
              </span>
              <span className="hero-plate-meta">Verified - Confidence 0.87</span>
            </div>
          </div>
        </div>
      </section>

      <section className="content-grid">
        <div className="card">
          <h2>Upload video footage</h2>
          <p>
            Select an MP4 or similar video file. Once uploaded, click the start button to run automatic detection on the
            footage.
          </p>
          <div className="upload-dropzone">
            <input
              ref={fileInputRef}
              className="hidden"
              type="file"
              accept="video/*"
              onChange={handleFileSelected}
            />
            {previewUrl && !previewError ? (
              <>
                <video
                  key={previewUrl}
                  className="upload-preview"
                  controls
                  muted
                  playsInline
                  preload="metadata"
                  onError={() => setPreviewError(true)}
                  onLoadedData={() => setPreviewError(false)}
                >
                  <source src={previewUrl} type={file?.type || "video/mp4"} />
                  Your browser does not support embedded video previews.
                </video>
                <div className="upload-actions">
                  <button
                    type="button"
                    className="button-tertiary"
                    onClick={openFilePicker}
                    disabled={uploading || isJobRunning}
                  >
                    Change video
                  </button>
                  <span className="file-name">{file ? file.name : "No file selected"}</span>
                </div>
              </>
            ) : previewError ? (
              <div className="upload-actions">
                <p className="subtext">Preview unavailable for this format, but the file is ready to process.</p>
                <button
                  type="button"
                  className="button-tertiary"
                  onClick={openFilePicker}
                  disabled={uploading || isJobRunning}
                >
                  Change video
                </button>
                <span className="file-name">{file ? file.name : "No file selected"}</span>
              </div>
            ) : (
              <>
                <svg width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="1.5">
                  <path
                    d="M12 16V4M12 4l-4 4M12 4l4 4M5 20h14"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <div className="upload-actions">
                  <button
                    type="button"
                    className="button-tertiary"
                    onClick={openFilePicker}
                    disabled={uploading || isJobRunning}
                  >
                    Choose video
                  </button>
                  <span className="file-name">No file selected yet</span>
                </div>
              </>
            )}
          </div>

          <button
            type="button"
            onClick={handleStartProcessing}
            disabled={!file || uploading || isJobRunning}
            className="button-primary"
          >
            {uploading ? "Uploading..." : "Start processing"}
          </button>
        </div>

        <section className="card">
          <h2>Processing status</h2>
          <p>Status: {statusLabel}</p>
          <div className="status-progress">
            <span style={{ width: `${job ? job.progress : 0}%` }} />
          </div>

          {!job && (
            <div className="status-banner">
              <div>
                <strong>No active jobs</strong>
                <p>Upload a video and click Start processing to begin.</p>
              </div>
            </div>
          )}

          {job?.status === "queued" && (
            <div className="status-banner">
              <div>
                <strong>Queued</strong>
                <p>Hang tight while we spin up the detection pipeline.</p>
              </div>
            </div>
          )}

          {job?.status === "processing" && (
            <div className="status-banner">
              <div>
                <strong>Analysing footage</strong>
                <p>Tracking vehicles, segmenting plates, and compiling results.</p>
              </div>
            </div>
          )}

          {job?.status === "failed" && (
            <div className="status-banner failed">
              <div>
                <strong>Processing failed</strong>
                <p>{job.error ?? "An unexpected error occurred while analysing this video."}</p>
              </div>
              <button
                className="button-tertiary"
                type="button"
                onClick={() => {
                  setJob(null);
                  setJobId(null);
                  setFile(null);
                  if (previewUrl) {
                    URL.revokeObjectURL(previewUrl);
                  }
                  setPreviewUrl(null);
                  setPreviewError(false);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = "";
                  }
                }}
              >
                Reset
              </button>
            </div>
          )}

          {job?.status === "completed" && jobId && (
            <div className="status-banner">
              <div>
                <strong>Your file is ready.</strong>
                <p>Download the annotated MP4 or the CSV export containing plate metadata.</p>
              </div>
              <div className="status-links">
                <a className="button-download" href={`${API_BASE}/jobs/${jobId}/download`}>
                  Download video
                </a>
                <a className="button-download" href={`${API_BASE}/jobs/${jobId}/csv`}>
                  Download CSV
                </a>
              </div>
            </div>
          )}
        </section>
      </section>

      <section>
        <div className="feature-grid">
          {features.map((feature) => (
            <article key={feature.title} className="feature-card">
              <div className="feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
