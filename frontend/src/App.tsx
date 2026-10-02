import { useRef, useState, useEffect } from "react";
import "./App.css";

type AnalysisState = "upload" | "analyzing" | "results";

interface ModelResult {
  prediction: "AI-GENERATED" | "REAL";
  confidence: number;
  ai_probability: number;
  real_probability: number;
}

interface AnalysisResult {
  filename: string;
  prediction: "AI-GENERATED" | "REAL";
  confidence: number;
  ai_probability: number;
  real_probability: number;
  models: {
    spatial: ModelResult;
    frequency: ModelResult;
    hybrid: ModelResult;
  };
  robustness: {
    original: ModelResult;
    jpeg_compression: ModelResult;
    resize: ModelResult;
    blur: ModelResult;
    noise: ModelResult;
  };
  message: string;
}

interface ImageDimensions {
  width: number;
  height: number;
}

const PIPELINE_STAGES = [
  { name: "Image preprocessing", desc: "Resizing and RGB normalization" },
  { name: "Spatial V3", desc: "ResNet50 visual texture analysis" },
  { name: "Frequency V3", desc: "2D Fast Fourier Transform spectrum" },
  { name: "Hybrid V3", desc: "Spatial + frequency feature fusion" },
  { name: "Robustness analysis", desc: "Perturbation and compression tests" },
];

// Helper functions
const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

const formatFileType = (file: File | null): string => {
  if (!file) return "Unknown";
  if (file.type === "image/jpeg") return "JPEG / JPG";
  if (file.type === "image/png") return "PNG";
  if (file.type === "image/webp") return "WebP";
  return file.type.replace("image/", "").toUpperCase();
};

// ============================================================
// TOP-LEVEL SUB-COMPONENTS (Declared outside App for performance)
// ============================================================

function ModelCard({
  name,
  description,
  result,
  isPrimary,
}: {
  name: string;
  description: string;
  result: ModelResult;
  isPrimary?: boolean;
}) {
  const isReal = result.prediction === "REAL";

  return (
    <div className={`model-card ${isPrimary ? "model-card-primary" : ""}`}>
      <div className="model-card-header">
        <div>
          <div className="model-name-row">
            <h3 className="model-title">{name}</h3>
            {isPrimary && <span className="primary-pill">Primary Fusion</span>}
          </div>
          <p className="model-desc">{description}</p>
        </div>
        <span
          className={`badge-status ${
            isReal ? "badge-real" : "badge-ai"
          }`}
        >
          {result.prediction}
        </span>
      </div>

      <div className="model-stat-grid">
        <div className="stat-box">
          <span className="stat-label">Confidence</span>
          <span className="stat-value">{result.confidence.toFixed(2)}%</span>
        </div>
        <div className="stat-box">
          <span className="stat-label">AI Probability</span>
          <span className="stat-value text-ai">
            {result.ai_probability.toFixed(2)}%
          </span>
        </div>
        <div className="stat-box">
          <span className="stat-label">Real Probability</span>
          <span className="stat-value text-real">
            {result.real_probability.toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="probability-bar-container">
        <div className="bar-labels">
          <span>Real {result.real_probability.toFixed(1)}%</span>
          <span>AI {result.ai_probability.toFixed(1)}%</span>
        </div>
        <div
          className="dual-progress-bar"
          role="progressbar"
          aria-valuenow={result.real_probability}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="bar-segment bar-real"
            style={{ width: `${result.real_probability}%` }}
          />
          <div
            className="bar-segment bar-ai"
            style={{ width: `${result.ai_probability}%` }}
          />
        </div>
      </div>
    </div>
  );
}

function RobustnessCard({
  title,
  result,
}: {
  title: string;
  result: ModelResult;
}) {
  const isReal = result.prediction === "REAL";

  return (
    <div className="robustness-card">
      <span className="robustness-title">{title}</span>
      <div className="robustness-score">
        <span className="robustness-confidence">
          {result.confidence.toFixed(2)}%
        </span>
        <span className="robustness-subtext">Confidence</span>
      </div>
      <span
        className={`badge-status badge-sm ${
          isReal ? "badge-real" : "badge-ai"
        }`}
      >
        {result.prediction}
      </span>
    </div>
  );
}

// ============================================================
// MAIN APP COMPONENT
// ============================================================

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [imageDimensions, setImageDimensions] = useState<ImageDimensions | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("upload");
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [pipelineIndex, setPipelineIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up object URL when changed or unmounted
  useEffect(() => {
    return () => {
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage);
      }
    };
  }, [selectedImage]);

  // ============================================================
  // IMAGE SELECTION & METADATA
  // ============================================================
  const handleImageChange = (file: File | undefined) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please select a valid image file (PNG, JPG, or WebP).");
      return;
    }

    setErrorMessage(null);
    setFileName(file.name);
    setSelectedFile(file);

    const objectUrl = URL.createObjectURL(file);
    setSelectedImage(objectUrl);
    setAnalysisResult(null);
    setAnalysisState("upload");

    // Extract genuine image resolution
    const img = new Image();
    img.onload = () => {
      setImageDimensions({
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };
    img.src = objectUrl;
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleImageChange(event.target.files?.[0]);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    handleImageChange(event.dataTransfer.files?.[0]);
  };

  // ============================================================
  // ANALYZE IMAGE PIPELINE
  // ============================================================
  const handleAnalyze = async () => {
    if (!selectedFile) {
      setErrorMessage("Please select an image to inspect.");
      return;
    }

    setErrorMessage(null);
    setAnalysisState("analyzing");
    setPipelineIndex(0);

    // Dynamic progression through the stages while backend is analyzing
    const stageInterval = setInterval(() => {
      setPipelineIndex((prev) => {
        if (prev < 3) return prev + 1;
        return prev;
      });
    }, 450);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorDetail = `Server responded with HTTP ${response.status}`;
        try {
          const errData = await response.json();
          if (errData.detail) errorDetail = errData.detail;
        } catch {
          // fallback
        }
        throw new Error(errorDetail);
      }

      const data: AnalysisResult = await response.json();

      // Show final stage completion cleanly
      clearInterval(stageInterval);
      setPipelineIndex(4);

      setTimeout(() => {
        setAnalysisResult(data);
        setAnalysisState("results");
      }, 400);
    } catch (error) {
      clearInterval(stageInterval);
      console.error("Forensic analysis error:", error);
      const message =
        error instanceof Error
          ? error.message
          : "An unexpected network or inference error occurred.";

      setErrorMessage(
        `Analysis failed: ${message}. Verify the backend server is running at http://127.0.0.1:8000`
      );
      setAnalysisState("upload");
    }
  };

  // ============================================================
  // RESET
  // ============================================================
  const handleReset = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setFileName("");
    setImageDimensions(null);
    setAnalysisResult(null);
    setAnalysisState("upload");
    setErrorMessage(null);
    setPipelineIndex(0);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Check model agreement dynamically from backend response
  const isAllAgree =
    analysisResult &&
    analysisResult.models.spatial.prediction ===
      analysisResult.models.frequency.prediction &&
    analysisResult.models.frequency.prediction ===
      analysisResult.models.hybrid.prediction;

  return (
    <div className="app-container">
      {/* ======================================================
          NAVBAR
      ======================================================= */}
      <header className="navbar">
        <div className="nav-container">
          <div className="logo" onClick={handleReset} style={{ cursor: "pointer" }}>
            <div className="logo-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <circle cx="12" cy="11" r="3"/>
              </svg>
            </div>
            <div className="logo-text">
              <span className="brand-name">AIDetect</span>
              <span className="brand-badge">Forensics V3</span>
            </div>
          </div>

          <nav className="nav-links">
            <a href="#how-it-works">How It Works</a>
            <a href="#models">Models</a>
            <a href="#about">About</a>
          </nav>
        </div>
      </header>

      <main className="main-content">
        {/* Error Banner */}
        {errorMessage && (
          <div className="error-banner" role="alert">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <span>{errorMessage}</span>
            <button className="error-close" onClick={() => setErrorMessage(null)} aria-label="Close error">
              ×
            </button>
          </div>
        )}

        {/* ====================================================
            UPLOAD SCREEN
        ===================================================== */}
        {analysisState === "upload" && (
          <div className="screen-wrapper">
            <section className="hero-section">
              <div className="hero-pill">
                <span className="pill-dot"></span>
                <span>AI IMAGE FORENSICS SUITE</span>
              </div>

              <h1 className="hero-title">
                Is this image AI-generated?
              </h1>

              <p className="hero-subtitle">
                Analyze your image using spatial, frequency-domain, and hybrid AI detection models.
              </p>

              {/* Upload Card */}
              <div
                className={`upload-zone ${isDragging ? "upload-zone-dragging" : ""} ${
                  selectedImage ? "upload-zone-has-file" : ""
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {!selectedImage ? (
                  <div className="upload-empty-state">
                    <div className="upload-icon-circle">
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                      </svg>
                    </div>

                    <h2 className="upload-heading">Drop your image here</h2>
                    <p className="upload-subtext">or choose a file from your computer</p>

                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Choose a File
                    </button>

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      onChange={handleFileInput}
                      hidden
                    />

                    <div className="upload-formats">
                      <span>Supported formats: PNG, JPG, JPEG, WebP</span>
                    </div>
                  </div>
                ) : (
                  <div className="upload-preview-state">
                    <div className="preview-image-wrapper">
                      <img
                        src={selectedImage}
                        alt="Selected for forensic analysis"
                        className="preview-image"
                      />
                    </div>

                    <div className="preview-info-card">
                      <div className="file-detail-header">
                        <span className="file-name-text">{fileName}</span>
                        <span className="file-size-badge">
                          {selectedFile ? formatFileSize(selectedFile.size) : ""}
                        </span>
                      </div>

                      {imageDimensions && (
                        <div className="file-meta-pills">
                          <span className="meta-pill">
                            {imageDimensions.width} × {imageDimensions.height} px
                          </span>
                          <span className="meta-pill">
                            {formatFileType(selectedFile)}
                          </span>
                        </div>
                      )}

                      <div className="preview-actions-row">
                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          Change File
                        </button>

                        <button
                          type="button"
                          className="btn btn-primary btn-analyze"
                          onClick={handleAnalyze}
                        >
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8"/>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                          </svg>
                          Analyze Image
                        </button>
                      </div>

                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/png,image/jpeg,image/webp"
                        onChange={handleFileInput}
                        hidden
                      />
                    </div>
                  </div>
                )}
              </div>
            </section>

            {/* Approaches Section */}
            <section className="section-block" id="how-it-works">
              <div className="section-header-center">
                <span className="section-eyebrow">DETECTION METHODOLOGY</span>
                <h2 className="section-title">Tri-Domain Forensic Architecture</h2>
                <p className="section-desc">
                  AIDetect evaluates authenticity across spatial textures, spectral frequencies, and high-order fusion features to detect subtle generator artifacts.
                </p>
              </div>

              <div className="method-grid" id="models">
                <div className="method-card">
                  <div className="method-index">01</div>
                  <h3 className="method-name">Spatial Analysis</h3>
                  <p className="method-desc">
                    Deep ResNet50 convolutional backbone inspects pixel-level color continuity, gradient inconsistencies, and structural synthesis boundaries.
                  </p>
                  <span className="method-tag">Spatial V3</span>
                </div>

                <div className="method-card">
                  <div className="method-index">02</div>
                  <h3 className="method-name">Frequency Analysis</h3>
                  <p className="method-desc">
                    2D Fast Fourier Transform (FFT) isolates frequency spectrum anomalies, uncovering periodic grid patterns and upsampling artifacts invisible in standard pixels.
                  </p>
                  <span className="method-tag">Frequency V3</span>
                </div>

                <div className="method-card method-card-highlight">
                  <div className="method-index">03</div>
                  <h3 className="method-name">Hybrid Fusion</h3>
                  <p className="method-desc">
                    Multi-modal feature fusion network combines spatial CNN representations with spectral Fourier embeddings into a unified classification decision.
                  </p>
                  <span className="method-tag method-tag-primary">Hybrid V3 (Primary)</span>
                </div>
              </div>
            </section>

            {/* Forensic Mission Section */}
            <section className="forensics-about-section" id="about">
              <div className="about-content">
                <div className="about-left">
                  <span className="section-eyebrow">FORENSIC RELIABILITY</span>
                  <h2 className="section-title">Why Multi-Model Verification Matters</h2>
                </div>
                <div className="about-right">
                  <p className="about-text">
                    Generative models have become exceptionally adept at mimicking natural photography. A single classifier looking solely at spatial textures can easily produce false positives on complex lighting, mirrors, or high ISO grain. By evaluating spatial features alongside Fourier domain signatures and hybrid fusion, AIDetect delivers a transparent, cross-validated authenticity assessment.
                  </p>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ====================================================
            ANALYZING SCREEN (PIPELINE STEPPER)
        ===================================================== */}
        {analysisState === "analyzing" && (
          <section className="analyzing-section">
            <div className="analyzing-container">
              <div className="scanner-badge">
                <span className="pulse-indicator"></span>
                <span>PROCESSING FORENSIC PIPELINE</span>
              </div>

              <h1 className="analyzing-title">Analyzing Image Authenticity</h1>
              <p className="analyzing-subtitle">
                Running multi-model forensic inference across spatial, spectral, and perturbation vectors.
              </p>

              {/* Polished Stepper */}
              <div className="pipeline-card">
                <div className="pipeline-list">
                  {PIPELINE_STAGES.map((stage, idx) => {
                    const isDone = pipelineIndex > idx;
                    const isActive = pipelineIndex === idx;

                    return (
                      <div
                        key={stage.name}
                        className={`pipeline-step ${
                          isDone
                            ? "step-done"
                            : isActive
                            ? "step-active"
                            : "step-pending"
                        }`}
                      >
                        <div className="step-marker">
                          {isDone ? (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                              <polyline points="20 6 9 17 4 12"/>
                            </svg>
                          ) : isActive ? (
                            <span className="step-dot-pulse">●</span>
                          ) : (
                            <span className="step-dot-empty">○</span>
                          )}
                        </div>

                        <div className="step-content">
                          <span className="step-title">{stage.name}</span>
                          <span className="step-subtext">{stage.desc}</span>
                        </div>

                        <div className="step-state-label">
                          {isDone ? (
                            <span className="state-badge state-done">Completed</span>
                          ) : isActive ? (
                            <span className="state-badge state-active">Analyzing...</span>
                          ) : (
                            <span className="state-badge state-pending">Queued</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ====================================================
            RESULTS SCREEN
        ===================================================== */}
        {analysisState === "results" && analysisResult && (
          <div className="results-container">
            {/* Top Analysis Header */}
            <div className="results-hero-header">
              <span className="section-eyebrow">ANALYSIS COMPLETE</span>
              <h1 className="results-main-title">Image Authenticity Report</h1>
              <p className="results-main-subtitle">
                Forensic evaluation generated by Spatial V3, Frequency V3, and Hybrid V3 fusion architectures.
              </p>
            </div>

            {/* Primary Analysis Showcase */}
            <div className="primary-dashboard-grid">
              {/* Left Column: Image preview + Metadata */}
              <div className="preview-meta-column">
                <div className="card image-inspect-card">
                  <div className="card-top-tag">
                    <span>Inspected Image</span>
                  </div>
                  <div className="inspected-image-container">
                    <img
                      src={selectedImage || ""}
                      alt="Inspected file preview"
                      className="inspected-image"
                    />
                  </div>
                </div>

                {/* Genuine Image Information Card */}
                <div className="card meta-info-card">
                  <h3 className="meta-card-title">Image Information</h3>
                  <div className="meta-grid">
                    <div className="meta-row">
                      <span className="meta-label">File Name</span>
                      <span className="meta-value truncate" title={fileName}>
                        {fileName}
                      </span>
                    </div>
                    <div className="meta-row">
                      <span className="meta-label">Resolution</span>
                      <span className="meta-value">
                        {imageDimensions
                          ? `${imageDimensions.width} × ${imageDimensions.height} px`
                          : "Detected via buffer"}
                      </span>
                    </div>
                    <div className="meta-row">
                      <span className="meta-label">File Type</span>
                      <span className="meta-value">
                        {formatFileType(selectedFile)}
                      </span>
                    </div>
                    <div className="meta-row">
                      <span className="meta-label">File Size</span>
                      <span className="meta-value">
                        {selectedFile ? formatFileSize(selectedFile.size) : "—"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Hybrid V3 Primary Result */}
              <div className="card primary-result-card">
                <div className="primary-result-top">
                  <span className="primary-result-eyebrow">HYBRID V3 PRIMARY PREDICTION</span>
                  <div
                    className={`large-prediction-badge ${
                      analysisResult.prediction === "REAL"
                        ? "badge-real-large"
                        : "badge-ai-large"
                    }`}
                  >
                    {analysisResult.prediction === "REAL" ? (
                      <>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                          <polyline points="22 4 12 14.01 9 11.01"/>
                        </svg>
                        <span>REAL IMAGE</span>
                      </>
                    ) : (
                      <>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                        </svg>
                        <span>AI-GENERATED</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="primary-confidence-section">
                  <div className="confidence-numeric">
                    <span className="confidence-number">
                      {analysisResult.confidence.toFixed(2)}%
                    </span>
                    <span className="confidence-caption">Prediction Confidence</span>
                  </div>

                  {/* Dual Probability Display */}
                  <div className="dual-probability-box">
                    <div className="prob-item">
                      <span className="prob-label">AI Probability</span>
                      <span className="prob-number text-ai">
                        {analysisResult.ai_probability.toFixed(2)}%
                      </span>
                    </div>
                    <div className="prob-item">
                      <span className="prob-label">Real Probability</span>
                      <span className="prob-number text-real">
                        {analysisResult.real_probability.toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  {/* Probability Ratio Bar */}
                  <div className="large-progress-wrapper">
                    <div className="dual-progress-bar dual-bar-lg">
                      <div
                        className="bar-segment bar-real"
                        style={{ width: `${analysisResult.real_probability}%` }}
                        title={`Real: ${analysisResult.real_probability.toFixed(2)}%`}
                      />
                      <div
                        className="bar-segment bar-ai"
                        style={{ width: `${analysisResult.ai_probability}%` }}
                        title={`AI: ${analysisResult.ai_probability.toFixed(2)}%`}
                      />
                    </div>
                  </div>
                </div>

                {/* Dynamic Model Agreement Section */}
                <div className="agreement-box">
                  <span className="agreement-title">Model Agreement</span>
                  {isAllAgree ? (
                    <div className="agreement-content agree-success">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                      </svg>
                      <div>
                        <strong>✓ All three models agree</strong>
                        <p>Spatial V3, Frequency V3, and Hybrid V3 independently classify this image as {analysisResult.prediction}.</p>
                      </div>
                    </div>
                  ) : (
                    <div className="agreement-content agree-warning">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                      </svg>
                      <div>
                        <strong>⚠ Models disagree</strong>
                        <p>
                          Spatial: <b>{analysisResult.models.spatial.prediction}</b> ({analysisResult.models.spatial.confidence.toFixed(1)}%), Frequency: <b>{analysisResult.models.frequency.prediction}</b> ({analysisResult.models.frequency.confidence.toFixed(1)}%), Hybrid: <b>{analysisResult.models.hybrid.prediction}</b> ({analysisResult.models.hybrid.confidence.toFixed(1)}%).
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Forensic Interpretation */}
                <div className="forensic-interpretation-note">
                  <div className="note-header">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"/>
                      <line x1="12" y1="16" x2="12" y2="12"/>
                      <line x1="12" y1="8" x2="12.01" y2="8"/>
                    </svg>
                    <strong>Forensic Interpretation</strong>
                  </div>
                  <p>
                    The Hybrid V3 model fuses deep spatial convolutional textures with 2D Fast Fourier frequency representations, observing empirical characteristics consistent with <b>{analysisResult.prediction}</b> photography.
                  </p>
                  <small>
                    AI detection models are probabilistic statistical estimators and should be utilized alongside human review and provenance verification.
                  </small>
                </div>
              </div>
            </div>

            {/* Model Comparison Section */}
            <section className="section-block comparison-section">
              <div className="section-header-left">
                <span className="section-eyebrow">MODEL COMPARISON</span>
                <h2 className="section-title">Comparative Multi-Model Outputs</h2>
                <p className="section-desc">
                  Inspect how each model independently evaluates the image across spatial representations, Fourier frequency spectrum, and unified fusion.
                </p>
              </div>

              <div className="models-row-grid">
                <ModelCard
                  name="Spatial V3"
                  description="ResNet50 spatial features"
                  result={analysisResult.models.spatial}
                />
                <ModelCard
                  name="Frequency V3"
                  description="FFT-based frequency features"
                  result={analysisResult.models.frequency}
                />
                <ModelCard
                  name="Hybrid V3"
                  description="Spatial + FFT feature fusion"
                  result={analysisResult.models.hybrid}
                  isPrimary
                />
              </div>
            </section>

            {/* Robustness Section */}
            <section className="section-block robustness-block">
              <div className="section-header-left">
                <span className="section-eyebrow">ROBUSTNESS CHECK</span>
                <h2 className="section-title">Robustness Check</h2>
                <p className="section-desc">
                  See how Hybrid V3 responds to common image transformations.
                </p>
              </div>

              <div className="robustness-row-grid">
                <RobustnessCard
                  title="Original"
                  result={analysisResult.robustness.original}
                />
                <RobustnessCard
                  title="JPEG Compression"
                  result={analysisResult.robustness.jpeg_compression}
                />
                <RobustnessCard
                  title="Resize"
                  result={analysisResult.robustness.resize}
                />
                <RobustnessCard
                  title="Blur"
                  result={analysisResult.robustness.blur}
                />
                <RobustnessCard
                  title="Noise"
                  result={analysisResult.robustness.noise}
                />
              </div>

              <div className="robustness-warning-card">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <p>
                  <strong>Note:</strong> These results demonstrate how Hybrid V3 evaluates this specific transformed image. They are individual inference predictions and do not represent benchmark dataset-level robustness accuracy, which is evaluated across thousands of standardized test samples.
                </p>
              </div>
            </section>

            {/* Bottom Actions */}
            <div className="results-actions-wrapper">
              <button
                type="button"
                className="btn btn-primary btn-reset"
                onClick={handleReset}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                  <path d="M3 3v5h5"/>
                </svg>
                Analyze Another Image
              </button>
            </div>
          </div>
        )}
      </main>

      {/* ======================================================
          FOOTER
      ======================================================= */}
      <footer className="footer">
        <div className="footer-container">
          <div className="footer-brand">
            <div className="logo-icon-sm">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </div>
            <span className="footer-title">AIDetect Forensics V3</span>
          </div>

          <p className="footer-text">
            Spatial, Frequency, and Multi-Modal AI Image Authenticity Detection.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;