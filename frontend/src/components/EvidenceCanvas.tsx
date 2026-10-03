import { useRef, useState } from "react";
import type { FC, MouseEvent, TouchEvent } from "react";
import { PlusIcon, XIcon, ArrowRightIcon, MaximizeIcon, GridIcon, EyeIcon } from "./Icons";
import type { ImageDimensions, HistoryItem } from "../types";

interface EvidenceCanvasProps {
  imageSrc: string;
  file: File | null;
  dimensions: ImageDimensions | null;
  isAnalyzing: boolean;
  isAnalyzed: boolean;
  onAnalyze: () => void;
  onClear: () => void;
  onChooseNew: () => void;
  onOpenFocusMode?: () => void;
  historyItems?: HistoryItem[];
  activeSpecimenId?: string | null;
  specimenNumber?: number;
  onSelectHistory?: (item: HistoryItem) => void;
}

export type EvidenceLayerMode = "original" | "grid" | "markers" | "edge";

export const EvidenceCanvas: FC<EvidenceCanvasProps> = ({
  imageSrc,
  file,
  dimensions,
  isAnalyzing,
  isAnalyzed,
  onAnalyze,
  onClear,
  onChooseNew,
  onOpenFocusMode,
  historyItems = [],
  activeSpecimenId,
  specimenNumber,
  onSelectHistory,
}) => {
  const frameRef = useRef<HTMLDivElement>(null);
  const [lensState, setLensState] = useState<{
    x: number;
    y: number;
    active: boolean;
  }>({
    x: 50,
    y: 50,
    active: false,
  });

  const [activeLayer, setActiveLayer] = useState<EvidenceLayerMode>("original");

  const [isTouchDevice] = useState(() => 
    typeof window !== "undefined" && ("ontouchstart" in window || navigator.maxTouchPoints > 0)
  );

  const updateCoordinates = (clientX: number, clientY: number) => {
    if (!frameRef.current || isAnalyzing) return;
    const rect = frameRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    const y = Math.max(0, Math.min(100, ((clientY - rect.top) / rect.height) * 100));
    setLensState({ x, y, active: true });
  };

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (isTouchDevice) return;
    updateCoordinates(e.clientX, e.clientY);
  };

  const handleMouseLeave = () => {
    setLensState((prev) => ({ ...prev, active: false }));
  };

  const handleTouchMove = (e: TouchEvent<HTMLDivElement>) => {
    if (!e.touches[0]) return;
    updateCoordinates(e.touches[0].clientX, e.touches[0].clientY);
  };

  const handleTouchEnd = () => {
    setLensState((prev) => ({ ...prev, active: false }));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className={`evidence-canvas-passport ${isAnalyzed ? "canvas-passport-analyzed" : ""}`}>
      {/* Evidence Passport Topbar */}
      <div className="canvas-header-bar">
        <div className="evidence-id-cluster">
          <span className="specimen-id-tag font-mono">
            SPECIMEN #{String(specimenNumber || 1).padStart(3, "0")}
          </span>
          <span className="evidence-filename" title={file?.name || "Specimen"}>
            {file?.name || "Specimen"}
          </span>
        </div>

        <div className="evidence-specs-cluster font-mono">
          {dimensions && (
            <span className="spec-item">{dimensions.width} × {dimensions.height}</span>
          )}
          {file && (
            <span className="spec-item">{file.type.replace("image/", "").toUpperCase()}</span>
          )}
          {file && (
            <span className="spec-item">{formatFileSize(file.size)}</span>
          )}

          <div className="canvas-top-actions">
            {onOpenFocusMode && (
              <button
                type="button"
                className="canvas-tool-btn"
                onClick={onOpenFocusMode}
                title="Expand to Focus Mode (Esc to exit)"
                aria-label="Focus mode"
              >
                <MaximizeIcon size={14} />
              </button>
            )}

            <button
              type="button"
              className="canvas-tool-btn canvas-tool-clear"
              onClick={onClear}
              title="Remove evidence"
              aria-label="Remove evidence"
              disabled={isAnalyzing}
            >
              <XIcon size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Slim Evidence Layers Control Strip */}
      <div className="evidence-layers-strip font-mono" role="toolbar" aria-label="Evidence display layers">
        <span className="layers-label">LAYER:</span>

        <button
          type="button"
          className={`layer-toggle-btn ${activeLayer === "original" ? "layer-active" : ""}`}
          onClick={() => setActiveLayer("original")}
        >
          <EyeIcon size={12} />
          <span>ORIGINAL</span>
        </button>

        <button
          type="button"
          className={`layer-toggle-btn ${activeLayer === "grid" ? "layer-active" : ""}`}
          onClick={() => setActiveLayer("grid")}
        >
          <GridIcon size={12} />
          <span>GRID</span>
        </button>

        <button
          type="button"
          className={`layer-toggle-btn ${activeLayer === "markers" ? "layer-active" : ""}`}
          onClick={() => setActiveLayer("markers")}
        >
          <span>MARKERS</span>
        </button>

        <button
          type="button"
          className={`layer-toggle-btn ${activeLayer === "edge" ? "layer-active" : ""}`}
          onClick={() => setActiveLayer("edge")}
          title="Visual-only high-pass edge contrast filter"
        >
          <span>EDGE FILTER</span>
        </button>
      </div>

      {/* Main Evidence Visual Hero Stage */}
      <div
        ref={frameRef}
        className={`evidence-display-stage ${isAnalyzing ? "stage-is-scanning" : ""} ${activeLayer === "edge" ? "layer-edge-filter" : ""}`}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {/* Four Forensic Corner Brackets */}
        {(activeLayer === "original" || activeLayer === "markers" || activeLayer === "grid") && (
          <>
            <span className="evidence-cb cb-tl" aria-hidden="true">┌</span>
            <span className="evidence-cb cb-tr" aria-hidden="true">┐</span>
            <span className="evidence-cb cb-bl" aria-hidden="true">└</span>
            <span className="evidence-cb cb-br" aria-hidden="true">┘</span>
          </>
        )}

        {/* Technical Perimeter Crosshair Markers */}
        {activeLayer === "markers" && (
          <>
            <span className="tech-marker marker-top font-mono">+</span>
            <span className="tech-marker marker-bottom font-mono">+</span>
            <span className="tech-marker marker-left font-mono">+</span>
            <span className="tech-marker marker-right font-mono">+</span>
            <div className="marker-coords-badge font-mono">
              FOV: NATIVE // {dimensions ? `${dimensions.width}×${dimensions.height}` : "AUTO"}
            </div>
          </>
        )}

        {/* Functional Technical Grid Overlay */}
        {activeLayer === "grid" && (
          <div className="canvas-grid-layer" aria-hidden="true" />
        )}

        {/* Center Forensic Optical Reticle */}
        <div className="center-optical-reticle" aria-hidden="true">
          <span className="optical-ring"></span>
          <span className="optical-cross">◎</span>
        </div>

        {/* Active Laser Scanning Beam (during analysis) */}
        {isAnalyzing && (
          <>
            <div className="forensic-scan-beam" aria-hidden="true" />
            <div className="forensic-scan-grid-overlay" aria-hidden="true" />
          </>
        )}

        {/* The Actual Image Evidence */}
        <img
          src={imageSrc}
          alt="Digital forensic specimen under examination"
          className="evidence-hero-image"
        />

        {/* Polished Interactive Forensic Magnification Loupe */}
        {lensState.active && !isAnalyzing && (
          <div
            className="forensic-lens"
            style={{
              left: `${lensState.x}%`,
              top: `${lensState.y}%`,
            }}
            aria-hidden="true"
          >
            {/* Magnified Specimen View */}
            <div
              className="lens-specimen-zoom"
              style={{
                backgroundImage: `url(${imageSrc})`,
                backgroundPosition: `${lensState.x}% ${lensState.y}%`,
                backgroundSize: `${dimensions ? Math.max(dimensions.width * 2.5, 960) : 960}px auto`,
              }}
            />
            {/* Glass highlight & reflection */}
            <div className="lens-glass-reflection" />
            
            {/* Reticle Crosshairs */}
            <div className="lens-crosshair-reticle">
              <span className="reticle-line reticle-h" />
              <span className="reticle-line reticle-v" />
              <span className="reticle-center-pip" />
            </div>

            {/* Coordinate HUD Tag */}
            <div className="lens-hud-indicator font-mono">
              <span className="hud-mag">2.5×</span>
              <span className="hud-coords">
                X: {lensState.x.toFixed(1)}% | Y: {lensState.y.toFixed(1)}%
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Evidence Thumbnails Strip (if session history exists) */}
      {historyItems.length > 0 && onSelectHistory && (
        <div className="evidence-thumbnails-strip">
          <span className="strip-label font-mono">SESSION SPECIMENS:</span>
          <div className="strip-scroll-row">
            {historyItems.slice(0, 5).map((item) => {
              const isItemActive = activeSpecimenId === item.id;
              const thumbSrc = item.thumbnailDataUrl || item.imageSrc;

              return (
                <div
                  key={item.id}
                  className={`strip-thumb-item ${isItemActive ? "thumb-active" : ""}`}
                  onClick={() => onSelectHistory(item)}
                  title={`SPECIMEN #${String(item.specimenNumber).padStart(3, "0")} - ${item.filename} (${item.verdict.replace("_", " ")})`}
                  role="button"
                  tabIndex={0}
                >
                  <img src={thumbSrc} alt="" />
                  {isItemActive ? (
                    <span className="thumb-indicator">ACTIVE</span>
                  ) : (
                    <span className="thumb-status-dot" />
                  )}
                </div>
              );
            })}

            <button
              type="button"
              className="strip-add-thumb-btn font-mono"
              onClick={onChooseNew}
              title="Add another specimen"
            >
              <PlusIcon size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Primary Action Section */}
      <div className="evidence-bottom-action-bar">
        {!isAnalyzed ? (
          <div className="action-row-pre-analysis">
            <button
              type="button"
              className="btn btn-secondary-evidence"
              onClick={onChooseNew}
              disabled={isAnalyzing}
            >
              <PlusIcon size={16} />
              <span>CHOOSE DIFFERENT IMAGE</span>
            </button>

            <button
              type="button"
              className={`btn btn-primary-analyze ${isAnalyzing ? "btn-scanning-active" : ""}`}
              onClick={onAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <span className="analyze-pulse-dot" />
                  <span>SCANNING...</span>
                  <div className="btn-progress-bar" />
                </>
              ) : (
                <>
                  <span className="analyze-sparkle font-mono">◉</span>
                  <span>ANALYZE EVIDENCE</span>
                  <ArrowRightIcon size={16} className="analyze-arrow" />
                </>
              )}
            </button>
          </div>
        ) : (
          <div className="action-row-post-analysis">
            <div className="post-analysis-status font-mono">
              <span className="status-confirmed-dot" />
              <span>EVIDENCE ANALYZED & VERIFIED</span>
            </div>

            <div className="post-analysis-actions">
              {onOpenFocusMode && (
                <button
                  type="button"
                  className="btn btn-secondary-evidence btn-sm"
                  onClick={onOpenFocusMode}
                >
                  <MaximizeIcon size={13} />
                  <span>FOCUS MODE</span>
                </button>
              )}

              <button
                type="button"
                className="btn btn-secondary-evidence btn-sm"
                onClick={onChooseNew}
              >
                <PlusIcon size={13} />
                <span>INSPECT NEW SPECIMEN</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
