import { useState, useRef, useEffect } from "react";
import type { FC, MouseEvent } from "react";
import { XIcon, GridIcon, EyeIcon } from "./Icons";
import type { ImageDimensions } from "../types";

interface FocusModeModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageSrc: string;
  filename?: string;
  dimensions?: ImageDimensions | null;
}

export const FocusModeModal: FC<FocusModeModalProps> = ({
  isOpen,
  onClose,
  imageSrc,
  filename = "Specimen",
  dimensions,
}) => {
  const frameRef = useRef<HTMLDivElement>(null);
  const [lensEnabled, setLensEnabled] = useState(true);
  const [gridEnabled, setGridEnabled] = useState(false);
  const [markersEnabled, setMarkersEnabled] = useState(true);
  const [highPassEnabled, setHighPassEnabled] = useState(false);

  const [lensPos, setLensPos] = useState<{ x: number; y: number; show: boolean }>({
    x: 50,
    y: 50,
    show: false,
  });

  // Handle ESC key to exit
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!frameRef.current || !lensEnabled) return;
    const rect = frameRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    const y = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
    setLensPos({ x, y, show: true });
  };

  const handleMouseLeave = () => {
    setLensPos((prev) => ({ ...prev, show: false }));
  };

  return (
    <div className="focus-mode-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="focus-mode-stage-wrapper" onClick={(e) => e.stopPropagation()}>
        {/* Minimal Focus Mode Top Header */}
        <div className="focus-mode-header font-mono">
          <div className="focus-header-meta">
            <span className="focus-badge">FOCUS MODE</span>
            <span className="focus-filename">{filename}</span>
            {dimensions && <span className="focus-dims">{dimensions.width}×{dimensions.height}</span>}
          </div>

          <div className="focus-header-controls">
            <button
              type="button"
              className={`focus-pill-btn ${lensEnabled ? "focus-btn-active" : ""}`}
              onClick={() => setLensEnabled(!lensEnabled)}
            >
              <EyeIcon size={12} />
              <span>LENS: {lensEnabled ? "ON" : "OFF"}</span>
            </button>

            <button
              type="button"
              className={`focus-pill-btn ${gridEnabled ? "focus-btn-active" : ""}`}
              onClick={() => setGridEnabled(!gridEnabled)}
            >
              <GridIcon size={12} />
              <span>GRID</span>
            </button>

            <button
              type="button"
              className={`focus-pill-btn ${markersEnabled ? "focus-btn-active" : ""}`}
              onClick={() => setMarkersEnabled(!markersEnabled)}
            >
              <span>MARKERS</span>
            </button>

            <button
              type="button"
              className={`focus-pill-btn ${highPassEnabled ? "focus-btn-active" : ""}`}
              onClick={() => setHighPassEnabled(!highPassEnabled)}
            >
              <span>EDGE FILTER</span>
            </button>

            <button
              type="button"
              className="focus-close-btn"
              onClick={onClose}
              title="Exit Focus Mode (Esc)"
              aria-label="Exit Focus Mode"
            >
              <XIcon size={18} />
              <span className="esc-tag">ESC</span>
            </button>
          </div>
        </div>

        {/* Maximized Evidence Frame */}
        <div
          ref={frameRef}
          className={`focus-canvas-stage ${highPassEnabled ? "layer-edge-filter" : ""}`}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {markersEnabled && (
            <>
              <span className="evidence-cb cb-tl">┌</span>
              <span className="evidence-cb cb-tr">┐</span>
              <span className="evidence-cb cb-bl">└</span>
              <span className="evidence-cb cb-br">┘</span>
              <span className="tech-marker marker-top font-mono">+</span>
              <span className="tech-marker marker-bottom font-mono">+</span>
            </>
          )}

          {gridEnabled && <div className="canvas-grid-layer" aria-hidden="true" />}

          <img src={imageSrc} alt="" className="focus-hero-image" />

          {lensEnabled && lensPos.show && (
            <div
              className="forensic-lens focus-lens"
              style={{
                left: `${lensPos.x}%`,
                top: `${lensPos.y}%`,
              }}
              aria-hidden="true"
            >
              <div
                className="lens-specimen-zoom"
                style={{
                  backgroundImage: `url(${imageSrc})`,
                  backgroundPosition: `${lensPos.x}% ${lensPos.y}%`,
                  backgroundSize: `${dimensions ? Math.max(dimensions.width * 2.8, 1100) : 1100}px auto`,
                }}
              />
              <div className="lens-glass-reflection" />
              <div className="lens-crosshair-reticle">
                <span className="reticle-line reticle-h" />
                <span className="reticle-line reticle-v" />
                <span className="reticle-center-pip" />
              </div>
              <div className="lens-hud-indicator font-mono">
                <span className="hud-mag">2.8×</span>
                <span className="hud-coords">
                  X:{lensPos.x.toFixed(1)}% | Y:{lensPos.y.toFixed(1)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
