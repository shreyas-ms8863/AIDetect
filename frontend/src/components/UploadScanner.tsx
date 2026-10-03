import { useRef, useState } from "react";
import type { FC, DragEvent, ChangeEvent } from "react";
import { PlusIcon } from "./Icons";

interface UploadScannerProps {
  onFileSelect: (file: File | undefined) => void;
}

export const UploadScanner: FC<UploadScannerProps> = ({ onFileSelect }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    onFileSelect(e.dataTransfer.files?.[0]);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    onFileSelect(e.target.files?.[0]);
  };

  return (
    <div className="scanner-container">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={handleInputChange}
        style={{ display: "none" }}
      />

      <div
        className={`evidence-scanner-card ${isDragging ? "scanner-dragging" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
      >
        {/* Subtle technical background grid */}
        <div className="scanner-bg-grid" aria-hidden="true" />

        {/* Layered forensic specimen cards illustration in background */}
        <div className="layered-specimen-illustration" aria-hidden="true">
          <div className="specimen-layer layer-back" />
          <div className="specimen-layer layer-mid" />
          <div className="specimen-layer layer-front">
            <span className="specimen-layer-corner tl">┌</span>
            <span className="specimen-layer-corner tr">┐</span>
            <span className="specimen-layer-corner bl">└</span>
            <span className="specimen-layer-corner br">┘</span>
            <span className="specimen-layer-reticle">◎</span>
          </div>
        </div>

        {/* Forensic Corner Brackets */}
        <span className="corner-bracket cb-tl" aria-hidden="true">┌</span>
        <span className="corner-bracket cb-tr" aria-hidden="true">┐</span>
        <span className="corner-bracket cb-bl" aria-hidden="true">└</span>
        <span className="corner-bracket cb-br" aria-hidden="true">┘</span>

        {/* Thin animated scanning beam */}
        <div className="scanner-idle-scanline" aria-hidden="true" />

        {/* Technical Corner Markers */}
        <div className="scanner-meta-tl font-mono">SPECIMEN // RECEPTACLE 01</div>
        <div className="scanner-meta-tr font-mono">STANDBY READY</div>
        <div className="scanner-meta-bl font-mono">AUTO-SCALE: ON</div>
        <div className="scanner-meta-br font-mono">COLOR: sRGB/P3</div>

        {/* Main Content Area */}
        <div className="scanner-content">
          <div className="scanner-prompt-cluster">
            <h2 className="scanner-primary-title">DROP IMAGE HERE</h2>
            <span className="scanner-or-divider">or</span>
            <button
              type="button"
              className="btn btn-select-image"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              <PlusIcon size={16} />
              <span>SELECT IMAGE</span>
            </button>
          </div>

          <p className="scanner-supported-text font-mono">
            SUPPORTED FORMATS: PNG • JPG • JPEG • WEBP
          </p>
        </div>
      </div>
    </div>
  );
};
