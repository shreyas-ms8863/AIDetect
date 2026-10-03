import type { FC } from "react";
import type { SystemStatus } from "../types";

interface HeaderProps {
  status: SystemStatus;
  onReset: () => void;
  onEvidenceClick?: () => void;
  onHistoryClick?: () => void;
  hasEvidence?: boolean;
  historyCount?: number;
}

export const Header: FC<HeaderProps> = ({
  status,
  onReset,
  onEvidenceClick,
  onHistoryClick,
  hasEvidence = false,
  historyCount = 0,
}) => {
  return (
    <header className="forensic-header">
      <div className="header-inner">
        {/* Left: Brand + Digital Forensics label */}
        <div 
          className="brand-cluster" 
          onClick={onReset} 
          role="button" 
          tabIndex={0} 
          onKeyDown={(e) => e.key === "Enter" && onReset()}
          title="AIDetect Home"
        >
          <div className="brand-dot-pulse"></div>
          <div className="brand-titles">
            <span className="brand-wordmark">AIDETECT</span>
            <span className="brand-submark">DIGITAL FORENSICS</span>
          </div>
        </div>

        {/* Right: Minimal Navigation (ANALYZE / EVIDENCE / HISTORY) + Status */}
        <nav className="header-nav-cluster" aria-label="Forensic navigation">
          <button
            type="button"
            className="nav-link-btn nav-link-active"
            onClick={onReset}
            title="Start new analysis"
          >
            ANALYZE
          </button>

          <button
            type="button"
            className={`nav-link-btn ${hasEvidence ? "" : "nav-link-disabled"}`}
            onClick={hasEvidence ? onEvidenceClick : undefined}
            disabled={!hasEvidence}
            title={hasEvidence ? "Scroll to current evidence" : "No evidence loaded"}
          >
            EVIDENCE
          </button>

          <button
            type="button"
            className="nav-link-btn nav-link-history"
            onClick={onHistoryClick}
            title="View Evidence Archive"
          >
            ARCHIVE
            {historyCount > 0 && <span className="nav-history-badge">{historyCount}</span>}
          </button>

          <div className="nav-divider" />

          {/* Engine Status Beacon */}
          <div className="header-engine-beacon" title={status.connected ? "Inference Engine Active" : "Connecting..."}>
            <span className={`beacon-dot ${status.connected ? "beacon-live" : "beacon-offline"}`} />
            <span className="beacon-label font-mono">
              {status.connected ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </nav>
      </div>
    </header>
  );
};
