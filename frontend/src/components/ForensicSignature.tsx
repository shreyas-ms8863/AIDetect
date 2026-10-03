import type { FC } from "react";
import { FingerprintIcon } from "./Icons";

interface ForensicSignatureProps {
  genAiProb: number;
  manipProb: number;
  fusionConf: number;
}

export const ForensicSignature: FC<ForensicSignatureProps> = ({
  genAiProb,
  manipProb,
  fusionConf,
}) => {
  const pGen = Math.round((genAiProb <= 1.0 ? genAiProb * 100 : genAiProb));
  const pManip = Math.round((manipProb <= 1.0 ? manipProb * 100 : manipProb));
  const pFusion = Math.round((fusionConf <= 1.0 ? fusionConf * 100 : fusionConf));

  // Deterministically generate 18 signature bars based on signal weights
  const bars = Array.from({ length: 18 }, (_, idx) => {
    const factor = Math.sin((idx / 18) * Math.PI) * 0.6 + 0.4;
    const height = Math.round(15 + factor * ((idx % 2 === 0 ? pGen : pManip) * 0.4 + pFusion * 0.25));
    return Math.min(64, Math.max(12, height));
  });

  return (
    <div className="bento-card signature-fingerprint-card">
      <div className="card-header-bar">
        <div className="card-title-cluster">
          <span className="card-category-tag font-mono">FORENSIC SIGNATURE</span>
          <span className="card-subtitle-tag font-mono">DIGITAL EVIDENCE FINGERPRINT</span>
        </div>
        <div className="card-icon-pill">
          <FingerprintIcon size={16} />
        </div>
      </div>

      <div className="signature-visual-core">
        {/* Radial Concentric Markers */}
        <div className="signature-radial-rings" aria-hidden="true">
          <div className="radial-ring ring-outer" />
          <div className="radial-ring ring-mid" />
          <div className="radial-ring ring-inner" />
          <div className="radial-pulse-glow" />
        </div>

        {/* Dynamic Vertical Forensic Bars */}
        <div className="signature-bars-spectrum" aria-hidden="true">
          {bars.map((h, i) => (
            <div
              key={i}
              className={`signature-bar ${i % 3 === 0 ? "bar-accent-cyan" : i % 2 === 0 ? "bar-accent-violet" : "bar-accent-blue"}`}
              style={{ height: `${h}px` }}
            />
          ))}
        </div>
      </div>

      {/* Tri-Axial Summary Metrics */}
      <div className="signature-triad-grid font-mono">
        <div className="triad-item">
          <span className="triad-label">GEN SIGNAL</span>
          <span className="triad-value">{pGen}%</span>
        </div>
        <div className="triad-item">
          <span className="triad-label">MANIP SIGNAL</span>
          <span className="triad-value">{pManip}%</span>
        </div>
        <div className="triad-item">
          <span className="triad-label">FUSION</span>
          <span className="triad-value">{pFusion}%</span>
        </div>
      </div>

      <p className="signature-disclaimer">
        Digital fingerprint visualization derived from model agreement weights.
      </p>
    </div>
  );
};
