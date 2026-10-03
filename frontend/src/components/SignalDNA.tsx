import { useId } from "react";
import type { FC } from "react";

interface SignalDNAProps {
  spatialProb: number;
  frequencyProb: number;
  fusionProb: number;
}

export const SignalDNA: FC<SignalDNAProps> = ({
  spatialProb,
  frequencyProb,
  fusionProb,
}) => {
  const gradientId = useId();

  // Normalize probabilities into 0.0 - 1.0 range
  const pS = (spatialProb <= 1.0 ? spatialProb : spatialProb / 100);
  const pF = (frequencyProb <= 1.0 ? frequencyProb : frequencyProb / 100);
  const pU = (fusionProb <= 1.0 ? fusionProb : fusionProb / 100);

  const width = 640;
  const height = 120;
  const baseline = 60;
  const totalPoints = 64;

  // Stream 1: Spatial Harmonic (Blue)
  const points1: [number, number][] = [];
  // Stream 2: Frequency Harmonic (Cyan)
  const points2: [number, number][] = [];
  // Stream 3: Fusion Stream (Violet)
  const points3: [number, number][] = [];

  for (let i = 0; i <= totalPoints; i++) {
    const x = (i / totalPoints) * width;
    const t = i / totalPoints;
    
    // Wave 1: Spatial
    const y1 = Math.max(12, Math.min(height - 12, baseline - (
      Math.sin(t * Math.PI * 6) * pS * 28 + Math.cos(t * Math.PI * 2) * 8
    )));
    points1.push([x, y1]);

    // Wave 2: Frequency
    const y2 = Math.max(12, Math.min(height - 12, baseline - (
      Math.cos(t * Math.PI * 10) * pF * 22 + Math.sin(t * Math.PI * 4) * 10
    )));
    points2.push([x, y2]);

    // Wave 3: Calibrated Fusion
    const y3 = Math.max(12, Math.min(height - 12, baseline - (
      Math.sin(t * Math.PI * 16) * pU * 16 + Math.cos(t * Math.PI * 8) * 12
    )));
    points3.push([x, y3]);
  }

  const makePath = (pts: [number, number][]) => 
    pts.reduce((acc, [x, y], idx) => idx === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`, "");

  const path1 = makePath(points1);
  const path2 = makePath(points2);
  const path3 = makePath(points3);
  const area1 = `${path1} L ${width} ${height} L 0 ${height} Z`;

  return (
    <div className="bento-card signal-profile-card">
      <div className="card-header-bar">
        <div className="card-title-cluster">
          <span className="card-category-tag font-mono">FORENSIC SIGNAL PROFILE</span>
          <span className="card-subtitle-tag font-mono">FLOWING MULTI-SPECTRAL BALANCES</span>
        </div>
        <div className="signal-streams-legend font-mono">
          <span className="legend-dot dot-blue">● SPATIAL</span>
          <span className="legend-dot dot-cyan">● FREQUENCY</span>
          <span className="legend-dot dot-violet">● FUSION</span>
        </div>
      </div>

      <div className="waveform-canvas-container">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="signal-waveform-svg"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id={`${gradientId}-blue`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#315FE8" />
              <stop offset="100%" stopColor="#4F7CFF" />
            </linearGradient>

            <linearGradient id={`${gradientId}-cyan`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22B8CF" />
              <stop offset="100%" stopColor="#06B6D4" />
            </linearGradient>

            <linearGradient id={`${gradientId}-violet`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#6C63D9" />
              <stop offset="100%" stopColor="#8B5CF6" />
            </linearGradient>

            <linearGradient id={`${gradientId}-area`} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#315FE8" stopOpacity="0.10" />
              <stop offset="100%" stopColor="#315FE8" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Technical horizontal reference grid */}
          <line x1="0" y1="28" x2={width} y2="28" stroke="#E2E8F0" strokeDasharray="3 3" />
          <line x1="0" y1="60" x2={width} y2="60" stroke="#CBD5E1" strokeDasharray="4 4" />
          <line x1="0" y1="92" x2={width} y2="92" stroke="#E2E8F0" strokeDasharray="3 3" />

          {/* Soft ambient area fill */}
          <path d={area1} fill={`url(#${gradientId}-area)`} />

          {/* Stream 1: Spatial (Blue) */}
          <path
            d={path1}
            fill="none"
            stroke={`url(#${gradientId}-blue)`}
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="signal-flowing-path"
          />

          {/* Stream 2: Frequency (Cyan) */}
          <path
            d={path2}
            fill="none"
            stroke={`url(#${gradientId}-cyan)`}
            strokeWidth="2.0"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="6 3"
            className="signal-flowing-path"
          />

          {/* Stream 3: Fusion (Violet) */}
          <path
            d={path3}
            fill="none"
            stroke={`url(#${gradientId}-violet)`}
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="signal-flowing-path"
          />
        </svg>
      </div>

      <div className="signal-profile-footer">
        <p className="signal-profile-disclaimer font-mono">
          Visual representation of detected signal balances across spatial and frequency channels. Not raw FFT output.
        </p>
        <div className="signal-metric-pills font-mono">
          <span className="pill-metric">SPATIAL: {(pS * 100).toFixed(1)}%</span>
          <span className="pill-metric">FREQ: {(pF * 100).toFixed(1)}%</span>
          <span className="pill-metric">FUSION: {(pU * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
};
