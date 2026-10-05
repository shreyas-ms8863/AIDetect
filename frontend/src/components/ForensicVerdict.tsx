import { useEffect, useState } from "react";
import type { FC, ReactNode } from "react";
import { ShieldCheckIcon, SparklesIcon, ImagePlusIcon, TriangleAlertIcon } from "./Icons";
import type { ForensicFinalResult, ForensicVerdict as VerdictType, ImageDimensions } from "../types";

interface ForensicVerdictProps {
  finalResult: ForensicFinalResult;
  filename?: string;
  dimensions?: ImageDimensions | null;
}

interface VerdictConfig {
  label: string;
  themeClass: string;
  accentColor: string;
  badgeBg: string;
  icon: ReactNode;
  explanation: string;
}

export const ForensicVerdict: FC<ForensicVerdictProps> = ({
  finalResult,
  filename = "Specimen",
  dimensions,
}) => {
  const [animatedProgress, setAnimatedProgress] = useState(0);

  const rawConf = finalResult.confidence <= 1.0 
    ? finalResult.confidence * 100 
    : finalResult.confidence;
  const confDisplay = rawConf.toFixed(1);
  const targetPercent = Math.min(100, Math.max(0, rawConf));

  useEffect(() => {
    // Smooth progress animation on mount
    const timer = setTimeout(() => {
      setAnimatedProgress(targetPercent);
    }, 60);
    return () => clearTimeout(timer);
  }, [targetPercent]);

  const getVerdictConfig = (verdict: VerdictType): VerdictConfig => {
    switch (verdict) {
      case "REAL_ORIGINAL":
        return {
          label: "REAL ORIGINAL",
          themeClass: "verdict-state-real",
          accentColor: "#059669",
          badgeBg: "rgba(5, 150, 105, 0.08)",
          icon: <ShieldCheckIcon size={26} />,
          explanation: "Classified as REAL ORIGINAL based on the current generation and manipulation signals.",
        };
      case "AI_GENERATED":
        return {
          label: "AI GENERATED",
          themeClass: "verdict-state-ai",
          accentColor: "#E11D48",
          badgeBg: "rgba(225, 29, 72, 0.08)",
          icon: <SparklesIcon size={26} />,
          explanation: "Classified as AI GENERATED based on the current generation and manipulation signals.",
        };
      case "AI_MANIPULATED":
        return {
          label: "AI MANIPULATED",
          themeClass: "verdict-state-manip",
          accentColor: "#6C63D9",
          badgeBg: "rgba(108, 99, 217, 0.08)",
          icon: <ImagePlusIcon size={26} />,
          explanation: "Classified as AI MANIPULATED based on the current generation and manipulation signals.",
        };
      case "UNCERTAIN":
      default:
        return {
          label: "UNCERTAIN",
          themeClass: "verdict-state-uncertain",
          accentColor: "#D97706",
          badgeBg: "rgba(217, 119, 6, 0.08)",
          icon: <TriangleAlertIcon size={26} />,
          explanation: "The available forensic signals do not provide a sufficiently clear classification.",
        };
    }
  };

  const config = getVerdictConfig(finalResult.label);

  // SVG Circular Confidence Orbit geometry
  const radius = 56;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedProgress / 100) * circumference;

  return (
    <div className={`evidence-passport-card ${config.themeClass}`}>
      {/* Passport Header Ribbon */}
      <div className="passport-header-ribbon">
        <div className="passport-title-group">
          <span className="passport-kicker font-mono">DIGITAL EVIDENCE PASSPORT</span>
          <span className="passport-seal-tag font-mono">AUTHENTICITY RECORD</span>
        </div>
        <div className="passport-id-badge font-mono">
          SEC-ID: #EV-{(rawConf * 13).toFixed(0).padStart(4, "0")}
        </div>
      </div>

      {/* Main Forensic Verdict Banner */}
      <div className="verdict-banner-container">
        <div className="verdict-tag-header">
          <span className="verdict-caption font-mono">FORENSIC VERDICT</span>
        </div>

        <div className="verdict-badge-row">
          <div 
            className="verdict-icon-container" 
            style={{ color: config.accentColor, backgroundColor: config.badgeBg }}
          >
            {config.icon}
          </div>
          <div className="verdict-title-wrap">
            <h2 className="verdict-primary-title" style={{ color: config.accentColor }}>
              {config.label}
            </h2>
          </div>
        </div>

        {/* Official Verdict Explanation */}
        <p className="verdict-formal-explanation">
          {finalResult.reason || config.explanation}
        </p>
      </div>

      {/* Confidence Orbit Section */}
      <div className="confidence-orbit-section">
        <div className="orbit-interactive-cluster">
          <div className="orbit-svg-wrapper">
            <svg className="confidence-orbit-svg" width="144" height="144" viewBox="0 0 144 144">
              {/* Outer Decorative Orbit Ring */}
              <circle
                className="orbit-outer-dash"
                cx="72"
                cy="72"
                r="68"
                strokeWidth="1"
                strokeDasharray="4 6"
              />
              {/* Background Track */}
              <circle
                className="orbit-bg-track"
                cx="72"
                cy="72"
                r={radius}
                strokeWidth="7"
              />
              {/* Foreground Animated Confidence Orbit Arc */}
              <circle
                className="orbit-active-arc"
                cx="72"
                cy="72"
                r={radius}
                strokeWidth="7"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                transform="rotate(-90 72 72)"
                style={{ stroke: config.accentColor }}
              />
            </svg>

            {/* Orbit Center Readout */}
            <div className="orbit-center-display">
              <span className="orbit-metric-value font-mono" style={{ color: config.accentColor }}>
                {confDisplay}%
              </span>
              <span className="orbit-metric-label font-mono">CONFIDENCE</span>
            </div>
          </div>

          <div className="orbit-clarification-text">
            <span className="orbit-heading font-mono">CONFIDENCE CALIBRATION</span>
            <p className="orbit-desc">
              Posterior confidence calibrated across whole-image synthesis and localized manipulation boundaries.
            </p>
          </div>
        </div>
      </div>

      {/* Evidence Passport Details Identity Section */}
      <div className="passport-identity-grid font-mono">
        <div className="passport-field-cell">
          <span className="field-label">EVIDENCE</span>
          <span className="field-value" title={filename}>
            {filename}
            {dimensions ? ` (${dimensions.width}×${dimensions.height})` : ""}
          </span>
        </div>

        <div className="passport-field-cell">
          <span className="field-label">STATUS</span>
          <span className="field-value text-semantic" style={{ color: config.accentColor }}>
            ● VERIFIED COMPLETE
          </span>
        </div>

        <div className="passport-field-cell">
          <span className="field-label">CONFIDENCE</span>
          <span className="field-value">{confDisplay}% CALIBRATED</span>
        </div>

        <div className="passport-field-cell">
          <span className="field-label">ANALYSIS</span>
          <span className="field-value">DUAL-STREAM SYNTHESIS</span>
        </div>
      </div>
    </div>
  );
};
