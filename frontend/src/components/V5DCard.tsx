import type { FC } from "react";
import type { V5DResult } from "../types";

interface V5DCardProps {
  v5_d: V5DResult;
}

export const V5DCard: FC<V5DCardProps> = ({ v5_d }) => {
  const isAi = v5_d.prediction === "AI-GENERATED";
  const rawAi = v5_d.raw_ai_probability ?? v5_d.ai_probability;
  const rawReal = v5_d.raw_real_probability ?? v5_d.real_probability;
  const calAi = v5_d.calibrated_ai_probability ?? v5_d.ai_probability;
  const calReal = v5_d.calibrated_real_probability ?? v5_d.real_probability;
  const isCalibrated = v5_d.calibrated_ai_probability !== undefined && v5_d.calibration_method !== "Raw_Softmax_Uncalibrated" && v5_d.calibration_method !== "Uncalibrated (Fit Pending)";

  return (
    <div className="bento-card v5d-card" style={{ borderLeft: isAi ? "4px solid var(--color-ai)" : "4px solid var(--color-real)" }}>
      {/* Header bar */}
      <div className="card-header-bar">
        <div className="card-title-cluster">
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span className="card-category-tag font-mono">NEXT-GEN DETECTOR</span>
            <span
              className="card-badge font-mono"
              style={{
                backgroundColor: isAi ? "rgba(225, 29, 72, 0.1)" : "rgba(16, 185, 129, 0.1)",
                color: isAi ? "var(--color-ai)" : "var(--color-real)",
                fontWeight: 700,
                border: "none",
              }}
            >
              V5-D
            </span>
            {isCalibrated && (
              <span
                className="card-badge font-mono"
                style={{
                  backgroundColor: "rgba(16, 185, 129, 0.12)",
                  color: "#059669",
                  fontWeight: 700,
                  fontSize: "8.5px",
                  border: "none",
                  letterSpacing: "0.03em",
                }}
              >
                POST-HOC CALIBRATED
              </span>
            )}
          </div>
          <span className="card-subtitle-tag font-mono">
            {v5_d.model_name || "V5-D Gated Residual Calibrated"} • TRIPLE-STREAM FUSION
          </span>
        </div>

        <div style={{ textAlign: "right" }}>
          <span
            className="font-mono"
            style={{
              fontSize: "13px",
              fontWeight: 800,
              color: isAi ? "var(--color-ai)" : "var(--color-real)",
              letterSpacing: "0.04em",
            }}
          >
            {v5_d.prediction}
          </span>
          <div className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
            CONFIDENCE: {v5_d.confidence.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Raw vs Calibrated Summary Pill */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "6px 12px",
          backgroundColor: "rgba(0, 0, 0, 0.025)",
          borderRadius: "6px",
          marginBottom: "12px",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <div className="font-mono" style={{ fontSize: "10px", color: "var(--text-muted)" }}>
          V5-D RAW: <span style={{ fontWeight: 700, color: "var(--text-primary)" }}>{rawAi.toFixed(1)}%</span>
        </div>
        <div className="font-mono" style={{ fontSize: "10px", color: isAi ? "var(--color-ai)" : "var(--color-real)" }}>
          V5-D CALIBRATED: <span style={{ fontWeight: 800 }}>{calAi.toFixed(1)}%</span>
        </div>
      </div>

      {/* Probability Bars (Calibrated with Raw Reference) */}
      <div className="domain-channels-grid" style={{ marginBottom: v5_d.gate_weights ? "16px" : "0" }}>
        <div className="channel-bar-item">
          <div className="channel-desc-row font-mono">
            <span className="channel-name">REAL AUTHENTIC</span>
            <span className="channel-percent text-emerald">
              {calReal.toFixed(1)}%
              {isCalibrated && (
                <span style={{ fontSize: "8.5px", color: "var(--text-muted)", marginLeft: "4px" }}>
                  (raw: {rawReal.toFixed(1)}%)
                </span>
              )}
            </span>
          </div>
          <div className="channel-meter-track">
            <div
              className="channel-meter-fill fill-emerald"
              style={{ width: `${Math.min(100, Math.max(0, calReal))}%` }}
            />
          </div>
        </div>

        <div className="channel-bar-item">
          <div className="channel-desc-row font-mono">
            <span className="channel-name">AI SYNTHETIC</span>
            <span className="channel-percent text-rose">
              {calAi.toFixed(1)}%
              {isCalibrated && (
                <span style={{ fontSize: "8.5px", color: "var(--text-muted)", marginLeft: "4px" }}>
                  (raw: {rawAi.toFixed(1)}%)
                </span>
              )}
            </span>
          </div>
          <div className="channel-meter-track">
            <div
              className="channel-meter-fill fill-rose"
              style={{ width: `${Math.min(100, Math.max(0, calAi))}%` }}
            />
          </div>
        </div>
      </div>

      {/* Method Note */}
      {isCalibrated && (
        <div
          className="font-mono"
          style={{
            fontSize: "8.5px",
            color: "var(--text-muted)",
            marginBottom: v5_d.gate_weights ? "14px" : "0",
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span>Calibrated probability ({v5_d.calibration_method || "Platt Scaling"})</span>
          <span>Optimization: Min NLL on validation partition</span>
        </div>
      )}

      {/* Adaptive Gating Distribution (Learned Feature Weights) */}
      {v5_d.gate_weights && (
        <div
          style={{
            paddingTop: "12px",
            borderTop: "1px dashed var(--border-subtle)",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
          }}
        >
          <div className="domain-label-bar">
            <span className="domain-title font-mono" style={{ fontSize: "9.5px" }}>
              ADAPTIVE GATING ENCODER WEIGHTS
            </span>
            <span className="domain-status font-mono" style={{ fontSize: "8.5px" }}>
              SPATIAL + FFT + NOISE RESIDUAL
            </span>
          </div>

          <div
            className="domain-channels-grid"
            style={{ gridTemplateColumns: "repeat(3, 1fr)", gap: "10px" }}
          >
            {/* Spatial Gate */}
            <div className="channel-bar-item">
              <div className="channel-desc-row font-mono" style={{ fontSize: "9px" }}>
                <span className="channel-name">SPATIAL</span>
                <span style={{ color: "#0284c7", fontWeight: 700 }}>
                  {(v5_d.gate_weights.spatial * 100).toFixed(1)}%
                </span>
              </div>
              <div className="channel-meter-track" style={{ height: "6px" }}>
                <div
                  className="channel-meter-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, v5_d.gate_weights.spatial * 100))}%`,
                    backgroundColor: "#0284c7",
                  }}
                />
              </div>
            </div>

            {/* Frequency Gate */}
            <div className="channel-bar-item">
              <div className="channel-desc-row font-mono" style={{ fontSize: "9px" }}>
                <span className="channel-name">FREQUENCY</span>
                <span style={{ color: "#7c3aed", fontWeight: 700 }}>
                  {(v5_d.gate_weights.frequency * 100).toFixed(1)}%
                </span>
              </div>
              <div className="channel-meter-track" style={{ height: "6px" }}>
                <div
                  className="channel-meter-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, v5_d.gate_weights.frequency * 100))}%`,
                    backgroundColor: "#7c3aed",
                  }}
                />
              </div>
            </div>

            {/* Residual Gate */}
            <div className="channel-bar-item">
              <div className="channel-desc-row font-mono" style={{ fontSize: "9px" }}>
                <span className="channel-name">NOISE RESIDUAL</span>
                <span style={{ color: "#f59e0b", fontWeight: 700 }}>
                  {(v5_d.gate_weights.residual * 100).toFixed(1)}%
                </span>
              </div>
              <div className="channel-meter-track" style={{ height: "6px" }}>
                <div
                  className="channel-meter-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, v5_d.gate_weights.residual * 100))}%`,
                    backgroundColor: "#f59e0b",
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
