import { useState } from "react";
import type { FC } from "react";
import type { ModelPrediction, ForensicBlock, V5DResult } from "../types";

interface SignalMatrixProps {
  models?: {
    spatial: ModelPrediction;
    frequency: ModelPrediction;
    hybrid: ModelPrediction;
  };
  forensic?: ForensicBlock;
  v5_d?: V5DResult;
}

export type MatrixTab = "final" | "v5_d" | "spatial" | "frequency" | "hybrid";

export const SignalMatrix: FC<SignalMatrixProps> = ({ models, forensic, v5_d }) => {
  const [activeTab, setActiveTab] = useState<MatrixTab>("final");

  // Primary forensic signals from Strategy E / Production Engine
  const genReal = forensic 
    ? (forensic.generation.probability_real * 100) 
    : models ? models.frequency.real_probability : 50;
  const genAi = forensic 
    ? (forensic.generation.probability_ai_generated * 100) 
    : models ? models.frequency.ai_probability : 50;

  const manipOriginal = forensic 
    ? (forensic.manipulation.probability_original * 100) 
    : 80;
  const manipAi = forensic 
    ? (forensic.manipulation.probability_ai_manipulated * 100) 
    : 20;

  // Selected specific model prediction if tabs are switched
  const selectedModel = activeTab === "spatial" 
    ? { name: "Spatial Signal", data: models?.spatial }
    : activeTab === "frequency" 
    ? { name: "Frequency Signal", data: models?.frequency }
    : activeTab === "hybrid" 
    ? { name: "Hybrid Signal", data: models?.hybrid }
    : null;

  return (
    <div className="bento-card signal-matrix-card">
      <div className="card-header-bar">
        <div className="card-title-cluster">
          <span className="card-category-tag font-mono">SIGNAL MATRIX</span>
          <span className="card-subtitle-tag font-mono">MULTI-DETECTOR FORENSIC METRICS</span>
        </div>

        {/* Scientific Segmented Control: FINAL / SPATIAL / FREQUENCY / HYBRID */}
        <div className="segmented-pill-control" role="tablist" aria-label="Signal matrix channels">
          <button
            type="button"
            className={`pill-tab ${activeTab === "final" ? "tab-active" : ""}`}
            onClick={() => setActiveTab("final")}
            role="tab"
            aria-selected={activeTab === "final"}
          >
            FINAL
          </button>

          {v5_d && (
            <button
              type="button"
              className={`pill-tab ${activeTab === "v5_d" ? "tab-active" : ""}`}
              onClick={() => setActiveTab("v5_d")}
              role="tab"
              aria-selected={activeTab === "v5_d"}
            >
              V5-D GATED
            </button>
          )}
          
          {models && (
            <>
              <button
                type="button"
                className={`pill-tab ${activeTab === "spatial" ? "tab-active" : ""}`}
                onClick={() => setActiveTab("spatial")}
                role="tab"
                aria-selected={activeTab === "spatial"}
              >
                SPATIAL
              </button>
              <button
                type="button"
                className={`pill-tab ${activeTab === "frequency" ? "tab-active" : ""}`}
                onClick={() => setActiveTab("frequency")}
                role="tab"
                aria-selected={activeTab === "frequency"}
              >
                FREQUENCY
              </button>
              <button
                type="button"
                className={`pill-tab ${activeTab === "hybrid" ? "tab-active" : ""}`}
                onClick={() => setActiveTab("hybrid")}
                role="tab"
                aria-selected={activeTab === "hybrid"}
              >
                HYBRID
              </button>
            </>
          )}
        </div>
      </div>

      {/* VIEW 1: FINAL (Primary Generation & Manipulation Dual-Stream Signals) */}
      {activeTab === "final" && (
        <div className="matrix-sections-container">
          {/* Generation Signal Section */}
          <div className="matrix-domain-group">
            <div className="domain-label-bar">
              <span className="domain-title font-mono">SYNTHETIC GENERATION SIGNALS</span>
              <span className="domain-status font-mono">WHOLE-IMAGE GLOBAL COHERENCE</span>
            </div>

            <div className="domain-channels-grid">
              <div className="channel-bar-item">
                <div className="channel-desc-row font-mono">
                  <span className="channel-name">REAL AUTHENTIC</span>
                  <span className="channel-percent text-emerald">{genReal.toFixed(1)}%</span>
                </div>
                <div className="channel-meter-track">
                  <div
                    className="channel-meter-fill fill-emerald"
                    style={{ width: `${Math.min(100, Math.max(0, genReal))}%` }}
                  />
                </div>
              </div>

              <div className="channel-bar-item">
                <div className="channel-desc-row font-mono">
                  <span className="channel-name">AI GENERATED</span>
                  <span className="channel-percent text-rose">{genAi.toFixed(1)}%</span>
                </div>
                <div className="channel-meter-track">
                  <div
                    className="channel-meter-fill fill-rose"
                    style={{ width: `${Math.min(100, Math.max(0, genAi))}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Manipulation Signal Section */}
          <div className="matrix-domain-group">
            <div className="domain-label-bar">
              <span className="domain-title font-mono">MANIPULATION & INPAINTING SIGNALS</span>
              <span className="domain-status font-mono">LOCALIZED SPECTRAL DISCONTINUITY</span>
            </div>

            <div className="domain-channels-grid">
              <div className="channel-bar-item">
                <div className="channel-desc-row font-mono">
                  <span className="channel-name">ORIGINAL REAL</span>
                  <span className="channel-percent text-emerald">{manipOriginal.toFixed(1)}%</span>
                </div>
                <div className="channel-meter-track">
                  <div
                    className="channel-meter-fill fill-emerald"
                    style={{ width: `${Math.min(100, Math.max(0, manipOriginal))}%` }}
                  />
                </div>
              </div>

              <div className="channel-bar-item">
                <div className="channel-desc-row font-mono">
                  <span className="channel-name">AI MANIPULATED</span>
                  <span className="channel-percent text-violet">{manipAi.toFixed(1)}%</span>
                </div>
                <div className="channel-meter-track">
                  <div
                    className="channel-meter-fill fill-violet"
                    style={{ width: `${Math.min(100, Math.max(0, manipAi))}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW: V5-D GATED RESIDUAL MODEL BREAKDOWN */}
      {activeTab === "v5_d" && v5_d && (
        <div className="matrix-single-model-view">
          <div className="model-view-header font-mono">
            <span className="model-view-title">{v5_d.model_name || "V5-D GATED RESIDUAL"} EVALUATION</span>
            <span className="model-view-confidence">
              CONFIDENCE: {v5_d.confidence.toFixed(1)}% ({v5_d.prediction})
            </span>
          </div>

          <div className="domain-channels-grid">
            <div className="channel-bar-item">
              <div className="channel-desc-row font-mono">
                <span className="channel-name">AUTHENTIC REAL LIKELIHOOD</span>
                <span className="channel-percent text-emerald">
                  {v5_d.real_probability.toFixed(1)}%
                </span>
              </div>
              <div className="channel-meter-track">
                <div
                  className="channel-meter-fill fill-emerald"
                  style={{ width: `${Math.min(100, Math.max(0, v5_d.real_probability))}%` }}
                />
              </div>
            </div>

            <div className="channel-bar-item">
              <div className="channel-desc-row font-mono">
                <span className="channel-name">AI SYNTHETIC LIKELIHOOD</span>
                <span className="channel-percent text-rose">
                  {v5_d.ai_probability.toFixed(1)}%
                </span>
              </div>
              <div className="channel-meter-track">
                <div
                  className="channel-meter-fill fill-rose"
                  style={{ width: `${Math.min(100, Math.max(0, v5_d.ai_probability))}%` }}
                />
              </div>
            </div>
          </div>

          {v5_d.gate_weights && (
            <div className="matrix-domain-group" style={{ marginTop: "14px", paddingTop: "14px", borderTop: "1px dashed var(--border-subtle)" }}>
              <div className="domain-label-bar">
                <span className="domain-title font-mono">ADAPTIVE FUSION GATE WEIGHTS</span>
                <span className="domain-status font-mono">3-STREAM FEATURE ATTENTION</span>
              </div>
              <div className="domain-channels-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                <div className="channel-bar-item">
                  <div className="channel-desc-row font-mono">
                    <span className="channel-name">SPATIAL</span>
                    <span className="channel-percent" style={{ color: "#0284c7" }}>
                      {(v5_d.gate_weights.spatial * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="channel-meter-track">
                    <div
                      className="channel-meter-fill"
                      style={{ width: `${Math.min(100, Math.max(0, v5_d.gate_weights.spatial * 100))}%`, backgroundColor: "#0284c7" }}
                    />
                  </div>
                </div>

                <div className="channel-bar-item">
                  <div className="channel-desc-row font-mono">
                    <span className="channel-name">FREQUENCY</span>
                    <span className="channel-percent" style={{ color: "#7c3aed" }}>
                      {(v5_d.gate_weights.frequency * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="channel-meter-track">
                    <div
                      className="channel-meter-fill"
                      style={{ width: `${Math.min(100, Math.max(0, v5_d.gate_weights.frequency * 100))}%`, backgroundColor: "#7c3aed" }}
                    />
                  </div>
                </div>

                <div className="channel-bar-item">
                  <div className="channel-desc-row font-mono">
                    <span className="channel-name">RESIDUAL</span>
                    <span className="channel-percent" style={{ color: "#f59e0b" }}>
                      {(v5_d.gate_weights.residual * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="channel-meter-track">
                    <div
                      className="channel-meter-fill"
                      style={{ width: `${Math.min(100, Math.max(0, v5_d.gate_weights.residual * 100))}%`, backgroundColor: "#f59e0b" }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* VIEW 2: INDIVIDUAL MODEL BREAKDOWN (SPATIAL / FREQUENCY / HYBRID) */}
      {selectedModel && selectedModel.data && (
        <div className="matrix-single-model-view">
          <div className="model-view-header font-mono">
            <span className="model-view-title">{selectedModel.name.toUpperCase()} EVALUATION</span>
            <span className="model-view-confidence">
              CONFIDENCE: {selectedModel.data.confidence.toFixed(1)}% ({selectedModel.data.prediction})
            </span>
          </div>

          <div className="domain-channels-grid">
            <div className="channel-bar-item">
              <div className="channel-desc-row font-mono">
                <span className="channel-name">AUTHENTIC REAL LIKELIHOOD</span>
                <span className="channel-percent text-emerald">
                  {selectedModel.data.real_probability.toFixed(1)}%
                </span>
              </div>
              <div className="channel-meter-track">
                <div
                  className="channel-meter-fill fill-emerald"
                  style={{ width: `${Math.min(100, Math.max(0, selectedModel.data.real_probability))}%` }}
                />
              </div>
            </div>

            <div className="channel-bar-item">
              <div className="channel-desc-row font-mono">
                <span className="channel-name">AI SYNTHETIC LIKELIHOOD</span>
                <span className="channel-percent text-rose">
                  {selectedModel.data.ai_probability.toFixed(1)}%
                </span>
              </div>
              <div className="channel-meter-track">
                <div
                  className="channel-meter-fill fill-rose"
                  style={{ width: `${Math.min(100, Math.max(0, selectedModel.data.ai_probability))}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
