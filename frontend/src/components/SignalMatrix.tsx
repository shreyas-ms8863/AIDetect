import { useState } from "react";
import type { FC } from "react";
import type { ModelPrediction, ForensicBlock } from "../types";

interface SignalMatrixProps {
  models?: {
    spatial: ModelPrediction;
    frequency: ModelPrediction;
    hybrid: ModelPrediction;
  };
  forensic?: ForensicBlock;
}

export type MatrixTab = "final" | "spatial" | "frequency" | "hybrid";

export const SignalMatrix: FC<SignalMatrixProps> = ({ models, forensic }) => {
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
