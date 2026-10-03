import { useState } from "react";
import type { FC } from "react";
import type { ModelPrediction } from "../types";

interface StressTestProps {
  robustness: {
    original: ModelPrediction;
    jpeg_compression: ModelPrediction;
    resize: ModelPrediction;
    blur: ModelPrediction;
    noise: ModelPrediction;
  };
}

interface StressVector {
  id: string;
  name: string;
  subname: string;
  stepNumber: string;
  prediction: "AI-GENERATED" | "REAL";
  confidence: number;
}

export const StressTest: FC<StressTestProps> = ({ robustness }) => {
  const [activeVectorId, setActiveVectorId] = useState<string>("jpeg");

  // Defensive extraction to support both flat ModelPrediction and nested hybrid dict
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const extractVectorData = (val: any): { prediction: "AI-GENERATED" | "REAL"; confidence: number } => {
    if (!val) return { prediction: "REAL", confidence: 50 };
    if (val.hybrid && val.hybrid.prediction) {
      return {
        prediction: val.hybrid.prediction,
        confidence: typeof val.hybrid.confidence === "number" ? val.hybrid.confidence : 50,
      };
    }
    return {
      prediction: val.prediction || "REAL",
      confidence: typeof val.confidence === "number" ? val.confidence : 50,
    };
  };

  const origData = extractVectorData(robustness.original);
  const jpegData = extractVectorData(robustness.jpeg_compression);
  const resizeData = extractVectorData(robustness.resize);
  const blurData = extractVectorData(robustness.blur);
  const noiseData = extractVectorData(robustness.noise);

  const vectors: StressVector[] = [
    {
      id: "original",
      stepNumber: "01",
      name: "BASELINE",
      subname: "Native Unaltered Specimen",
      prediction: origData.prediction,
      confidence: origData.confidence,
    },
    {
      id: "jpeg",
      stepNumber: "02",
      name: "JPEG COMPRESSION",
      subname: "Quality 35 DCT Quantization",
      prediction: jpegData.prediction,
      confidence: jpegData.confidence,
    },
    {
      id: "resize",
      stepNumber: "03",
      name: "DOWNSAMPLING",
      subname: "50% Bilinear Scale Reduction",
      prediction: resizeData.prediction,
      confidence: resizeData.confidence,
    },
    {
      id: "blur",
      stepNumber: "04",
      name: "GAUSSIAN BLUR",
      subname: "Spatial Convolution σ=2",
      prediction: blurData.prediction,
      confidence: blurData.confidence,
    },
    {
      id: "noise",
      stepNumber: "05",
      name: "ADDITIVE NOISE",
      subname: "Gaussian Perturbation σ=12",
      prediction: noiseData.prediction,
      confidence: noiseData.confidence,
    },
  ];

  const activeVector = vectors.find((v) => v.id === activeVectorId) || vectors[1];

  return (
    <div className="bento-card forensic-timeline-card">
      <div className="card-header-bar">
        <div className="card-title-cluster">
          <span className="card-category-tag font-mono">FORENSIC TIMELINE // ROBUSTNESS</span>
          <span className="card-subtitle-tag font-mono">SIGNAL STABILITY UNDER TRANSFORMATION</span>
        </div>
        <div className="card-badge font-mono">5 EVALUATION STATIONS</div>
      </div>

      {/* Connected Horizontal Timeline Rail */}
      <div className="timeline-rail-wrapper">
        <div className="timeline-rail-wire" aria-hidden="true" />

        <div className="timeline-stations-row font-mono">
          {vectors.map((vec) => {
            const isReal = vec.prediction === "REAL";
            const isSelected = vec.id === activeVectorId;

            return (
              <div
                key={vec.id}
                className={`timeline-station-node ${isSelected ? "station-selected" : ""}`}
                onClick={() => setActiveVectorId(vec.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && setActiveVectorId(vec.id)}
              >
                {/* Station Node Marker */}
                <div className={`station-pip ${isReal ? "pip-real" : "pip-ai"}`}>
                  <span className="pip-center-dot" />
                </div>

                <div className="station-meta-block">
                  <span className="station-step">{vec.stepNumber}</span>
                  <span className="station-name">{vec.name}</span>
                  <span className={`station-pill ${isReal ? "tag-real" : "tag-ai"}`}>
                    {vec.prediction}
                  </span>
                  <span className="station-conf">{vec.confidence.toFixed(1)}%</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Selected Vector Inspection Strip */}
      <div className="timeline-active-inspection-strip font-mono">
        <div className="inspection-left">
          <span className="inspection-tag">STATION {activeVector.stepNumber} // {activeVector.name}:</span>
          <span className="inspection-sub">{activeVector.subname}</span>
        </div>

        <div className="inspection-right">
          <span className="inspection-verdict-label">POST-TRANSFORMATION VERDICT:</span>
          <span className={`inspection-verdict-pill ${activeVector.prediction === "REAL" ? "tag-real" : "tag-ai"}`}>
            {activeVector.prediction} ({activeVector.confidence.toFixed(1)}% CONFIDENCE)
          </span>
        </div>
      </div>
    </div>
  );
};
