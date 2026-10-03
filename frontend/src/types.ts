export type ForensicVerdict = 
  | "REAL_ORIGINAL" 
  | "AI_GENERATED" 
  | "AI_MANIPULATED" 
  | "UNCERTAIN";

export interface ModelPrediction {
  prediction: "AI-GENERATED" | "REAL";
  confidence: number;
  ai_probability: number;
  real_probability: number;
}

export interface GenerationDetectorResult {
  model_name: string;
  probability_real: number;
  probability_ai_generated: number;
  label: "REAL" | "AI_GENERATED";
  confidence: number;
}

export interface ManipulationDetectorResult {
  model_name: string;
  probability_original: number;
  probability_ai_manipulated: number;
  label: "ORIGINAL_REAL" | "AI_MANIPULATED";
  confidence: number;
}

export interface ForensicFinalResult {
  label: ForensicVerdict;
  confidence: number;
  fusion_confidence?: number;
  fusion_probabilities?: {
    REAL_ORIGINAL: number;
    AI_GENERATED: number;
    AI_MANIPULATED: number;
  };
  reason: string;
  decision_case: string;
  strategy: string;
}

export interface ForensicBlock {
  generation: GenerationDetectorResult;
  manipulation: ManipulationDetectorResult;
  final: ForensicFinalResult;
  elapsed_seconds?: number;
}

export interface FullAnalysisResponse {
  filename: string;
  prediction: "AI-GENERATED" | "REAL";
  confidence: number;
  ai_probability: number;
  real_probability: number;
  models?: {
    spatial: ModelPrediction;
    frequency: ModelPrediction;
    hybrid: ModelPrediction;
  };
  robustness?: {
    original: ModelPrediction;
    jpeg_compression: ModelPrediction;
    resize: ModelPrediction;
    blur: ModelPrediction;
    noise: ModelPrediction;
  };
  forensic?: ForensicBlock;
  final_label?: ForensicVerdict;
  final_confidence?: number;
  strategy?: string;
  decision_case?: string;
  message: string;
}

export interface ImageDimensions {
  width: number;
  height: number;
}

export interface SystemStatus {
  connected: boolean;
  device?: string;
  modelVersion?: string;
  defaultStrategy?: string;
}

export interface HistoryItem {
  id: string;
  specimenNumber: number;
  filename: string;
  file?: File | null;
  imageSrc: string;
  thumbnailDataUrl: string;
  dimensions: ImageDimensions | null;
  verdict: ForensicVerdict;
  confidence: number;
  timestamp: string;
  dateStr?: string;
  result: FullAnalysisResponse;
}
