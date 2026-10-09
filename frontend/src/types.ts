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

export interface ReliabilityMetrics {
  generation_disagreement: number;
  spatial_frequency_disagreement: number;
  hybrid_disagreement: number;
  domain_risk_score?: number;
  domain_risk?: "LOW" | "MEDIUM" | "HIGH" | string;
  domain_type?: "NATURAL_PHOTO" | "DOCUMENT_LIKE" | "SCREENSHOT_LIKE" | "UNKNOWN" | string;
  reliability_state: "RELIABLE" | "MODERATE" | "LOW" | string;
}

export interface DomainSignals {
  paper_canvas_fraction?: number;
  text_glyph_count?: number;
  rectilinear_line_energy?: number;
  monochromatic_fraction?: number;
  [key: string]: number | undefined;
}

export interface DomainAnalysis {
  domain_type: "NATURAL_PHOTO" | "DOCUMENT_LIKE" | "SCREENSHOT_LIKE" | "UNKNOWN" | string;
  domain_risk: "LOW" | "MEDIUM" | "HIGH" | string;
  domain_risk_score: number;
  domain_signals?: DomainSignals;
  summary?: string;
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
  decision_source?: string;
  consensus_state?: string;
  reliability_state?: "RELIABLE" | "MODERATE" | "LOW" | string;
  reliability_metrics?: ReliabilityMetrics;
  domain_type?: string;
  domain_risk?: "LOW" | "MEDIUM" | "HIGH" | string;
  domain_risk_score?: number;
  reason: string;
  decision_case: string;
  strategy: string;
}

export interface ForensicBlock {
  generation: GenerationDetectorResult;
  manipulation: ManipulationDetectorResult;
  final: ForensicFinalResult;
  domain?: DomainAnalysis;
  elapsed_seconds?: number;
}

export interface V5DGateWeights {
  spatial: number;
  frequency: number;
  residual: number;
}

export interface V5DResult {
  prediction: "AI-GENERATED" | "REAL";
  confidence: number;
  ai_probability: number;
  real_probability: number;
  raw_ai_probability?: number;
  raw_real_probability?: number;
  calibrated_ai_probability?: number;
  calibrated_real_probability?: number;
  calibration_method?: string;
  model_name?: string;
  gate_weights?: V5DGateWeights;
  error?: string;
}

export interface ForensicCrossCheck {
  status: "CONSISTENT" | "CONFLICTING" | "LOW_RELIABILITY" | string;
  strategy_e_verdict?: string;
  strategy_e_confidence?: number;
  strategy_e_reason?: string;
  explanation: string;
}

export interface FullAnalysisResponse {
  filename: string;
  prediction: "AI-GENERATED" | "REAL";
  confidence: number;
  ai_probability: number;
  real_probability: number;
  primary_verdict?: "AI-GENERATED" | "REAL" | string;
  primary_confidence?: number;
  forensic_cross_check?: ForensicCrossCheck;
  models?: {
    spatial: ModelPrediction;
    frequency: ModelPrediction;
    hybrid: ModelPrediction;
  };
  v5_d?: V5DResult;
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
