import type { FullAnalysisResponse, SystemStatus, ForensicBlock } from "./types";

const BACKEND_URL = "http://127.0.0.1:8000";

/**
 * Fetch server status from GET /
 */
export async function fetchSystemStatus(): Promise<SystemStatus> {
  try {
    const res = await fetch(`${BACKEND_URL}/`, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return {
      connected: true,
      device: data.device,
      modelVersion: data.model_version,
      defaultStrategy: data.forensic_pipeline?.default_strategy || "v2_calibrated",
    };
  } catch (err) {
    console.warn("Backend connectivity check failed:", err);
    return { connected: false };
  }
}

/**
 * Perform complete forensic analysis on an uploaded image file.
 * Hits POST /analyze which returns legacy multi-model/robustness metrics
 * AND the additive authoritative Strategy E forensic decision block.
 */
export async function analyzeImageFile(
  file: File,
  strategy?: string
): Promise<FullAnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const endpoint = strategy 
    ? `${BACKEND_URL}/analyze?strategy=${encodeURIComponent(strategy)}`
    : `${BACKEND_URL}/analyze`;

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Server returned status HTTP ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errorDetail);
  }

  const data: FullAnalysisResponse = await response.json();

  // If forensic block is present (Phase 14/15 Production), ensure consistency
  if (!data.forensic && data.prediction) {
    // Fallback normalization in case legacy format is returned
    const pReal = data.real_probability / 100;
    const pAi = data.ai_probability / 100;
    const fallbackForensic: ForensicBlock = {
      generation: {
        model_name: "V4_Frequency_Generation_Detector",
        probability_real: pReal,
        probability_ai_generated: pAi,
        label: pAi >= 0.5 ? "AI_GENERATED" : "REAL",
        confidence: Math.max(pReal, pAi),
      },
      manipulation: {
        model_name: "Phase7_Frequency_Manipulation_Detector",
        probability_original: pReal,
        probability_ai_manipulated: pAi,
        label: pAi >= 0.5 ? "AI_MANIPULATED" : "ORIGINAL_REAL",
        confidence: Math.max(pReal, pAi),
      },
      final: {
        label: data.prediction === "REAL" ? "REAL_ORIGINAL" : "AI_GENERATED",
        confidence: data.confidence / 100,
        reason: `Legacy model prediction: ${data.prediction} with ${(data.confidence).toFixed(1)}% confidence.`,
        decision_case: "LEGACY_FALLBACK",
        strategy: "v1_baseline",
      },
    };
    data.forensic = fallbackForensic;
  }

  return data;
}
