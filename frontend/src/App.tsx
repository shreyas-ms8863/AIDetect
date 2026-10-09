import { useState, useEffect, useRef } from "react";
import "./App.css";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { UploadScanner } from "./components/UploadScanner";
import { EvidenceCanvas } from "./components/EvidenceCanvas";
import { SignalConnectionLine } from "./components/SignalConnectionLine";
import { ForensicVerdict } from "./components/ForensicVerdict";
import { ForensicSignature } from "./components/ForensicSignature";
import { SignalDNA } from "./components/SignalDNA";
import { SignalMatrix } from "./components/SignalMatrix";
import { V5DCard } from "./components/V5DCard";
import { StressTest } from "./components/StressTest";
import { AnalyzeAnother } from "./components/AnalyzeAnother";
import { HistoryDrawer } from "./components/HistoryDrawer";
import { FocusModeModal } from "./components/FocusModeModal";
import { Footer } from "./components/Footer";
import { TriangleAlertIcon, XIcon } from "./components/Icons";
import { fetchSystemStatus, analyzeImageFile } from "./api";
import { createThumbnailDataUrl } from "./utils/imageThumbnail";
import type { 
  FullAnalysisResponse, 
  ImageDimensions, 
  SystemStatus, 
  HistoryItem 
} from "./types";

export function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageDimensions, setImageDimensions] = useState<ImageDimensions | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<FullAnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({ connected: false });

  // Evidence Archive (History) State
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [activeSpecimenId, setActiveSpecimenId] = useState<string | null>(null);
  const [specimenCounter, setSpecimenCounter] = useState(0);

  // Focus Mode State
  const [isFocusModeOpen, setIsFocusModeOpen] = useState(false);

  const fileInputHiddenRef = useRef<HTMLInputElement>(null);
  const evidenceSectionRef = useRef<HTMLDivElement>(null);

  // Poll backend engine status
  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      const status = await fetchSystemStatus();
      if (isMounted) setSystemStatus(status);
    };
    checkStatus();
    const interval = setInterval(checkStatus, 12000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleFileSelect = (file: File | undefined) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please select a valid image file (JPG, PNG, or WebP).");
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);

    const objectUrl = URL.createObjectURL(file);
    setSelectedImage(objectUrl);
    setAnalysisResult(null);
    setIsAnalyzing(false);
    setActiveSpecimenId(null);

    const img = new Image();
    img.onload = () => {
      setImageDimensions({
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };
    img.src = objectUrl;
  };

  const handleClear = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setImageDimensions(null);
    setAnalysisResult(null);
    setIsAnalyzing(false);
    setErrorMessage(null);
    setActiveSpecimenId(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setErrorMessage("Please select an image file to analyze.");
      return;
    }

    setErrorMessage(null);
    setIsAnalyzing(true);

    try {
      const result = await analyzeImageFile(selectedFile);
      setAnalysisResult(result);
      setIsAnalyzing(false);

      // Generate a persistent base64 thumbnail for session archive (immune to revocation)
      const thumbnailDataUrl = await createThumbnailDataUrl(selectedFile);

      // Increment sequential specimen counter
      const nextNum = specimenCounter + 1;
      setSpecimenCounter(nextNum);

      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const dateStr = now.toLocaleDateString([], { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase();

      const primaryPred = result.primary_verdict ?? result.v5_d?.prediction ?? result.prediction;
      const primaryConf = result.primary_confidence ?? result.v5_d?.confidence ?? (result.forensic?.final?.confidence ?? result.confidence);
      const normalizedConf = primaryConf <= 1.0 ? primaryConf * 100 : primaryConf;

      const newId = `specimen-${Date.now()}`;
      const newItem: HistoryItem = {
        id: newId,
        specimenNumber: nextNum,
        filename: selectedFile.name,
        file: selectedFile,
        imageSrc: selectedImage || thumbnailDataUrl,
        thumbnailDataUrl: thumbnailDataUrl || selectedImage || "",
        dimensions: imageDimensions,
        verdict: primaryPred === "AI-GENERATED" ? "AI_GENERATED" : "REAL_ORIGINAL",
        confidence: normalizedConf,
        timestamp: timeStr,
        dateStr: dateStr,
        result: result,
      };

      setHistoryItems((prev) => [newItem, ...prev.slice(0, 49)]);
      setActiveSpecimenId(newId);
    } catch (err) {
      console.error("Forensic scan failed:", err);
      const msg = err instanceof Error ? err.message : "Inference connection error.";
      setErrorMessage(`Forensic analysis could not be completed: ${msg}`);
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    handleClear();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleScrollToEvidence = () => {
    evidenceSectionRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Instant restoration of historical specimen into main interface
  const handleSelectHistory = (item: HistoryItem) => {
    // Generate fresh object URL if file is retained, or use persistent data URL
    let displayUrl = item.thumbnailDataUrl || item.imageSrc;
    if (item.file) {
      displayUrl = URL.createObjectURL(item.file);
    }

    setSelectedFile(item.file ?? null);
    setSelectedImage(displayUrl);
    setImageDimensions(item.dimensions);
    setAnalysisResult(item.result);
    setActiveSpecimenId(item.id);
    setIsHistoryOpen(false);

    // Scroll to evidence viewport
    setTimeout(() => {
      evidenceSectionRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 50);
  };

  const isAnalyzed = Boolean(analysisResult && analysisResult.forensic);

  // Compute active specimen number for display
  const currentSpecimenNumber = activeSpecimenId
    ? historyItems.find((h) => h.id === activeSpecimenId)?.specimenNumber ?? 1
    : specimenCounter + 1;

  return (
    <div className="digital-evidence-app">
      <Header
        status={systemStatus}
        onReset={handleReset}
        onEvidenceClick={handleScrollToEvidence}
        onHistoryClick={() => setIsHistoryOpen(true)}
        hasEvidence={Boolean(selectedImage)}
        historyCount={historyItems.length}
      />

      <main className="evidence-main-content">
        <div className="main-viewport-constrain">
          {/* Global Error Banner */}
          {errorMessage && (
            <div className="error-hud-alert" role="alert">
              <TriangleAlertIcon size={18} className="error-glyph" />
              <span className="error-text">{errorMessage}</span>
              <button
                type="button"
                className="error-close-btn"
                onClick={() => setErrorMessage(null)}
                aria-label="Dismiss error"
              >
                <XIcon size={15} />
              </button>
            </div>
          )}

          {/* Hidden File Input for Choose New file */}
          <input
            ref={fileInputHiddenRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => handleFileSelect(e.target.files?.[0])}
            style={{ display: "none" }}
          />

          {/* STAGE 1: EMPTY STATE -> HERO + EVIDENCE SCANNER */}
          {!selectedImage && (
            <div className="empty-state-wrapper">
              <Hero />
              <UploadScanner onFileSelect={handleFileSelect} />
            </div>
          )}

          {/* STAGE 2: SPECIMEN LOADED -> EVIDENCE PASSPORT BENTO WORKFLOW */}
          {selectedImage && (
            <div ref={evidenceSectionRef} className="evidence-investigation-flow">
              {/* PRIMARY BENTO ROW: HERO EVIDENCE CANVAS ──► SIGNAL CONNECTION ──► VERDICT PASSPORT */}
              <div className={`primary-evidence-stage ${isAnalyzed ? "stage-revealed" : "stage-staging"}`}>
                <div className="stage-evidence-side">
                  <EvidenceCanvas
                    imageSrc={selectedImage}
                    file={selectedFile}
                    dimensions={imageDimensions}
                    isAnalyzing={isAnalyzing}
                    isAnalyzed={isAnalyzed}
                    onAnalyze={handleAnalyze}
                    onClear={handleClear}
                    onChooseNew={() => fileInputHiddenRef.current?.click()}
                    onOpenFocusMode={() => setIsFocusModeOpen(true)}
                    historyItems={historyItems}
                    activeSpecimenId={activeSpecimenId}
                    specimenNumber={currentSpecimenNumber}
                    onSelectHistory={handleSelectHistory}
                  />
                </div>

                {isAnalyzed && analysisResult?.forensic && (
                  <>
                    <SignalConnectionLine />
                    <div className="stage-verdict-side">
                      <ForensicVerdict
                        finalResult={analysisResult.forensic.final}
                        v5_d={analysisResult.v5_d}
                        crossCheck={analysisResult.forensic_cross_check}
                        primaryVerdict={analysisResult.primary_verdict}
                        primaryConfidence={analysisResult.primary_confidence}
                        filename={selectedFile?.name}
                        dimensions={imageDimensions}
                      />
                    </div>
                  </>
                )}
              </div>

              {/* POST-ANALYSIS SECONDARY FORENSIC BENTO GRID */}
              {isAnalyzed && analysisResult?.forensic && (
                <div className="evidence-secondary-bento">
                  {/* Two-Column Middle Bento: Signal Profile (Waveform) & Signature (Fingerprint) */}
                  <div className="bento-two-col-row">
                    <SignalDNA
                      spatialProb={analysisResult.models?.spatial.ai_probability ?? 50}
                      frequencyProb={analysisResult.models?.frequency.ai_probability ?? 50}
                      fusionProb={analysisResult.forensic.final.confidence * 100}
                    />

                    <ForensicSignature
                      genAiProb={analysisResult.forensic.generation.probability_ai_generated}
                      manipProb={analysisResult.forensic.manipulation.probability_ai_manipulated}
                      fusionConf={analysisResult.forensic.final.confidence}
                    />
                  </div>

                  {/* V5-D Gated Residual Next-Gen Model Showcase (if returned by backend) */}
                  {analysisResult.v5_d && !analysisResult.v5_d.error && (
                    <V5DCard v5_d={analysisResult.v5_d} />
                  )}

                  {/* Full-Width Signal Matrix Bento */}
                  <SignalMatrix
                    models={analysisResult.models}
                    forensic={analysisResult.forensic}
                    v5_d={analysisResult.v5_d}
                  />

                  {/* Optional Robustness Stress Test (if returned by backend) */}
                  {analysisResult.robustness && (
                    <StressTest robustness={analysisResult.robustness} />
                  )}

                  {/* Bottom Dock: Analyze Another */}
                  <AnalyzeAnother onReset={handleReset} />
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      <Footer />

      {/* Evidence Archive Drawer Modal */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        items={historyItems}
        activeId={activeSpecimenId}
        onSelect={handleSelectHistory}
        onClear={() => {
          setHistoryItems([]);
          setActiveSpecimenId(null);
        }}
      />

      {/* Full Focus Mode Modal */}
      {selectedImage && (
        <FocusModeModal
          isOpen={isFocusModeOpen}
          onClose={() => setIsFocusModeOpen(false)}
          imageSrc={selectedImage}
          filename={selectedFile?.name}
          dimensions={imageDimensions}
        />
      )}
    </div>
  );
}

export default App;