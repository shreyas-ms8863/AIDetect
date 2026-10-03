import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import type { FC, ReactNode } from "react";
import { 
  XIcon, 
  SearchIcon, 
  ArrowRightIcon, 
  EvidenceFileIcon, 
  ArchiveXIcon,
  ShieldCheckIcon,
  SparklesIcon,
  ImagePlusIcon,
  TriangleAlertIcon,
  CrosshairIcon
} from "./Icons";
import type { HistoryItem, ForensicVerdict } from "../types";

interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  items: HistoryItem[];
  activeId?: string | null;
  onSelect: (item: HistoryItem) => void;
  onClear: () => void;
}

type VerdictFilter = "ALL" | "REAL" | "AI_GENERATED" | "AI_MANIPULATED" | "UNCERTAIN";

interface VerdictTheme {
  label: string;
  dotColor: string;
  textColor: string;
  pillBg: string;
  barColor: string;
  icon: ReactNode;
}

export const HistoryDrawer: FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  items,
  activeId,
  onSelect,
  onClear,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFilter, setSelectedFilter] = useState<VerdictFilter>("ALL");
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [imageErrors, setImageErrors] = useState<Record<string, boolean>>({});

  const searchInputRef = useRef<HTMLInputElement>(null);

  const handleClose = useCallback(() => {
    setShowClearConfirm(false);
    setSearchQuery("");
    setSelectedFilter("ALL");
    onClose();
  }, [onClose]);

  // Close on ESC key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (showClearConfirm) {
          setShowClearConfirm(false);
        } else {
          handleClose();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, showClearConfirm, handleClose]);

  // Color & badge configurations strictly adhering to Section 3
  const getVerdictTheme = (verdict: ForensicVerdict): VerdictTheme => {
    switch (verdict) {
      case "REAL_ORIGINAL":
        return {
          label: "REAL ORIGINAL",
          dotColor: "#059669",
          textColor: "#059669",
          pillBg: "rgba(5, 150, 105, 0.08)",
          barColor: "#059669",
          icon: <ShieldCheckIcon size={13} />,
        };
      case "AI_GENERATED":
        return {
          label: "AI GENERATED",
          dotColor: "#E11D48",
          textColor: "#E11D48",
          pillBg: "rgba(225, 29, 72, 0.08)",
          barColor: "#E11D48",
          icon: <SparklesIcon size={13} />,
        };
      case "AI_MANIPULATED":
        return {
          label: "AI MANIPULATED",
          dotColor: "#6C63D9",
          textColor: "#6C63D9",
          pillBg: "rgba(108, 99, 217, 0.08)",
          barColor: "#6C63D9",
          icon: <ImagePlusIcon size={13} />,
        };
      case "UNCERTAIN":
      default:
        return {
          label: "UNCERTAIN",
          dotColor: "#D97706",
          textColor: "#D97706",
          pillBg: "rgba(217, 119, 6, 0.08)",
          barColor: "#D97706",
          icon: <TriangleAlertIcon size={13} />,
        };
    }
  };

  // Filtered items based on search and verdict tab
  const filteredItems = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    return items.filter((item) => {
      // Verdict filter
      if (selectedFilter === "REAL" && item.verdict !== "REAL_ORIGINAL") return false;
      if (selectedFilter === "AI_GENERATED" && item.verdict !== "AI_GENERATED") return false;
      if (selectedFilter === "AI_MANIPULATED" && item.verdict !== "AI_MANIPULATED") return false;
      if (selectedFilter === "UNCERTAIN" && item.verdict !== "UNCERTAIN") return false;

      // Search query (filename, specimen number, or verdict label)
      if (!query) return true;
      const numStr = `#${String(item.specimenNumber).padStart(3, "0")}`.toLowerCase();
      const rawNum = String(item.specimenNumber);
      const filenameMatch = item.filename.toLowerCase().includes(query);
      const numMatch = numStr.includes(query) || rawNum.includes(query) || `specimen #${rawNum}`.includes(query);
      const verdictMatch = item.verdict.toLowerCase().replace("_", " ").includes(query);

      return filenameMatch || numMatch || verdictMatch;
    });
  }, [items, searchQuery, selectedFilter]);

  // Specimen count label
  const countLabel = useMemo(() => {
    const count = items.length;
    if (count === 0) return "0 SPECIMENS THIS SESSION";
    if (count === 1) return "1 SPECIMEN THIS SESSION";
    return `${count} SPECIMENS THIS SESSION`;
  }, [items.length]);

  const handleImageError = (id: string) => {
    setImageErrors((prev) => ({ ...prev, [id]: true }));
  };

  const handleConfirmClear = () => {
    onClear();
    setShowClearConfirm(false);
  };

  if (!isOpen) return null;

  return (
    <div 
      className="evidence-archive-backdrop" 
      onClick={handleClose} 
      role="dialog" 
      aria-modal="true"
      aria-label="Evidence Archive Drawer"
    >
      <div 
        className="evidence-archive-drawer" 
        onClick={(e) => e.stopPropagation()}
      >
        {/* ==================================================== */}
        {/* 1. DRAWER HEADER */}
        {/* ==================================================== */}
        <header className="archive-drawer-header">
          <div className="archive-title-cluster">
            <div className="archive-badge-kicker font-mono">
              EVIDENCE ARCHIVE
            </div>
            <div className="archive-session-count font-mono">
              {countLabel}
            </div>
            <p className="archive-subtitle">
              Specimens analyzed during this session
            </p>
          </div>

          <button
            type="button"
            className="archive-close-btn"
            onClick={handleClose}
            aria-label="Close Evidence Archive (Esc)"
            title="Close Evidence Archive (Esc)"
          >
            <XIcon size={16} />
          </button>
        </header>

        {/* ==================================================== */}
        {/* 2. SEARCH & FILTER CONTROLS (when items exist) */}
        {/* ==================================================== */}
        {items.length > 0 && (
          <div className="archive-controls-bar">
            {/* Search Input */}
            <div className="archive-search-wrap">
              <SearchIcon size={14} className="search-glyph" />
              <input
                ref={searchInputRef}
                type="text"
                className="archive-search-input font-mono"
                placeholder="Search evidence..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Search evidence by filename, specimen, or verdict"
              />
              {searchQuery && (
                <button
                  type="button"
                  className="search-clear-btn"
                  onClick={() => setSearchQuery("")}
                  aria-label="Clear search query"
                >
                  <XIcon size={13} />
                </button>
              )}
            </div>

            {/* Verdict Filter Pills */}
            <div className="archive-filter-pills font-mono" role="tablist" aria-label="Filter by verdict">
              {(["ALL", "REAL", "AI_GENERATED", "AI_MANIPULATED", "UNCERTAIN"] as VerdictFilter[]).map((filterKey) => {
                const isActive = selectedFilter === filterKey;
                const filterLabels: Record<VerdictFilter, string> = {
                  ALL: "ALL",
                  REAL: "REAL",
                  AI_GENERATED: "AI GEN",
                  AI_MANIPULATED: "AI MANIP",
                  UNCERTAIN: "UNCERTAIN",
                };

                return (
                  <button
                    key={filterKey}
                    type="button"
                    role="tab"
                    aria-selected={isActive}
                    className={`archive-pill-btn ${isActive ? "pill-active" : ""}`}
                    onClick={() => setSelectedFilter(filterKey)}
                  >
                    {filterLabels[filterKey]}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* 3. ARCHIVE EVIDENCE CARDS BODY */}
        {/* ==================================================== */}
        <div className="archive-body-viewport">
          {/* EMPTY STATE */}
          {items.length === 0 ? (
            <div className="archive-empty-container">
              <div className="empty-icon-shield">
                <EvidenceFileIcon size={36} className="empty-forensic-icon" />
                <span className="empty-corner cb-tl">┌</span>
                <span className="empty-corner cb-tr">┐</span>
                <span className="empty-corner cb-bl">└</span>
                <span className="empty-corner cb-br">┘</span>
              </div>
              <h4 className="empty-primary-title font-mono">EVIDENCE ARCHIVE</h4>
              <p className="empty-lead-msg">No specimens analyzed yet.</p>
              <p className="empty-sub-msg">
                Analyze an image to create your first evidence record.
              </p>
            </div>
          ) : filteredItems.length === 0 ? (
            /* SEARCH / FILTER EMPTY STATE */
            <div className="archive-no-matches-container font-mono">
              <CrosshairIcon size={22} className="no-matches-icon" />
              <p className="no-matches-title">NO MATCHING EVIDENCE</p>
              <span className="no-matches-sub">
                No session specimens match "{searchQuery}"
              </span>
              <button
                type="button"
                className="btn-reset-filters"
                onClick={() => {
                  setSearchQuery("");
                  setSelectedFilter("ALL");
                }}
              >
                RESET FILTERS
              </button>
            </div>
          ) : (
            /* EVIDENCE SPECIMEN CARDS */
            <div className="archive-cards-list" role="list">
              {filteredItems.map((item, idx) => {
                const theme = getVerdictTheme(item.verdict);
                const isActiveSpecimen = activeId === item.id;
                const formattedNumber = `SPECIMEN #${String(item.specimenNumber).padStart(3, "0")}`;
                const hasImageError = Boolean(imageErrors[item.id]);
                const imageSource = item.thumbnailDataUrl || item.imageSrc;

                // Calibrated confidence (normalized between 0 and 100)
                const confValue = Math.min(100, Math.max(0, item.confidence));
                const confDisplay = confValue.toFixed(1);

                return (
                  <div
                    key={item.id}
                    role="listitem"
                    className={`evidence-archive-card ${isActiveSpecimen ? "card-active-specimen" : ""}`}
                    onClick={() => onSelect(item)}
                    style={{ animationDelay: `${idx * 40}ms` }}
                    tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && onSelect(item)}
                    aria-label={`${formattedNumber}, ${theme.label}, ${confDisplay}% confidence`}
                  >
                    {/* Visual Card Left: Real Thumbnail (16:10 aspect ratio) */}
                    <div className="card-media-column">
                      <div className="card-thumb-frame">
                        {!hasImageError && imageSource ? (
                          <img
                            src={imageSource}
                            alt=""
                            className="card-thumb-image"
                            loading="lazy"
                            onError={() => handleImageError(item.id)}
                          />
                        ) : (
                          /* Intentional Forensic Fallback Placeholder (no broken-image icon) */
                          <div className="card-thumb-fallback font-mono">
                            <CrosshairIcon size={20} className="fallback-crosshair" />
                            <span className="fallback-tag">SPECIMEN</span>
                            <span className="fallback-preview-label">RECORD</span>
                          </div>
                        )}
                        <span className="thumb-corner cb-tl">┌</span>
                        <span className="thumb-corner cb-br">┘</span>
                      </div>
                    </div>

                    {/* Visual Card Right: Forensic Passport Details */}
                    <div className="card-info-column">
                      {/* Top Row: Specimen ID & Active Tag */}
                      <div className="card-header-row font-mono">
                        <span className="card-specimen-id">{formattedNumber}</span>
                        {isActiveSpecimen && (
                          <span className="card-active-badge">
                            <span className="active-pip" />
                            ACTIVE
                          </span>
                        )}
                      </div>

                      {/* Verdict Badge Row */}
                      <div className="card-verdict-row">
                        <span 
                          className="card-verdict-dot" 
                          style={{ backgroundColor: theme.dotColor }}
                        />
                        <span 
                          className="card-verdict-label font-mono" 
                          style={{ color: theme.textColor }}
                        >
                          {theme.label}
                        </span>
                      </div>

                      {/* Filename (small muted text underneath) */}
                      <div className="card-filename-row">
                        <span className="card-filename-text" title={item.filename}>
                          {item.filename}
                        </span>
                      </div>

                      {/* Prominent Confidence Row */}
                      <div className="card-confidence-block font-mono">
                        <div className="conf-label-row">
                          <span className="conf-kicker">CONFIDENCE</span>
                          <span className="conf-percent" style={{ color: theme.textColor }}>
                            {confDisplay}%
                          </span>
                        </div>
                        {/* Thin Horizontal Confidence Meter */}
                        <div className="conf-meter-track">
                          <div 
                            className="conf-meter-fill"
                            style={{ 
                              width: `${confValue}%`,
                              backgroundColor: theme.barColor,
                            }}
                          />
                        </div>
                      </div>

                      {/* Bottom Row: Timestamp & "VIEW EVIDENCE →" CTA */}
                      <div className="card-bottom-row font-mono">
                        <span className="card-timestamp">
                          {item.timestamp}
                        </span>
                        <span className="card-view-cta">
                          <span>VIEW EVIDENCE</span>
                          <ArrowRightIcon size={12} className="cta-arrow" />
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ==================================================== */}
        {/* 4. DRAWER FOOTER: CLEAR SESSION ARCHIVE */}
        {/* ==================================================== */}
        {items.length > 0 && (
          <footer className="archive-drawer-footer">
            {!showClearConfirm ? (
              <button
                type="button"
                className="btn-clear-archive font-mono"
                onClick={() => setShowClearConfirm(true)}
              >
                <ArchiveXIcon size={14} />
                <span>CLEAR SESSION ARCHIVE</span>
              </button>
            ) : (
              /* Inline Confirmation Dialog */
              <div className="clear-confirm-dialog font-mono">
                <p className="confirm-notice">
                  Clear all analyzed specimens from this session?
                </p>
                <div className="confirm-buttons-row">
                  <button
                    type="button"
                    className="btn-confirm-cancel"
                    onClick={() => setShowClearConfirm(false)}
                  >
                    CANCEL
                  </button>
                  <button
                    type="button"
                    className="btn-confirm-clear"
                    onClick={handleConfirmClear}
                  >
                    CLEAR ARCHIVE
                  </button>
                </div>
              </div>
            )}
          </footer>
        )}
      </div>
    </div>
  );
};

export default HistoryDrawer;
