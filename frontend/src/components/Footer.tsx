import type { FC } from "react";

export const Footer: FC = () => {
  return (
    <footer className="forensic-footer">
      <div className="footer-inner">
        <div className="footer-brand-column">
          <div className="footer-brand-title">
            <span className="brand-dot"></span>
            <span className="brand-name">AIDETECT</span>
            <span className="brand-sub font-mono">DIGITAL FORENSICS</span>
          </div>
          <p className="footer-text">
            Autonomous multi-frequency forensic verification suite for digital imagery.
          </p>
        </div>

        <div className="footer-meta-column font-mono">
          <span className="meta-line">SPATIAL CONVOLUTIONAL TEXTURE ANALYSIS</span>
          <span className="meta-line">2D DISCRETE FOURIER TRANSFORM (FFT)</span>
          <span className="meta-line">CALIBRATED POSTERIOR PROBABILITY FUSION</span>
        </div>
      </div>
      <div className="footer-copyright font-mono">
        © 2026 AIDETECT DIGITAL FORENSICS • CONFIDENTIAL EVIDENCE SYSTEM
      </div>
    </footer>
  );
};
