import type { FC } from "react";

export const Hero: FC = () => {
  return (
    <section className="evidence-hero">
      <div className="hero-eyebrow">
        <span className="hero-eyebrow-dot"></span>
        <span className="hero-eyebrow-text font-mono">AIDETECT • DIGITAL EVIDENCE WORKSTATION</span>
      </div>

      <h1 className="hero-headline hero-headline-triptych">
        <span>VERIFY.</span>
        <span>INSPECT.</span>
        <span>UNCOVER.</span>
      </h1>

      <p className="hero-subtext">
        Analyze an image for signs of synthetic generation or AI-assisted manipulation.
      </p>

      <div className="hero-technical-strip">
        <span className="tech-tag font-mono">SPATIAL CONVOLUTIONAL PROFILES</span>
        <span className="tech-sep">•</span>
        <span className="tech-tag font-mono">2D FOURIER FREQUENCY HARMONICS</span>
        <span className="tech-sep">•</span>
        <span className="tech-tag font-mono">CALIBRATED POSTERIOR VERDICT</span>
      </div>
    </section>
  );
};
