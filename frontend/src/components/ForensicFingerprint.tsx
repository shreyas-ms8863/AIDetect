import type { FC } from "react";

interface ForensicFingerprintProps {
  genProb: number;
  manipProb: number;
  fusionProb: number;
}

export const ForensicFingerprint: FC<ForensicFingerprintProps> = ({
  genProb,
  manipProb,
  fusionProb,
}) => {
  const pGen = Math.round((genProb <= 1.0 ? genProb * 100 : genProb));
  const pManip = Math.round((manipProb <= 1.0 ? manipProb * 100 : manipProb));
  const pFusion = Math.round((fusionProb <= 1.0 ? fusionProb * 100 : fusionProb));

  return (
    <div className="bento-tile fingerprint-tile">
      <div className="tile-header">
        <span className="tile-eyebrow">FORENSIC FINGERPRINT</span>
        <span className="tile-tag">TRI-AXIAL COUPLING</span>
      </div>

      <div className="fingerprint-rings-cluster">
        {/* Ring 1: Generation */}
        <div className="fp-ring-item">
          <div className="fp-ring-circle ring-gen">
            <span className="fp-ring-val">{pGen}%</span>
          </div>
          <span className="fp-ring-label">GENERATE</span>
        </div>

        <div className="fp-ring-connector">──</div>

        {/* Ring 2: Manipulation */}
        <div className="fp-ring-item">
          <div className="fp-ring-circle ring-manip">
            <span className="fp-ring-val">{pManip}%</span>
          </div>
          <span className="fp-ring-label">MANIPUL.</span>
        </div>

        <div className="fp-ring-connector">──</div>

        {/* Ring 3: Fusion */}
        <div className="fp-ring-item">
          <div className="fp-ring-circle ring-fusion">
            <span className="fp-ring-val">{pFusion}%</span>
          </div>
          <span className="fp-ring-label">FUSION</span>
        </div>
      </div>
    </div>
  );
};
