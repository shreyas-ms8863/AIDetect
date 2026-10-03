import type { FC } from "react";

export const SignalConnectionLine: FC = () => {
  return (
    <div className="signal-connection-lane" aria-hidden="true">
      <div className="connection-track">
        {/* Animated signal node & traveling photons */}
        <span className="signal-photon photon-1" />
        <span className="signal-photon photon-2" />
        <span className="signal-photon photon-3" />
        
        {/* Center Analysis Node */}
        <div className="signal-center-node">
          <span className="node-pip" />
          <span className="node-caption font-mono">ANALYSIS</span>
        </div>
      </div>
      <div className="signal-arrow-terminus font-mono">►</div>
    </div>
  );
};
