import type { FC } from "react";
import { RotateCcwIcon } from "./Icons";

interface AnalyzeAnotherProps {
  onReset: () => void;
}

export const AnalyzeAnother: FC<AnalyzeAnotherProps> = ({ onReset }) => {
  return (
    <div className="analyze-another-dock">
      <button
        type="button"
        className="btn btn-secondary-tech btn-analyze-another"
        onClick={onReset}
      >
        <RotateCcwIcon size={16} />
        <span>ANALYZE ANOTHER SPECIMEN</span>
      </button>
    </div>
  );
};
