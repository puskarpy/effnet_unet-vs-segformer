import SectionLabel from "./SectionLabel";
import { MODELS } from "../config/constants";

export default function ResultDisplay({ result }) {
  if (!result) return null;

  const models = [
    {
      id: "efficientnet",
      data: result.efficientnet,
    },
    {
      id: "segformer",
      data: result.segformer,
    },
  ];

  return (
    <section className="border border-line bg-panel p-5">
      <SectionLabel>Prediction Result</SectionLabel>

      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-1 font-mono text-[12px] text-ink-soft">
        <p>
          <span className="text-ink-faint">Patient:</span>{" "}
          {result.patientId}
        </p>

        <p>
          <span className="text-ink-faint">Slice:</span>{" "}
          {result.selectedSlice}
        </p>
      </div>

      {models.map((model) => {
        if (!model.data) return null;

        const modelLabel =
          MODELS.find((m) => m.id === model.id)?.label ||
          model.id;

        return (
          <div key={model.id} className="mt-6">
            <div className="font-mono text-[12px] text-ink-soft mb-3">
              <span className="text-ink-faint">Model:</span>{" "}
              {modelLabel}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border border-line bg-paper p-3">
                <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
                  Flair
                </p>

                <img
                  src={model.data.flair}
                  alt={`FLAIR for ${result.patientId}`}
                  className="w-full max-h-120 object-contain"
                />
              </div>

              <div className="border border-line bg-paper p-3">
                <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
                  Ground Truth
                </p>

                <img
                  src={model.data.ground_truth}
                  alt={`Ground truth for ${result.patientId}`}
                  className="w-full max-h-120 object-contain"
                />
              </div>

              <div className="border border-line bg-paper p-3">
                <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
                  Prediction
                </p>

                <img
                  src={model.data.prediction}
                  alt={`${modelLabel} prediction for ${result.patientId}`}
                  className="w-full max-h-120 object-contain"
                />
              </div>

              <div className="border border-line bg-paper p-3">
                <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
                  Overlay
                </p>

                <img
                  src={model.data.overlay}
                  alt={`${modelLabel} overlay for ${result.patientId}`}
                  className="w-full max-h-120 object-contain"
                />
              </div>
            </div>

            <a
              href={model.data.overlay}
              download={`${result.patientId}_${model.id}_overlay.png`}
              className="mt-4 inline-block border border-line-strong px-4 py-2 text-[12px] font-mono uppercase tracking-[0.08em] text-ink hover:border-ink hover:bg-paper transition-colors"
            >
              Download Result
            </a>
          </div>
        );
      })}
    </section>
  );
}