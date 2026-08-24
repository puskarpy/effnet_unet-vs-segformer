import { MODELS } from "../config/constants";
import SectionLabel from "./SectionLabel";

export default function ModelSelector({ selectedModel, onChange }) {
  const active = MODELS.find((m) => m.id === selectedModel);

  return (
    <section className="border border-line bg-panel p-5">
      <SectionLabel>Segmentation Model</SectionLabel>

      <div className="mt-4">
        <select
          value={selectedModel}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none border border-line-strong bg-paper px-3 py-2.5 text-[13px] text-ink focus:outline-none focus:border-ink"
        >
          {MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>

        <div className="mt-4 space-y-1.5">
          {MODELS.map((m) => (
            <p
              key={m.id}
              className={`text-[12px] leading-relaxed ${
                m.id === selectedModel ? "text-ink-soft" : "text-ink-faint"
              }`}
            >
              <span className="font-mono font-medium">{m.label}:</span>{" "}
              {m.description}
            </p>
          ))}
        </div>
      </div>
    </section>
  );
}
