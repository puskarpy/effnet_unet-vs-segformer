export default function Header() {
  return (
    <header className="border-b border-line bg-panel">
      <div className="mx-auto max-w-4xl px-6 py-5 flex items-baseline justify-between">
        <div>
          <h1 className="text-[17px] font-semibold tracking-tight text-ink">
            Brain Tumor Segmentation
          </h1>
          <p className="mt-0.5 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
            MRI Segmentation Research Tool
          </p>
        </div>
        <span className="hidden sm:inline-block font-mono text-[11px] text-ink-faint border border-line px-2 py-1">
          v0.1
        </span>
      </div>
    </header>
  );
}
