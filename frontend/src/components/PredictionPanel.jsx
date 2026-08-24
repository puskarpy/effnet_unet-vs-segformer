export default function PredictionPanel({ isLoading, error, onRun }) {
  return (
    <section className="border border-line bg-panel p-5">
      <button
        type="button"
        disabled={isLoading}
        onClick={onRun}
        className="w-full bg-ink text-panel py-3 text-[13px] font-medium tracking-wide hover:bg-accent transition-colors disabled:opacity-60 disabled:hover:bg-ink flex items-center justify-center gap-2.5"
      >
        {isLoading && <span className="spinner" />}
        {isLoading ? "Running Segmentation..." : "Run Segmentation"}
      </button>

      {error && (
        <div className="mt-3 border border-accent/40 bg-accent-soft px-3 py-2.5">
          <p className="text-[13px] text-accent">{error}</p>
        </div>
      )}
    </section>
  );
}
