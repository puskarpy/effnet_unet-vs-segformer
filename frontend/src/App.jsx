import { useState } from "react";
import Header from "./components/Header";
import PatientUpload from "./components/PatientUpload";
import PredictionPanel from "./components/PredictionPanel";
import ResultDisplay from "./components/ResultDisplay";
import { predictTumor } from "./services/api";

export default function App() {
  const [patientName, setPatientName] = useState("");
  const [files, setFiles] = useState(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  function handleFilesSelected(folderName, selectedFiles) {
    setPatientName(folderName);
    setFiles(selectedFiles);

    setResult(null);
    setError("");
  }

  async function handleRun() {
    setError("");

    if (!patientName || !files) {
      setError(
        "Select a patient folder before running segmentation."
      );
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const prediction = await predictTumor(files);

      setResult(prediction);
    } catch (err) {
      setError(
        err?.message ||
          "Prediction failed. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  }

  const fileCount = files
    ? Object.values(files).filter(Boolean).length
    : 0;

  return (
    <div className="min-h-screen bg-scanlines">
      <Header />

      <main className="mx-auto max-w-4xl px-6 py-8 space-y-5">
        <div className="gap-5">
          <PatientUpload
            patientName={patientName}
            fileCount={fileCount}
            onFilesSelected={handleFilesSelected}
          />
        </div>

        <PredictionPanel
          isLoading={isLoading}
          error={error}
          onRun={handleRun}
        />

        <ResultDisplay result={result} />
      </main>

      <footer className="mx-auto max-w-4xl px-6 pb-8">
        <p className="font-mono text-[11px] text-ink-faint">
          Research prototype — not for clinical use.
        </p>
      </footer>
    </div>
  );
}