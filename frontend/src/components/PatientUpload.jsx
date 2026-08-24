import { useRef, useState } from "react";
import SectionLabel from "./SectionLabel";

export default function PatientUpload({ patientName, fileCount, onFilesSelected }) {
  const inputRef = useRef(null);
  const [dirSupported] = useState(() => "webkitdirectory" in document.createElement("input"));

function handleChange(e) {
  const fileList = Array.from(e.target.files || []);

  if (fileList.length === 0) return;

  const relPath = fileList[0].webkitRelativePath;

  const folderName = relPath
    ? relPath.split("/")[0]
    : "selected_files";

  const files = {
    flair: null,
    t1: null,
    t1ce: null,
    t2: null,
    seg: null,
  };

  for (const file of fileList) {
    const name = file.name.toLowerCase();

    if (name.includes("_flair")) {
      files.flair = file;
    } else if (name.includes("_t1ce")) {
      files.t1ce = file;
    } else if (
      name.includes("_t1") &&
      !name.includes("_t1ce")
    ) {
      files.t1 = file;
    } else if (name.includes("_t2")) {
      files.t2 = file;
    } else if (name.includes("_seg")) {
      files.seg = file;
    }
  }

  const missing = Object.entries(files)
    .filter(([, file]) => !file)
    .map(([name]) => name);

  if (missing.length > 0) {
    alert(
      `Missing MRI files: ${missing.join(", ")}`
    );
    return;
  }

  onFilesSelected(
    folderName,
    files
  );
}

  return (
    <section className="border border-line bg-panel p-5">
      <SectionLabel>Patient Data</SectionLabel>
      <p className="mt-2 text-[13px] text-ink-soft leading-relaxed">
        Select a folder containing the patient's MRI images.
      </p>

      <div className="mt-4">
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          multiple
          webkitdirectory={dirSupported ? "true" : undefined}
          directory={dirSupported ? "true" : undefined}
          onChange={handleChange}
        />

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="w-full border border-dashed border-line-strong px-4 py-6 text-center hover:border-ink hover:bg-paper transition-colors"
        >
          <span className="block font-mono text-[11px] uppercase tracking-widest text-ink-faint">
            {dirSupported ? "Click to select patient folder" : "Click to select MRI files"}
          </span>
        </button>

        {patientName && (
          <div className="mt-3 flex items-center justify-between border border-line bg-paper px-3 py-2">
            <div>
              <p className="font-mono text-[13px] text-ink">{patientName}</p>
              <p className="font-mono text-[11px] text-ink-faint">
                {fileCount} MRI file{fileCount === 1 ? "" : "s"} selected
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                if (inputRef.current) inputRef.current.value = "";
                onFilesSelected("", []);
              }}
              className="font-mono text-[11px] text-ink-faint hover:text-accent"
            >
              clear
            </button>
          </div>
        )}

        {!dirSupported && (
          <p className="mt-2 font-mono text-[11px] text-ink-faint">
            Folder selection is not supported in this browser — select the
            individual MRI files instead.
          </p>
        )}
      </div>
    </section>
  );
}
