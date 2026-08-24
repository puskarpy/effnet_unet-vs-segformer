// Central place for backend configuration.
// Change API_BASE_URL to point the frontend at a different backend instance.

export const API_BASE_URL = "http://localhost:8000";
export const PREDICT_ENDPOINT = `${API_BASE_URL}/api/predict/`;

export const MODELS = [
  {
    id: "efficientnet",
    label: "EfficientNet",
    description: "CNN-based segmentation model",
  },
  {
    id: "segformer",
    label: "SegFormer",
    description: "Transformer-based segmentation model",
  },
];

export const DEFAULT_MODEL = "efficientnet";

// Accepted MRI file extensions, used for lightweight client-side validation.
export const ACCEPTED_EXTENSIONS = [".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg"];
