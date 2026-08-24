import { API_BASE_URL, PREDICT_ENDPOINT } from "../config/constants";

export async function predictTumor(files) {
  const formData = new FormData();

  formData.append("flair", files.flair);
  formData.append("t1", files.t1);
  formData.append("t1ce", files.t1ce);
  formData.append("t2", files.t2);
  formData.append("seg", files.seg);

  const response = await fetch(
    PREDICT_ENDPOINT,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    let message = "Prediction failed.";

    try {
      const error = await response.json();

      if (typeof error.detail === "string") {
        message = error.detail;
      }
    } catch {
      // Ignore JSON parsing error
    }

    throw new Error(message);
  }

  const data = await response.json();

  const addBaseUrl = (model) => ({
    flair: `${API_BASE_URL}${model.flair}`,
    ground_truth: `${API_BASE_URL}${model.ground_truth}`,
    prediction: `${API_BASE_URL}${model.prediction}`,
    overlay: `${API_BASE_URL}${model.overlay}`,
  });

  return {
    success: data.success,
    patientId: data.patient_id,
    selectedSlice: data.selected_slice,

    efficientnet: addBaseUrl(
      data.efficientnet
    ),

    segformer: addBaseUrl(
      data.segformer
    ),
  };
}