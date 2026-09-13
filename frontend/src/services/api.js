import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export const detectImage = async (file, confThreshold = 0.35, iouThreshold = 0.45) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('conf_threshold', confThreshold);
  formData.append('iou_threshold', iouThreshold);

  const response = await axios.post(`${API_BASE}/api/detect`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const askQuestion = async (question, file = null, confThreshold = 0.35) => {
  const formData = new FormData();
  formData.append('question', question);
  if (file) formData.append('file', file);
  formData.append('conf_threshold', confThreshold);

  const response = await axios.post(`${API_BASE}/api/ask`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};
