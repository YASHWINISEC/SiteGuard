import React, { useState } from 'react';
import Header from '../components/Header';
import ImageUploader from '../components/ImageUploader';
import DetectionViewer from '../components/DetectionViewer';
import DetectionList from '../components/DetectionList';
import SafetySummary from '../components/SafetySummary';
import QuestionBox from '../components/QuestionBox';
import { detectImage } from '../services/api';

export default function Dashboard() {
  const [currentFile, setCurrentFile] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [annotatedBase64, setAnnotatedBase64] = useState(null);
  const [detections, setDetections] = useState([]);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [modelName, setModelName] = useState('RT-DETR');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImageSelected = async (file) => {
    setCurrentFile(file);
    setOriginalUrl(URL.createObjectURL(file));
    setAnnotatedBase64(null);
    setIsProcessing(true);

    try {
      const data = await detectImage(file);
      setDetections(data.detections || []);
      setAnnotatedBase64(data.annotated_image_base64);
      setInferenceTime(data.inference_time_ms);
      setModelName(data.model_name);
    } catch (err) {
      alert(`Detection failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Header modelName={modelName} inferenceTime={inferenceTime} />

      <main style={{ flex: 1, padding: '1.5rem 2rem', maxWidth: '1600px', width: '100%', margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '1.5rem' }}>
          
          {/* Left Column: Viewport & Natural Language Reasoner */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <ImageUploader onImageSelected={handleImageSelected} isProcessing={isProcessing} />
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Accepted: .JPG, .PNG • 704x704 Native Resolution
                </span>
              </div>

              <DetectionViewer
                annotatedBase64={annotatedBase64}
                originalUrl={originalUrl}
                isProcessing={isProcessing}
              />
            </div>

            <QuestionBox currentImageFile={currentFile} />
          </div>

          {/* Right Column: Safety KPIs & Detection List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <SafetySummary detections={detections} />
            <DetectionList detections={detections} />
          </div>

        </div>
      </main>
    </div>
  );
}
