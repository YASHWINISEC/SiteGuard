import React from 'react';

export default function DetectionViewer({ annotatedBase64, originalUrl, isProcessing }) {
  return (
    <div style={{
      background: '#06090e',
      borderRadius: 'var(--radius-md)',
      border: '1px solid rgba(255,255,255,0.05)',
      minHeight: '440px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {annotatedBase64 ? (
        <img
          src={`data:image/jpeg;base64,${annotatedBase64}`}
          alt="Detection output"
          style={{ maxWidth: '100%', maxHeight: '560px', objectFit: 'contain' }}
        />
      ) : originalUrl ? (
        <img
          src={originalUrl}
          alt="Source upload"
          style={{ maxWidth: '100%', maxHeight: '560px', objectFit: 'contain' }}
        />
      ) : (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: '3rem', marginBottom: '0.5rem' }}>📷</div>
          <p>Upload an image or sample to view RT-DETR safety detections.</p>
        </div>
      )}

      {isProcessing && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(0,0,0,0.6)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#10b981',
          fontWeight: 600
        }}>
          ⚡ Running RT-DETR Inference...
        </div>
      )}
    </div>
  );
}
