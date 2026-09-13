import React, { useRef } from 'react';

export default function ImageUploader({ onImageSelected, isProcessing }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onImageSelected(e.target.files[0]);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        style={{ display: 'none' }}
      />
      <button
        className="btn btn-primary"
        onClick={() => fileInputRef.current.click()}
        disabled={isProcessing}
      >
        📁 {isProcessing ? 'Processing...' : 'Upload Image'}
      </button>
    </div>
  );
}
