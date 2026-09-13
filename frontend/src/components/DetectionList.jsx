import React from 'react';

export default function DetectionList({ detections = [] }) {
  return (
    <div className="card" style={{ maxHeight: '380px', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>🔍 Detected Objects ({detections.length})</h3>
      </div>

      {detections.length === 0 ? (
        <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center', padding: '1rem' }}>
          No detections yet.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {detections.map((det, idx) => (
            <div
              key={idx}
              style={{
                background: 'rgba(255,255,255,0.02)',
                borderLeft: `3px solid ${det.is_violation ? '#ef4444' : '#10b981'}`,
                padding: '0.6rem 0.75rem',
                borderRadius: '0 6px 6px 0',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.82rem', color: det.is_violation ? '#fca5a5' : '#f8fafc' }}>
                  {det.is_violation ? '⚠️ ' : ''}{det.class_name}
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                  Box: [{det.bbox.join(', ')}]
                </div>
              </div>
              <div style={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: det.is_violation ? '#ef4444' : '#10b981'
              }}>
                {Math.round(det.confidence * 100)}%
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
