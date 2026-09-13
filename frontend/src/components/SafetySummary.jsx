import React from 'react';

export default function SafetySummary({ detections = [] }) {
  const persons = detections.filter(d => d.class_name === 'Person');
  const hardhats = detections.filter(d => d.class_name === 'Hardhat');
  const noHardhats = detections.filter(d => d.class_name === 'NO-Hardhat');
  const vests = detections.filter(d => d.class_name === 'Safety Vest');
  const noVests = detections.filter(d => d.class_name === 'NO-Safety Vest');

  const totalViolations = noHardhats.length + noVests.length;
  let score = 100 - (noHardhats.length * 20) - (noVests.length * 15);
  if (persons.length > 0 && hardhats.length === 0 && vests.length === 0 && totalViolations === 0) {
    score = 70;
  }
  const safetyScore = Math.max(0, Math.min(100, score));

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>🛡️ Site Safety Index</h3>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <div style={{
            fontSize: '2.2rem',
            fontWeight: 800,
            color: safetyScore >= 90 ? '#10b981' : (safetyScore >= 70 ? '#f59e0b' : '#ef4444')
          }}>
            {safetyScore}%
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Compliance Score
          </div>
        </div>

        <div style={{
          padding: '0.35rem 0.85rem',
          borderRadius: '999px',
          fontSize: '0.75rem',
          fontWeight: 700,
          background: safetyScore >= 90 ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
          color: safetyScore >= 90 ? '#34d399' : '#f87171'
        }}>
          {safetyScore >= 90 ? 'SAFE' : (safetyScore >= 70 ? 'MODERATE RISK' : 'HIGH RISK')}
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '0.5rem',
        fontSize: '0.75rem',
        background: 'rgba(255,255,255,0.02)',
        padding: '0.75rem',
        borderRadius: '8px'
      }}>
        <div>Workers: <strong>{persons.length}</strong></div>
        <div>Violations: <strong style={{ color: totalViolations > 0 ? '#ef4444' : '#10b981' }}>{totalViolations}</strong></div>
        <div>NO-Hardhat: <strong style={{ color: noHardhats.length > 0 ? '#ef4444' : '#10b981' }}>{noHardhats.length}</strong></div>
        <div>NO-Vest: <strong style={{ color: noVests.length > 0 ? '#ef4444' : '#10b981' }}>{noVests.length}</strong></div>
      </div>
    </div>
  );
}
