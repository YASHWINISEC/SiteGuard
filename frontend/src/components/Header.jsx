import React from 'react';

export default function Header({ modelName, inferenceTime }) {
  return (
    <header style={{
      background: 'rgba(10, 13, 20, 0.85)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid var(--border-color)',
      padding: '0.85rem 2rem',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div style={{
          width: '38px',
          height: '38px',
          background: 'linear-gradient(135deg, #10b981, #06b6d4)',
          borderRadius: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '1.2rem'
        }}>
          🦺
        </div>
        <div>
          <h1 style={{ fontFamily: 'Outfit', fontSize: '1.25rem', fontWeight: 800 }}>
            SiteGuard <span style={{ color: '#10b981' }}>AI</span>
          </h1>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>
            RT-DETR Construction Safety & Reasoning Platform
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
        <div style={{
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid var(--border-color)',
          padding: '0.3rem 0.75rem',
          borderRadius: '999px',
          fontSize: '0.75rem',
          color: '#93c5fd'
        }}>
          ⚡ {modelName || 'RT-DETR Safety Engine'} {inferenceTime ? `(${inferenceTime}ms)` : ''}
        </div>
        <div style={{
          background: 'rgba(16,185,129,0.15)',
          border: '1px solid rgba(16,185,129,0.4)',
          color: '#34d399',
          padding: '0.3rem 0.75rem',
          borderRadius: '999px',
          fontSize: '0.75rem',
          fontWeight: 700
        }}>
          🛡️ OSHA 1926 Active
        </div>
      </div>
    </header>
  );
}
