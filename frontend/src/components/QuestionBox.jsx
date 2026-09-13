import React, { useState } from 'react';
import { askQuestion } from '../services/api';

export default function QuestionBox({ currentImageFile }) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  const sampleQuestions = [
    "How many workers are in this image?",
    "Is anyone not wearing a helmet or vest?",
    "What's the most common object here?",
    "What is OSHA Standard 1926.100?"
  ];

  const handleAsk = async (qText) => {
    const query = qText || question;
    if (!query.trim()) return;

    setLoading(true);
    setResponse(null);
    try {
      const data = await askQuestion(query, currentImageFile);
      setResponse(data);
    } catch (err) {
      alert(`Reasoning query failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>🤖 Natural Language Safety Reasoner (Part B)</h3>
        <span style={{ fontSize: '0.7rem', color: '#38bdf8', background: 'rgba(6,182,212,0.1)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
          Zero Frameworks • Pure Python Logic
        </span>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
          placeholder="Ask a question about the image or OSHA safety rules..."
          style={{
            flex: 1,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '0.6rem 0.85rem',
            color: '#fff',
            fontSize: '0.85rem',
            outline: 'none'
          }}
        />
        <button
          className="btn btn-primary"
          onClick={() => handleAsk()}
          disabled={loading || !question.trim()}
        >
          {loading ? 'Reasoning...' : 'Ask'}
        </button>
      </div>

      {/* Suggested Quick Questions */}
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
        {sampleQuestions.map((sq, i) => (
          <button
            key={i}
            onClick={() => {
              setQuestion(sq);
              handleAsk(sq);
            }}
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '4px',
              color: 'var(--text-secondary)',
              fontSize: '0.7rem',
              padding: '0.25rem 0.5rem',
              cursor: 'pointer'
            }}
          >
            {sq}
          </button>
        ))}
      </div>

      {/* Response Box */}
      {response && (
        <div style={{
          background: response.status === 'insufficient_information' ? 'rgba(239,68,68,0.1)' : 'rgba(255,255,255,0.02)',
          border: `1px solid ${response.status === 'insufficient_information' ? 'rgba(239,68,68,0.3)' : 'var(--border-color)'}`,
          borderRadius: '8px',
          padding: '0.85rem',
          marginTop: '0.5rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{
              fontSize: '0.72rem',
              fontWeight: 700,
              background: 'rgba(59,130,246,0.2)',
              color: '#93c5fd',
              padding: '0.15rem 0.45rem',
              borderRadius: '4px'
            }}>
              Intent: {response.intent}
            </span>
            <span style={{ fontSize: '0.7rem', color: response.guardrail?.passed ? '#34d399' : '#f87171' }}>
              Guardrail: {response.guardrail?.passed ? 'PASSED' : 'FLAGGED INSUFFICIENT'}
            </span>
          </div>

          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.5rem' }}>
            {response.answer}
          </div>

          {/* Reasoning Steps Trace */}
          {response.reasoning_trace && (
            <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.5rem', marginTop: '0.5rem' }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                Reasoning Trace:
              </div>
              {response.reasoning_trace.map((step, idx) => (
                <div key={idx} style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
                  <strong style={{ color: '#38bdf8' }}>Step {step.step_number} ({step.component}):</strong> {step.result}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
