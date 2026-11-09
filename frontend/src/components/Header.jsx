import React from 'react';
import { Activity, ShieldCheck, User, Zap, Info } from 'lucide-react';

export default function Header({ patientId, setPatientId, isServerReady }) {
  return (
    <header className="header-bar" role="banner">
      {/* Brand & System Title */}
      <div className="brand-section">
        <div className="brand-logo-badge" aria-hidden="true">
          <Activity size={22} strokeWidth={2.2} />
        </div>
        <div className="brand-title">
          <h1>DTCR-U-Net Clinical Workstation</h1>
          <p>Dual-Task AI Pulmonary CT Lesion Segmentation & Volumetry</p>
        </div>
      </div>

      {/* Workstation Controls & Telemetry */}
      <div className="header-status-group">
        {/* Patient Record ID Input */}
        <div className="patient-meta-box">
          <User size={15} color="var(--color-text-secondary)" aria-hidden="true" />
          <label htmlFor="patient-id-input" className="patient-input-label">
            Record ID:
          </label>
          <input
            id="patient-id-input"
            type="text"
            className="patient-input"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            placeholder="e.g. PT-2026-8819"
            title="Enter Patient Identifier for Diagnostic Records & PDF Report"
            aria-label="Patient Identification Number"
            spellCheck="false"
          />
        </div>

        {/* Backend Heartbeat Status Pill */}
        <div
          className="api-badge"
          style={{
            background: isServerReady ? 'var(--color-status-clear-bg)' : 'var(--color-status-moderate-bg)',
            borderColor: isServerReady ? 'var(--color-status-clear-border)' : 'var(--color-status-moderate-border)',
            color: isServerReady ? 'var(--color-status-clear)' : 'var(--color-status-moderate)',
          }}
          role="status"
          aria-live="polite"
        >
          <span
            className={`status-dot ${isServerReady ? 'active' : 'connecting'}`}
            aria-hidden="true"
          />
          <span>{isServerReady ? 'Inference Engine Active' : 'Connecting Engine...'}</span>
        </div>
      </div>
    </header>
  );
}
