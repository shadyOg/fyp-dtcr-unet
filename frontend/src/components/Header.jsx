import React from 'react';
import { Activity, ShieldCheck, User } from 'lucide-react';

export default function Header({ patientId, setPatientId, isServerReady }) {
  return (
    <header className="header-bar">
      <div className="brand-section">
        <div className="brand-logo-badge">
          <Activity size={24} />
        </div>
        <div className="brand-title">
          <h1>DTCR-U-Net Clinical Workstation</h1>
          <p>Dual-Task AI Pulmonary CT Lesion Segmentation & Volumetry</p>
        </div>
      </div>

      <div className="header-status-group">
        <div className="patient-meta-box">
          <User size={15} color="var(--color-brand-secondary)" />
          <input
            type="text"
            className="patient-input"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            placeholder="Patient ID / Record"
            title="Enter Patient ID for Diagnostic Report"
          />
        </div>

        <div className="api-badge" style={{
          background: isServerReady ? 'rgba(23, 191, 40, 0.1)' : 'rgba(236, 148, 44, 0.1)',
          borderColor: isServerReady ? 'rgba(23, 191, 40, 0.3)' : 'rgba(236, 148, 44, 0.3)',
          color: isServerReady ? 'var(--color-status-clear)' : 'var(--color-status-moderate)',
        }}>
          <span className="status-dot" style={{
            backgroundColor: isServerReady ? 'var(--color-status-clear)' : 'var(--color-status-moderate)',
            boxShadow: isServerReady ? '0 0 8px var(--color-status-clear)' : '0 0 8px var(--color-status-moderate)'
          }} />
          <span>{isServerReady ? 'Inference Engine Active' : 'Connecting to Backend...'}</span>
        </div>
      </div>
    </header>
  );
}
