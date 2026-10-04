import React from 'react';
import { ShieldCheck, AlertTriangle, AlertCircle, FileText, Activity, Stethoscope } from 'lucide-react';

export default function ClinicalCard({ analysisData, onOpenReportModal }) {
  if (!analysisData) {
    return (
      <div className="clinical-card" style={{ opacity: 0.6, textAlign: 'center', padding: '3rem 1.5rem' }}>
        <Stethoscope size={36} color="var(--color-brand-secondary)" style={{ margin: '0 auto 1rem auto' }} />
        <div style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--color-text-bright)' }}>
          Awaiting CT Scan
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', marginTop: '0.35rem' }}>
          Upload a 2D or 3D scan on the left and click "Analyze Scan" to generate the diagnostic assessment.
        </div>
      </div>
    );
  }

  const {
    severity,
    severity_code,
    overall_involvement_pct,
    total_lesion_volume_cm3,
    total_lung_volume_cm3,
    clinical_finding,
    action_recommendation,
  } = analysisData;

  const renderSeverityIcon = () => {
    switch (severity_code) {
      case 'clear':
        return <ShieldCheck size={24} />;
      case 'mild':
        return <Activity size={24} />;
      case 'moderate':
        return <AlertTriangle size={24} />;
      case 'severe':
      default:
        return <AlertCircle size={24} />;
    }
  };

  return (
    <div className="clinical-card">
      <div className="card-heading">
        <div className="card-title">Diagnostic Assessment</div>
      </div>

      {/* Severity Status Banner */}
      <div className={`severity-banner ${severity_code}`}>
        <div className="severity-icon-badge">
          {renderSeverityIcon()}
        </div>
        <div className="severity-text-group">
          <h3>{severity}</h3>
          <p>{overall_involvement_pct}% Total Lung Involvement</p>
        </div>
      </div>

      {/* Quantitative Volumetry Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-label">Lesion Volume</span>
          <span className="metric-value" style={{ color: 'var(--color-brand-secondary)' }}>
            {total_lesion_volume_cm3}
            <span className="metric-unit">cm³</span>
          </span>
        </div>

        <div className="metric-box">
          <span className="metric-label">Total Lung Field</span>
          <span className="metric-value" style={{ color: 'var(--color-action-blue)' }}>
            {total_lung_volume_cm3}
            <span className="metric-unit">cm³</span>
          </span>
        </div>
      </div>

      {/* Non-Technical Findings */}
      <div className="findings-box">
        <div className="findings-title">
          <Activity size={14} />
          <span>Radiological Observation</span>
        </div>
        <div>{clinical_finding}</div>
      </div>

      {/* Action Recommendation */}
      <div className="findings-box" style={{ background: 'rgba(112, 0, 165, 0.05)', borderColor: 'rgba(168, 101, 200, 0.25)' }}>
        <div className="findings-title" style={{ color: 'var(--color-brand-secondary)' }}>
          <Stethoscope size={14} />
          <span>Clinical Correlation</span>
        </div>
        <div style={{ color: 'var(--color-text-bright)' }}>{action_recommendation}</div>
      </div>

      {/* PDF Export Section */}
      <div className="export-section">
        <button
          type="button"
          className="btn-export-report"
          onClick={onOpenReportModal}
        >
          <FileText size={18} />
          <span>Export Medical Report (PDF)</span>
        </button>
      </div>
    </div>
  );
}
