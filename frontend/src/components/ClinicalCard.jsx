import React from 'react';
import { ShieldCheck, AlertTriangle, AlertCircle, FileText, Activity, Stethoscope, CheckCircle2 } from 'lucide-react';

export default function ClinicalCard({ analysisData, onOpenReportModal, onAnalyze, isAnalyzing }) {
  if (!analysisData) {
    return (
      <div className="clinical-card" style={{ opacity: 0.6, textAlign: 'center', padding: '3rem 1.5rem' }}>
        <Stethoscope size={36} color="var(--color-brand-secondary)" style={{ margin: '0 auto 1rem auto' }} />
        <div style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--color-text-bright)' }}>
          Awaiting CT Scan
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', marginTop: '0.35rem' }}>
          Upload any 2D or 3D scan on the left (.npy, .dcm, .nii, .png) to inspect the image and run AI analysis.
        </div>
      </div>
    );
  }

  if (analysisData.is_preview) {
    return (
      <div className="clinical-card">
        <div className="card-heading">
          <div className="card-title">Scan Loaded for Inspection</div>
        </div>

        <div style={{
          background: 'rgba(37, 180, 248, 0.08)',
          border: '1px solid rgba(37, 180, 248, 0.25)',
          borderRadius: 'var(--radius-module)',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-accent-sky)', fontWeight: 600 }}>
            <Activity size={18} />
            <span>Ready for Analysis</span>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--color-text-main)' }}>
            This scan is currently in <strong>Preview Mode</strong>. You can inspect all slices and adjust zoom on the viewer.
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
            <span>Format: {analysisData.metadata?.format?.toUpperCase() || 'CT'}</span>
            <span>•</span>
            <span>Total Slices: {analysisData.total_slices}</span>
          </div>
        </div>

        <div className="export-section" style={{ marginTop: 'auto' }}>
          <button
            type="button"
            className="btn-primary"
            style={{ width: '100%', justifyContent: 'center', padding: '0.85rem' }}
            onClick={onAnalyze}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                <span>Running DTCR-U-Net...</span>
              </>
            ) : (
              <>
                <Activity size={18} />
                <span>Run AI Lesion Analysis</span>
              </>
            )}
          </button>
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
    has_ground_truth,
    overall_dice,
    overall_iou,
    overall_sensitivity,
    overall_specificity,
    ground_truth_filename,
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

      {/* Ground Truth Validation Box (If uploaded) */}
      {has_ground_truth && (
        <div style={{
          background: 'rgba(23, 191, 40, 0.08)',
          border: '1px solid rgba(23, 191, 40, 0.3)',
          borderRadius: 'var(--radius-module)',
          padding: '0.85rem 1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--color-status-clear)', fontWeight: 600, fontSize: '0.85rem' }}>
              <CheckCircle2 size={16} />
              <span>Ground Truth Validation</span>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
              {ground_truth_filename}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.4rem 0.6rem', borderRadius: 'var(--radius-chip)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>DICE AGREEMENT</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--color-status-clear)' }}>{overall_dice}%</div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.4rem 0.6rem', borderRadius: 'var(--radius-chip)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>IoU OVERLAP</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--color-action-blue)' }}>{overall_iou}%</div>
            </div>
          </div>
        </div>
      )}

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
