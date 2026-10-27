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
    const isValidCt = analysisData.is_valid_ct !== false;

    return (
      <div className="clinical-card">
        <div className="card-heading">
          <div className="card-title">
            {isValidCt ? 'Scan Loaded for Inspection' : 'Invalid Image Input'}
          </div>
        </div>

        {isValidCt ? (
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
        ) : (
          <div style={{
            background: 'rgba(229, 35, 35, 0.08)',
            border: '1px solid rgba(229, 35, 35, 0.3)',
            borderRadius: 'var(--radius-module)',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-status-severe)', fontWeight: 600 }}>
              <AlertCircle size={18} />
              <span>Invalid Image Type</span>
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--color-text-main)', lineHeight: 1.4 }}>
              {analysisData.validation_reason || 'Invalid image type, please upload a valid CT scan.'}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)', background: 'rgba(0,0,0,0.2)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-chip)' }}>
              ℹ️ Please upload a valid CT scan (DICOM, NIfTI, .npy, or grayscale CT image).
            </div>
          </div>
        )}

        <div className="export-section" style={{ marginTop: 'auto' }}>
          <button
            type="button"
            className="btn-primary"
            style={{
              width: '100%',
              justifyContent: 'center',
              padding: '0.85rem',
              opacity: isValidCt ? 1 : 0.45,
              cursor: isValidCt ? 'pointer' : 'not-allowed',
              background: isValidCt ? undefined : 'var(--color-border-subtle)',
            }}
            onClick={isValidCt ? onAnalyze : undefined}
            disabled={isAnalyzing || !isValidCt}
            title={isValidCt ? 'Run AI Analysis' : (analysisData.validation_reason || 'Invalid image type, please upload a valid CT scan.')}
          >
            {isAnalyzing ? (
              <>
                <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                <span>Running DTCR-U-Net...</span>
              </>
            ) : isValidCt ? (
              <>
                <Activity size={18} />
                <span>Run AI Lesion Analysis</span>
              </>
            ) : (
              <>
                <AlertCircle size={18} color="var(--color-status-severe)" />
                <span>Analysis Disabled</span>
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
    action_recommendation,
    has_ground_truth,
    ai_accuracy,
    accuracy_tier,
    accuracy_interpretation,
    overall_dice,
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

  const displayAccuracy = ai_accuracy ?? overall_dice;

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
          padding: '0.9rem 1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.65rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--color-status-clear)', fontWeight: 600, fontSize: '0.85rem' }}>
              <CheckCircle2 size={16} />
              <span>Ground Truth Validation</span>
            </div>
            <span style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', maxWidth: 130, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} title={ground_truth_filename}>
              {ground_truth_filename}
            </span>
          </div>

          {/* Unified AI Accuracy Hero Card */}
          <div style={{
            background: 'rgba(0, 0, 0, 0.25)',
            padding: '0.75rem 0.85rem',
            borderRadius: 'var(--radius-card)',
            border: '1px solid rgba(23, 191, 40, 0.2)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.45rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                AI Accuracy
              </span>
              <span style={{
                fontSize: '0.72rem',
                fontWeight: 600,
                padding: '0.15rem 0.5rem',
                borderRadius: 'var(--radius-chip)',
                background: displayAccuracy >= 70 ? 'rgba(23, 191, 40, 0.2)' : 'rgba(236, 148, 44, 0.2)',
                color: displayAccuracy >= 70 ? 'var(--color-status-clear)' : 'var(--color-status-moderate)',
              }}>
                {accuracy_tier || (displayAccuracy >= 70 ? 'Strong Match' : 'Moderate Overlap')}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.4rem' }}>
              <span style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--color-status-clear)', lineHeight: 1 }}>
                {displayAccuracy}%
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                match with true lesion annotation
              </span>
            </div>

            {/* Visual accuracy progress bar */}
            <div style={{ width: '100%', height: 6, background: 'rgba(255, 255, 255, 0.1)', borderRadius: 3, overflow: 'hidden', marginTop: '0.15rem' }}>
              <div style={{
                width: `${Math.min(100, Math.max(0, displayAccuracy))}%`,
                height: '100%',
                background: displayAccuracy >= 70 ? 'var(--color-status-clear)' : 'var(--color-status-moderate)',
                borderRadius: 3,
                transition: 'width 0.4s ease',
              }} />
            </div>

            {accuracy_interpretation && (
              <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.2rem', lineHeight: 1.35 }}>
                {accuracy_interpretation}
              </div>
            )}
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
