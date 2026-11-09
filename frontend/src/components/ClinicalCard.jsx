import React from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  AlertCircle,
  FileText,
  Activity,
  Stethoscope,
  CheckCircle2,
  Sparkles,
  BarChart3
} from 'lucide-react';

export default function ClinicalCard({
  analysisData,
  onOpenReportModal,
  onAnalyze,
  isAnalyzing
}) {
  // Empty State: Awaiting Scan
  if (!analysisData) {
    return (
      <aside className="clinical-card" aria-label="Clinical Diagnostic Assessment">
        <div style={{ textAlign: 'center', padding: '3.5rem 1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid var(--border-standard)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--color-text-secondary)',
              marginBottom: '1rem'
            }}
            aria-hidden="true"
          >
            <Stethoscope size={26} strokeWidth={1.8} />
          </div>
          <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            Awaiting CT Scan
          </h2>
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-secondary)', marginTop: '0.35rem', maxWidth: 280, lineHeight: 1.45 }}>
            Upload a 2D or 3D thoracic scan on the left (.npy, .dcm, .nii, .png) to inspect imagery and generate diagnostic findings.
          </p>
        </div>
      </aside>
    );
  }

  // Preview Mode: Scan Loaded for Inspection
  if (analysisData.is_preview) {
    const isValidCt = analysisData.is_valid_ct !== false;

    return (
      <aside className="clinical-card" aria-label="Clinical Scan Inspection">
        <div className="card-heading">
          <h2 className="card-title">
            {isValidCt ? 'Scan Pre-Analysis Telemetry' : 'Invalid Image Format'}
          </h2>
        </div>

        {isValidCt ? (
          <div
            style={{
              background: 'rgba(10, 132, 255, 0.06)',
              border: '1px solid rgba(10, 132, 255, 0.22)',
              borderRadius: 'var(--radius-card)',
              padding: '1.15rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-action-blue)', fontWeight: 600, fontSize: 'var(--text-sm)' }}>
              <Activity size={16} strokeWidth={2} aria-hidden="true" />
              <span>Inspection Ready</span>
            </div>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-primary)', lineHeight: 1.45 }}>
              The scan is loaded in <strong>Preview Mode</strong>. You can navigate axial slices, inspect tissue densities, and adjust zoom before running inference.
            </p>
            <div style={{ display: 'flex', gap: '0.625rem', fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)' }}>
              <span>Format: <strong>{analysisData.metadata?.format?.toUpperCase() || 'CT'}</strong></span>
              <span>•</span>
              <span>Total Slices: <strong className="tabular-nums">{analysisData.total_slices}</strong></span>
            </div>
          </div>
        ) : (
          <div
            style={{
              background: 'var(--color-status-severe-bg)',
              border: '1px solid var(--color-status-severe-border)',
              borderRadius: 'var(--radius-card)',
              padding: '1.15rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-status-severe)', fontWeight: 600, fontSize: 'var(--text-sm)' }}>
              <AlertCircle size={16} strokeWidth={2} aria-hidden="true" />
              <span>Guardrail Activated</span>
            </div>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-primary)', lineHeight: 1.45 }}>
              {analysisData.validation_reason || 'The uploaded file does not match thoracic CT morphology.'}
            </p>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)', background: 'rgba(0,0,0,0.25)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-pill)' }}>
              Please supply an axial chest CT scan (DICOM, NIfTI, .npy, or grayscale CT image).
            </div>
          </div>
        )}

        <div style={{ marginTop: 'auto', paddingTop: '1rem' }}>
          <button
            type="button"
            className="btn-primary"
            style={{
              width: '100%',
              padding: '0.75rem 1rem',
              fontSize: 'var(--text-sm)',
              opacity: isValidCt ? 1 : 0.45,
              cursor: isValidCt ? 'pointer' : 'not-allowed',
            }}
            onClick={isValidCt ? onAnalyze : undefined}
            disabled={isAnalyzing || !isValidCt}
            aria-label="Run Dual-Task CT Lesion Analysis"
          >
            {isAnalyzing ? (
              <>
                <div className="spinner" aria-hidden="true" />
                <span>Running DTCR-U-Net Engine...</span>
              </>
            ) : isValidCt ? (
              <>
                <Sparkles size={16} strokeWidth={2} aria-hidden="true" />
                <span>Run AI Lesion Analysis</span>
              </>
            ) : (
              <>
                <AlertCircle size={16} strokeWidth={2} aria-hidden="true" />
                <span>Analysis Locked</span>
              </>
            )}
          </button>
        </div>
      </aside>
    );
  }

  // Analyzed State: Complete Clinical Diagnostic Report
  const {
    severity,
    severity_code,
    overall_involvement_pct,
    total_lesion_volume_cm3,
    total_lung_volume_cm3,
    clinical_finding,
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
        return <ShieldCheck size={22} strokeWidth={2} />;
      case 'mild':
        return <Activity size={22} strokeWidth={2} />;
      case 'moderate':
        return <AlertTriangle size={22} strokeWidth={2} />;
      case 'severe':
      default:
        return <AlertCircle size={22} strokeWidth={2} />;
    }
  };

  const formatMetric = (val) => {
    if (val === undefined || val === null) return '0.0';
    const num = Number(val);
    if (!isFinite(num)) return '0.0';
    return num.toFixed(1);
  };

  const displayAccuracy = ai_accuracy ?? overall_dice;
  const accuracyNum = Number(displayAccuracy) || 0;
  const isHighMatch = accuracyNum >= 70;

  return (
    <aside className="clinical-card" aria-label="Clinical Volumetry and Diagnostic Findings">
      <div className="card-heading">
        <h2 className="card-title">Diagnostic Assessment</h2>
      </div>

      {/* Severity Status Banner */}
      <div className={`severity-banner ${severity_code || 'severe'}`} role="status">
        <div className="severity-icon-badge" aria-hidden="true">
          {renderSeverityIcon()}
        </div>
        <div className="severity-text-group">
          <h3>{severity || 'Severity Assessment'}</h3>
          <p className="tabular-nums">
            {formatMetric(overall_involvement_pct)}% Total Pulmonary Involvement
          </p>
        </div>
      </div>

      {/* Ground Truth Validation Box (If uploaded) */}
      {has_ground_truth && (
        <div
          style={{
            background: 'var(--color-status-clear-bg)',
            border: '1px solid var(--color-status-clear-border)',
            borderRadius: 'var(--radius-card)',
            padding: '0.875rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.65rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--color-status-clear)', fontWeight: 600, fontSize: 'var(--text-xs)' }}>
              <CheckCircle2 size={14} strokeWidth={2} aria-hidden="true" />
              <span>Ground Truth Validation</span>
            </div>
            <span
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--color-text-secondary)',
                maxWidth: 140,
                textOverflow: 'ellipsis',
                overflow: 'hidden',
                whiteSpace: 'nowrap'
              }}
              title={ground_truth_filename}
            >
              {ground_truth_filename}
            </span>
          </div>

          {/* Unified AI Accuracy Telemetry Card */}
          <div
            style={{
              background: 'rgba(0, 0, 0, 0.25)',
              padding: '0.75rem 0.875rem',
              borderRadius: 'var(--radius-control)',
              border: '1px solid rgba(48, 209, 88, 0.2)',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.45rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                AI Model Accuracy
              </span>
              <span
                style={{
                  fontSize: 'var(--text-xs)',
                  fontWeight: 600,
                  padding: '0.15rem 0.5rem',
                  borderRadius: 'var(--radius-pill)',
                  background: isHighMatch ? 'var(--color-status-clear-bg)' : 'var(--color-status-moderate-bg)',
                  color: isHighMatch ? 'var(--color-status-clear)' : 'var(--color-status-moderate)',
                }}
              >
                {accuracy_tier || (isHighMatch ? 'Strong Match' : 'Moderate Overlap')}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.4rem' }}>
              <span
                className="tabular-nums"
                style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--color-status-clear)', lineHeight: 1 }}
              >
                {formatMetric(displayAccuracy)}%
              </span>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)' }}>
                agreement with clinician annotation
              </span>
            </div>

            {/* Visual accuracy progress bar */}
            <div
              style={{
                width: '100%',
                height: 5,
                background: 'rgba(255, 255, 255, 0.12)',
                borderRadius: 3,
                overflow: 'hidden',
                marginTop: '0.15rem'
              }}
              role="progressbar"
              aria-valuenow={accuracyNum}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, accuracyNum))}%`,
                  height: '100%',
                  background: isHighMatch ? 'var(--color-status-clear)' : 'var(--color-status-moderate)',
                  borderRadius: 3,
                  transition: 'width 0.4s var(--ease-apple)',
                }}
              />
            </div>

            {accuracy_interpretation && (
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)', marginTop: '0.2rem', lineHeight: 1.35 }}>
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
          <span className="metric-value tabular-nums" style={{ color: 'var(--color-brand-secondary)' }}>
            {formatMetric(total_lesion_volume_cm3)}
            <span className="metric-unit">cm³</span>
          </span>
        </div>

        <div className="metric-box">
          <span className="metric-label">Total Lung Field</span>
          <span className="metric-value tabular-nums" style={{ color: 'var(--color-action-blue)' }}>
            {formatMetric(total_lung_volume_cm3)}
            <span className="metric-unit">cm³</span>
          </span>
        </div>
      </div>

      {/* Action Recommendation & Clinical Correlation */}
      <div className="findings-box">
        <div className="findings-title">
          <Stethoscope size={14} strokeWidth={2} aria-hidden="true" />
          <span>Clinical Correlation</span>
        </div>
        {clinical_finding && (
          <p className="findings-body" style={{ fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '0.35rem' }}>
            {clinical_finding}
          </p>
        )}
        <p className="findings-body">{action_recommendation}</p>
      </div>

      {/* PDF Export Section */}
      <div style={{ marginTop: 'auto', paddingTop: '0.5rem' }}>
        <button
          type="button"
          className="btn-primary btn-brand-accent"
          style={{
            width: '100%',
            padding: '0.75rem 1rem',
            fontSize: 'var(--text-sm)',
            justifyContent: 'center',
          }}
          onClick={onOpenReportModal}
          aria-label="Export diagnostic findings as official PDF medical report..."
        >
          <FileText size={16} strokeWidth={2} aria-hidden="true" />
          <span>Export Medical Report (PDF)...</span>
        </button>
      </div>
    </aside>
  );
}
