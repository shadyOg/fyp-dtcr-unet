import React, { useState, useEffect, useRef } from 'react';
import { X, Download, FileText, CheckCircle2, AlertCircle } from 'lucide-react';

export default function ReportModal({ isOpen, onClose, analysisData, initialPatientId }) {
  const [patientId, setPatientId] = useState(initialPatientId || `PT-${Math.floor(1000 + Math.random() * 9000)}`);
  const [patientName, setPatientName] = useState('Anonymous / De-identified');
  const [age, setAge] = useState('54');
  const [gender, setGender] = useState('Male');
  const [referringPhysician, setReferringPhysician] = useState('Attending Pulmonologist');
  const [isGenerating, setIsGenerating] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const modalRef = useRef(null);
  const firstInputRef = useRef(null);
  const previousActiveElementRef = useRef(null);
  const downloadTimerRef = useRef(null);

  // Sync initial patient ID whenever modal opens or initialPatientId prop changes
  useEffect(() => {
    if (isOpen && initialPatientId) {
      setPatientId(initialPatientId);
    }
  }, [isOpen, initialPatientId]);

  // Clean up any pending download timers on unmount
  useEffect(() => {
    return () => {
      if (downloadTimerRef.current) {
        clearTimeout(downloadTimerRef.current);
      }
    };
  }, []);

  // Handle focus trapping, Escape key dismissal, and focus restoration
  useEffect(() => {
    if (!isOpen) return;

    setErrorMessage(null);
    previousActiveElementRef.current = document.activeElement;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        if (!isGenerating) {
          onClose();
        }
        return;
      }

      if (e.key === 'Tab') {
        if (!modalRef.current) return;
        const focusableElements = modalRef.current.querySelectorAll(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        if (focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        // Ensure focus stays trapped inside modal even if focus moved outside
        if (!modalRef.current.contains(document.activeElement)) {
          e.preventDefault();
          firstElement.focus();
          return;
        }

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    const timer = setTimeout(() => {
      firstInputRef.current?.focus();
    }, 50);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      clearTimeout(timer);
      if (
        previousActiveElementRef.current &&
        typeof previousActiveElementRef.current.focus === 'function' &&
        document.contains(previousActiveElementRef.current)
      ) {
        previousActiveElementRef.current.focus();
      }
    };
  }, [isOpen, onClose, isGenerating]);

  if (!isOpen) return null;

  const handleDownloadPdf = async () => {
    try {
      setIsGenerating(true);
      setDownloadSuccess(false);
      setErrorMessage(null);

      const response = await fetch('/api/export-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_data: analysisData,
          patient_info: {
            patient_id: patientId,
            patient_name: patientName,
            age: age,
            gender: gender,
            referring_physician: referringPhysician,
          },
        }),
      });

      if (!response.ok) {
        let errDetail = `Export failed with status: ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson && errJson.detail) errDetail = errJson.detail;
          else if (errJson && errJson.error) errDetail = errJson.error;
        } catch (_) { }
        throw new Error(errDetail);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `DTCR_Diagnostic_Report_${patientId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setDownloadSuccess(true);
      downloadTimerRef.current = setTimeout(() => {
        setDownloadSuccess(false);
        onClose();
      }, 1400);
    } catch (err) {
      console.error('Failed to generate PDF:', err);
      setErrorMessage(
        err.message || 'Could not generate PDF report. Please verify that the backend inference engine is active and reachable.'
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div
      className="modal-overlay"
      onClick={isGenerating ? undefined : onClose}
      role="presentation"
    >
      <div
        className="modal-content"
        ref={modalRef}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="report-modal-title"
      >
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
            <FileText size={20} color="var(--color-brand-secondary)" strokeWidth={2} aria-hidden="true" />
            <h3 id="report-modal-title">Generate Patient Diagnostic Report</h3>
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onClose}
            disabled={isGenerating}
            aria-label="Close dialog (Escape)"
          >
            <X size={16} strokeWidth={2} aria-hidden="true" />
          </button>
        </div>

        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
          Enter patient clinical metadata to format into the official DTCR-U-Net diagnostic PDF summary report.
        </p>

        {errorMessage && (
          <div
            role="alert"
            style={{
              padding: '0.75rem 0.875rem',
              background: 'var(--color-status-severe-bg)',
              border: '1px solid var(--color-status-severe-border)',
              borderRadius: 'var(--radius-control)',
              color: 'var(--color-status-severe)',
              fontSize: 'var(--text-xs)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.625rem',
            }}
          >
            <AlertCircle size={16} strokeWidth={2} style={{ flexShrink: 0 }} aria-hidden="true" />
            <span style={{ lineHeight: 1.4 }}>{errorMessage}</span>
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleDownloadPdf();
          }}
          style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
        >
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.875rem' }}>
            <div className="form-group">
              <label htmlFor="modal-patient-id" className="form-label">
                Patient Record ID
              </label>
              <input
                id="modal-patient-id"
                ref={firstInputRef}
                type="text"
                className="form-input tabular-nums"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="modal-patient-name" className="form-label">
                Full Name / De-identified
              </label>
              <input
                id="modal-patient-name"
                type="text"
                className="form-input"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="modal-patient-age" className="form-label">
                Age
              </label>
              <input
                id="modal-patient-age"
                type="text"
                className="form-input tabular-nums"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="modal-patient-gender" className="form-label">
                Gender
              </label>
              <select
                id="modal-patient-gender"
                className="form-input"
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                style={{ cursor: 'pointer' }}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
                <option value="N/A">Not Disclosed</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="modal-referring-physician" className="form-label">
              Referring Physician / Pulmonologist
            </label>
            <input
              id="modal-referring-physician"
              type="text"
              className="form-input"
              value={referringPhysician}
              onChange={(e) => setReferringPhysician(e.target.value)}
              required
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.625rem', marginTop: '0.5rem' }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={onClose}
              disabled={isGenerating}
            >
              Cancel
            </button>

            <button
              type="submit"
              className="btn-primary btn-brand-accent"
              disabled={isGenerating}
              aria-label="Generate and download medical PDF diagnostic report"
            >
              {isGenerating ? (
                <>
                  <div className="spinner" aria-hidden="true" />
                  <span>Compiling PDF Report...</span>
                </>
              ) : downloadSuccess ? (
                <>
                  <CheckCircle2 size={16} strokeWidth={2} aria-hidden="true" />
                  <span>Downloaded!</span>
                </>
              ) : (
                <>
                  <Download size={16} strokeWidth={2} aria-hidden="true" />
                  <span>Download PDF Report</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
