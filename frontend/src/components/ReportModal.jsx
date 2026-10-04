import React, { useState } from 'react';
import { X, Download, FileText, CheckCircle2 } from 'lucide-react';

export default function ReportModal({ isOpen, onClose, analysisData, initialPatientId }) {
  const [patientId, setPatientId] = useState(initialPatientId || `PT-${Math.floor(1000 + Math.random() * 9000)}`);
  const [patientName, setPatientName] = useState('Anonymous / De-identified');
  const [age, setAge] = useState('54');
  const [gender, setGender] = useState('Male');
  const [referringPhysician, setReferringPhysician] = useState('Attending Pulmonologist');
  const [isGenerating, setIsGenerating] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);

  if (!isOpen) return null;

  const handleDownloadPdf = async () => {
    try {
      setIsGenerating(true);
      setDownloadSuccess(false);

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
        throw new Error(`Export failed with status: ${response.status}`);
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
      setTimeout(() => {
        setDownloadSuccess(false);
        onClose();
      }, 1200);
    } catch (err) {
      console.error('Failed to generate PDF:', err);
      alert('Could not generate PDF report. Please ensure backend is running.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileText size={20} color="var(--color-brand-secondary)" />
            <h3>Generate Patient Medical Report</h3>
          </div>
          <button type="button" className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>
          Enter patient information to format into the official radiology PDF diagnostic report.
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem' }}>
          <div className="form-group">
            <label className="form-label">Patient ID</label>
            <input
              type="text"
              className="form-input"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Full Name / Tag</label>
            <input
              type="text"
              className="form-input"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Age</label>
            <input
              type="text"
              className="form-input"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Gender</label>
            <select
              className="form-input"
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              style={{ background: 'var(--color-card-bg-elevated)' }}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
              <option value="N/A">Not Disclosed</option>
            </select>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Referring Physician</label>
          <input
            type="text"
            className="form-input"
            value={referringPhysician}
            onChange={(e) => setReferringPhysician(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
          <button type="button" className="btn-secondary" onClick={onClose} disabled={isGenerating}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-primary btn-brand"
            onClick={handleDownloadPdf}
            disabled={isGenerating}
          >
            {isGenerating ? (
              <>
                <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                <span>Compiling PDF...</span>
              </>
            ) : downloadSuccess ? (
              <>
                <CheckCircle2 size={16} />
                <span>Downloaded!</span>
              </>
            ) : (
              <>
                <Download size={16} />
                <span>Download PDF Report</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
