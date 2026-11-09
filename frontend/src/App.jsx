import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import RadiologyViewer from './components/RadiologyViewer';
import ClinicalCard from './components/ClinicalCard';
import ReportModal from './components/ReportModal';

export default function App() {
  const [patientId, setPatientId] = useState('PT-2026-8819');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isComparingGt, setIsComparingGt] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [error, setError] = useState(null);
  const [isServerReady, setIsServerReady] = useState(false);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);

  // Check backend engine health periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          setIsServerReady(true);
        } else {
          setIsServerReady(false);
        }
      } catch (err) {
        setIsServerReady(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 6000);
    return () => clearInterval(interval);
  }, []);

  // Ingest scan file and retrieve immediate preview & metadata
  const handleFileSelected = useCallback(async (file) => {
    setSelectedFile(file);
    setError(null);
    setAnalysisData(null);

    if (!file) return;

    if (file.size === 0) {
      setError('The selected file is empty (0 bytes). Please upload a valid CT scan image or volumetric series.');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/preview', {
        method: 'POST',
        body: formData,
      });
      const json = await res.json();
      if (res.ok && json.success) {
        setAnalysisData(json.data);
      } else {
        setError(json.error || 'Failed to parse CT scan preview. Please verify file format.');
      }
    } catch (err) {
      console.error('Preview load error:', err);
      setError('Unable to connect to the DTCR-U-Net preview engine. Please verify backend service.');
    }
  }, []);

  // Execute DTCR-U-Net dual-task inference
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    if (selectedFile.size === 0) {
      setError('Cannot analyze an empty file (0 bytes). Please upload a valid CT scan.');
      return;
    }

    try {
      setIsAnalyzing(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('threshold', '0.5');
      formData.append('fuse_dual_task', 'true');

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      const json = await response.json();

      if (!response.ok || !json.success) {
        throw new Error(json.error || `Server returned error status ${response.status}`);
      }

      setAnalysisData(json.data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'Failed to complete scan analysis. Please verify scan format.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile]);

  // Upload and compare clinician ground truth mask
  const handleUploadGroundTruth = useCallback(async (maskFile) => {
    if (!maskFile || !analysisData) return;

    try {
      setIsComparingGt(true);
      setError(null);

      const formData = new FormData();
      formData.append('mask_file', maskFile);
      formData.append('analysis_json', JSON.stringify(analysisData));

      const response = await fetch('/api/compare-ground-truth', {
        method: 'POST',
        body: formData,
      });

      const json = await response.json();

      if (!response.ok || !json.success) {
        throw new Error(json.error || `Failed to compare mask with status ${response.status}`);
      }

      setAnalysisData(json.data);
    } catch (err) {
      console.error('Ground truth comparison error:', err);
      setError(err.message || 'Failed to compare ground truth mask. Please verify mask format.');
    } finally {
      setIsComparingGt(false);
    }
  }, [analysisData]);

  return (
    <div className="app-container">
      {/* Header Bar */}
      <Header
        patientId={patientId}
        setPatientId={setPatientId}
        isServerReady={isServerReady}
      />

      {/* Main Workstation Dashboard */}
      <main className="dashboard-grid">
        {/* Left Section: File Upload & Synchronized Canvas Viewer */}
        <section className="viewer-panel" aria-label="Radiology Viewer Section">
          <FileUpload
            onFileSelected={handleFileSelected}
            selectedFile={selectedFile}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            error={error}
          />

          {analysisData && (
            <RadiologyViewer
              analysisData={analysisData}
              onUploadGroundTruth={handleUploadGroundTruth}
              isComparingGt={isComparingGt}
              onAnalyze={handleAnalyze}
              isAnalyzing={isAnalyzing}
            />
          )}
        </section>

        {/* Right Section: Clinical Summary & Actionable Volumetry */}
        <ClinicalCard
          analysisData={analysisData}
          onOpenReportModal={() => setIsReportModalOpen(true)}
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
        />
      </main>

      {/* Medical PDF Diagnostic Report Modal Sheet */}
      <ReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        analysisData={analysisData}
        initialPatientId={patientId}
      />

      {/* Workstation System Footer */}
      <footer className="workstation-footer">
        DTCR-U-Net Clinical Workstation • Dual-Task Consistency & Multi-Scale Attention AI • Version 2.0 (Clinical Release)
      </footer>
    </div>
  );
}
