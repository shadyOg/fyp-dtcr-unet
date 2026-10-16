import React, { useState, useEffect } from 'react';
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

  // Check backend health on load
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

  const handleFileSelected = async (file) => {
    setSelectedFile(file);
    setError(null);
    setAnalysisData(null);

    if (!file) return;

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
      }
    } catch (err) {
      console.error('Preview load error:', err);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

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
  };

  const handleUploadGroundTruth = async (maskFile) => {
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
  };

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
        <div className="viewer-panel">
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
        </div>

        {/* Right Section: Clinical Summary & Actionable Findings */}
        <aside className="clinical-panel">
          <ClinicalCard
            analysisData={analysisData}
            onOpenReportModal={() => setIsReportModalOpen(true)}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
          />
        </aside>
      </main>

      {/* Medical PDF Diagnostic Report Modal */}
      <ReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        analysisData={analysisData}
        initialPatientId={patientId}
      />
    </div>
  );
}
