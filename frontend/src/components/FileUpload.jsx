import React, { useRef, useState } from 'react';
import { UploadCloud, FileImage, Layers, Play, AlertCircle, RefreshCw } from 'lucide-react';

export default function FileUpload({ onFileSelected, selectedFile, onAnalyze, isAnalyzing, error }) {
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);
  const fileInputRef = useRef(null);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (e.dataTransfer.files.length > 1) {
        console.warn('Multiple files dropped; selecting the primary file:', e.dataTransfer.files[0].name);
      }
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelected(e.target.files[0]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === undefined || bytes === null) return '0 B';
    const numBytes = Number(bytes);
    if (!isFinite(numBytes) || numBytes <= 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(numBytes) / Math.log(k));
    const safeIndex = Math.min(Math.max(0, i), sizes.length - 1);
    return `${parseFloat((numBytes / Math.pow(k, safeIndex)).toFixed(2))} ${sizes[safeIndex]}`;
  };

  const isVolumetric = selectedFile?.name?.endsWith('.nii') || selectedFile?.name?.endsWith('.nii.gz');

  return (
    <section className="upload-container" aria-label="CT Scan File Ingestion">
      {!selectedFile ? (
        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={handleKeyDown}
          tabIndex={0}
          role="button"
          aria-label="Upload CT Scan Image or Volumetric Dataset. Click or drag and drop."
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleInputChange}
            style={{ display: 'none' }}
            accept=".png,.jpg,.jpeg,.dcm,.nii,.nii.gz,.npy"
            tabIndex={-1}
          />

          <div className="upload-icon-circle" aria-hidden="true">
            <UploadCloud size={26} strokeWidth={1.8} />
          </div>

          <h2 className="upload-title">Drop CT Scan Volume or Image</h2>
          <p className="upload-subtitle">
            Upload single-slice 2D CT scans or full 3D volumetric series for AI lesion segmentation & volumetry
          </p>

          <div className="format-tags" aria-label="Supported file formats">
            <span className="format-tag">2D: .PNG / .JPG</span>
            <span className="format-tag">DICOM: .DCM</span>
            <span className="format-tag">3D NIfTI: .NII / .NII.GZ</span>
            <span className="format-tag">Numpy: .NPY</span>
          </div>
        </div>
      ) : (
        <div className="active-file-card">
          <div className="file-info-group">
            <div className="file-icon-badge" aria-hidden="true">
              {isVolumetric ? (
                <Layers size={20} strokeWidth={1.8} />
              ) : (
                <FileImage size={20} strokeWidth={1.8} />
              )}
            </div>
            <div style={{ minWidth: 0 }}>
              <div className="file-name-text" title={selectedFile.name}>
                {selectedFile.name}
              </div>
              <div className="file-size-text tabular-nums">
                {formatFileSize(selectedFile.size)} • {isVolumetric ? '3D Volumetric Series' : '2D CT Slice'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.625rem', alignItems: 'center', flexShrink: 0 }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                onFileSelected(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              disabled={isAnalyzing}
              aria-label="Change current scan file"
            >
              <RefreshCw size={14} strokeWidth={1.8} />
              <span>Change File</span>
            </button>

            <button
              type="button"
              className="btn-primary"
              onClick={onAnalyze}
              disabled={isAnalyzing}
              aria-label="Run AI lesion analysis on this CT scan"
            >
              {isAnalyzing ? (
                <>
                  <div className="spinner" aria-hidden="true" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play size={14} fill="currentColor" strokeWidth={1.8} aria-hidden="true" />
                  <span>Analyze Scan</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div
          role="alert"
          style={{
            padding: '0.75rem 1rem',
            background: 'var(--color-status-severe-bg)',
            border: '1px solid var(--color-status-severe-border)',
            borderRadius: 'var(--radius-card)',
            color: 'var(--color-status-severe)',
            fontSize: 'var(--text-sm)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.625rem',
            boxShadow: 'var(--shadow-sm)'
          }}
        >
          <AlertCircle size={18} strokeWidth={2} style={{ flexShrink: 0 }} aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}
    </section>
  );
}
