import React, { useRef, useState } from 'react';
import { UploadCloud, FileImage, Layers, Play, CheckCircle2, AlertCircle } from 'lucide-react';

export default function FileUpload({ onFileSelected, selectedFile, onAnalyze, isAnalyzing, error }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelected(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="upload-container">
      {!selectedFile ? (
        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleInputChange}
            style={{ display: 'none' }}
            accept=".png,.jpg,.jpeg,.dcm,.nii,.nii.gz,.npy"
          />
          <div className="upload-icon-circle">
            <UploadCloud size={32} />
          </div>
          <div className="upload-title">Drag & Drop CT Scan Image or Volume</div>
          <div className="upload-subtitle">
            Upload single-slice 2D CT scans or full 3D volumetric series for AI lesion segmentation
          </div>
          <div className="format-tags">
            <span className="format-tag">2D: .PNG / .JPG</span>
            <span className="format-tag">DICOM: .DCM</span>
            <span className="format-tag">3D NIfTI: .NII / .NII.GZ</span>
            <span className="format-tag">Numpy: .NPY</span>
          </div>
        </div>
      ) : (
        <div className="active-file-card">
          <div className="file-info-group">
            <div className="file-icon-badge">
              {selectedFile.name.endsWith('.nii') || selectedFile.name.endsWith('.nii.gz') ? (
                <Layers size={22} />
              ) : (
                <FileImage size={22} />
              )}
            </div>
            <div>
              <div className="file-name-text">{selectedFile.name}</div>
              <div className="file-size-text">{formatFileSize(selectedFile.size)}</div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                onFileSelected(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              disabled={isAnalyzing}
            >
              Change File
            </button>
            <button
              type="button"
              className="btn-primary"
              onClick={onAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play size={16} fill="currentColor" />
                  <span>Analyze Scan</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div style={{
          marginTop: '0.75rem',
          padding: '0.75rem 1rem',
          background: 'var(--color-status-severe-bg)',
          border: '1px solid rgba(229, 35, 35, 0.3)',
          borderRadius: 'var(--radius-module)',
          color: 'var(--color-status-severe)',
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
