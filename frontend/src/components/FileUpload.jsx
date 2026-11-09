import React, { useRef, useState } from 'react';
import { UploadCloud, FileImage, Layers, Play, AlertCircle, RefreshCw } from 'lucide-react';

export default function FileUpload({
  onFileSelected,
  selectedFile,
  analysisData,
  onAnalyze,
  isAnalyzing,
  error
}) {
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
    e.target.value = '';
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === undefined || bytes === null) return '';
    const numBytes = Number(bytes);
    if (!isFinite(numBytes) || numBytes <= 0) return '';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(numBytes) / Math.log(k));
    const safeIndex = Math.min(Math.max(0, i), sizes.length - 1);
    return `${parseFloat((numBytes / Math.pow(k, safeIndex)).toFixed(1))} ${sizes[safeIndex]}`;
  };

  const hasScan = Boolean(selectedFile || analysisData);
  const fileName = selectedFile?.name || analysisData?.metadata?.filename || 'Loaded CT Series';
  const isVolumetric =
    fileName?.toLowerCase().endsWith('.nii') ||
    fileName?.toLowerCase().endsWith('.nii.gz') ||
    (analysisData?.total_slices && analysisData.total_slices > 1);

  const getFormatBadge = () => {
    const lowerName = (fileName || '').toLowerCase();
    if (lowerName.endsWith('.nii.gz')) return '.NII.GZ';
    if (lowerName.endsWith('.nii')) return '.NII';
    if (lowerName.endsWith('.dcm')) return '.DCM';
    if (lowerName.endsWith('.npy')) return '.NPY';
    if (lowerName.endsWith('.png')) return '.PNG';
    if (lowerName.endsWith('.jpg') || lowerName.endsWith('.jpeg')) return '.JPG';

    const fmt = (analysisData?.metadata?.format || '').toLowerCase();
    if (fmt === 'nifti') return isVolumetric ? '.NII.GZ' : '.NII';
    if (fmt === 'dicom') return '.DCM';
    if (fmt === 'numpy') return '.NPY';
    if (fmt === 'standard_image') {
      const ext = fileName && fileName.includes('.') ? fileName.split('.').pop().toUpperCase() : '2D';
      return `.${ext}`;
    }

    if (fileName && fileName.includes('.')) {
      return `.${fileName.split('.').pop().toUpperCase()}`;
    }
    return 'CT';
  };

  const totalSlices = analysisData?.total_slices || (isVolumetric ? '3D' : 1);
  const isPreview = Boolean(analysisData?.is_preview);
  const isValidCt = analysisData?.is_valid_ct !== false;
  const fileSizeStr = formatFileSize(selectedFile?.size);

  return (
    <section className="upload-container" aria-label="CT Scan File Ingestion">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleInputChange}
        style={{ display: 'none' }}
        accept=".png,.jpg,.jpeg,.dcm,.nii,.nii.gz,.npy"
        tabIndex={-1}
      />

      {!hasScan ? (
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
        /* Docked Compact 36px Capsule / Ribbon */
        <div
          className={`upload-capsule-docked ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          role="region"
          aria-label="Active scan information ribbon"
        >
          <div className="capsule-left">
            <div className="capsule-icon" aria-hidden="true">
              {isVolumetric ? <Layers size={14} strokeWidth={2} /> : <FileImage size={14} strokeWidth={2} />}
            </div>

            <span className="capsule-filename" title={fileName}>
              {fileName}
            </span>

            <span className="capsule-badge format-badge">
              {getFormatBadge()}
            </span>

            <span className="capsule-badge slice-badge tabular-nums">
              {typeof totalSlices === 'number'
                ? `${totalSlices} ${totalSlices === 1 ? 'Slice' : 'Slices'}`
                : totalSlices}
            </span>

            {fileSizeStr && (
              <span className="capsule-filesize tabular-nums">
                {fileSizeStr}
              </span>
            )}
          </div>

          <div className="capsule-right">
            {isPreview && isValidCt && (
              <button
                type="button"
                className="btn-capsule-analyze"
                onClick={onAnalyze}
                disabled={isAnalyzing}
                aria-label="Execute AI lesion inference on loaded scan"
              >
                {isAnalyzing ? (
                  <>
                    <div className="spinner" style={{ width: 11, height: 11, borderWidth: 1.5 }} aria-hidden="true" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Play size={11} fill="currentColor" strokeWidth={2} aria-hidden="true" />
                    <span>Analyze</span>
                  </>
                )}
              </button>
            )}

            <button
              type="button"
              className="btn-capsule-change"
              onClick={() => {
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                  fileInputRef.current.click();
                }
              }}
              disabled={isAnalyzing}
              aria-label="Change current CT scan file"
              title="Select or drop a different scan file"
            >
              <RefreshCw size={12} strokeWidth={1.8} aria-hidden="true" />
              <span>Change Scan</span>
            </button>
          </div>
        </div>
      )}

      {error && (
        <div
          role="alert"
          style={{
            padding: '0.625rem 0.875rem',
            background: 'var(--color-status-severe-bg)',
            border: '1px solid var(--color-status-severe-border)',
            borderRadius: 'var(--radius-card)',
            color: 'var(--color-status-severe)',
            fontSize: 'var(--text-xs)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            boxShadow: 'var(--shadow-sm)'
          }}
        >
          <AlertCircle size={16} strokeWidth={2} style={{ flexShrink: 0 }} aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}
    </section>
  );
}
