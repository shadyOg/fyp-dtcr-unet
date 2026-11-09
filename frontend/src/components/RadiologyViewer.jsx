import React, { useState, useEffect, useRef } from 'react';
import {
  Layers,
  ChevronLeft,
  ChevronRight,
  Sliders,
  Maximize2,
  Minimize2,
  UploadCloud,
  CheckCircle2,
  Sparkles,
  AlertCircle,
  Eye,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';

export default function RadiologyViewer({
  analysisData,
  onUploadGroundTruth,
  isComparingGt,
  onAnalyze,
  isAnalyzing
}) {
  const [currentSliceIdx, setCurrentSliceIdx] = useState(0);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'gt' | 'composite' | 'contour'
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isPlayingCine, setIsPlayingCine] = useState(false);
  const maskInputRef = useRef(null);
  const viewerContainerRef = useRef(null);

  // Reset slice index & zoom when a new file is loaded
  useEffect(() => {
    setCurrentSliceIdx(0);
    setZoomLevel(1);
    setIsPlayingCine(false);
  }, [analysisData?.metadata?.filename]);

  const slices = analysisData?.slices || [];
  const totalSlices = slices.length;
  const isPreview = Boolean(analysisData?.is_preview);
  const isValidCt = analysisData?.is_valid_ct !== false;
  const hasGroundTruth = Boolean(analysisData?.has_ground_truth);
  const currentSlice = slices[currentSliceIdx] || slices[0];

  // Cine-loop playback for multi-slice 3D CT volumes (pauses if modal is open)
  useEffect(() => {
    if (!isPlayingCine || totalSlices <= 1) return;
    const interval = setInterval(() => {
      const isModalOpen = Boolean(
        document.querySelector('[role="dialog"]') ||
        document.querySelector('.modal-overlay')
      );
      if (isModalOpen) return;
      setCurrentSliceIdx((prev) => (prev + 1) % totalSlices);
    }, 180);
    return () => clearInterval(interval);
  }, [isPlayingCine, totalSlices]);

  // Keyboard navigation for slice stepping (skips if inside active input or modal dialog)
  useEffect(() => {
    const handleKeyDown = (e) => {
      const activeEl = document.activeElement;
      const targetTag = e.target?.tagName?.toUpperCase();
      const activeTag = activeEl?.tagName?.toUpperCase();
      const inputTags = ['INPUT', 'SELECT', 'TEXTAREA'];

      const isInputActive =
        inputTags.includes(targetTag) ||
        inputTags.includes(activeTag) ||
        Boolean(e.target?.isContentEditable) ||
        Boolean(activeEl?.isContentEditable);

      const isModalOpen = Boolean(
        document.querySelector('[role="dialog"]') ||
        document.querySelector('.modal-overlay')
      );

      if (isInputActive || isModalOpen) return;
      if (totalSlices <= 1) return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setIsPlayingCine(false);
        setCurrentSliceIdx((prev) => Math.max(0, prev - 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setIsPlayingCine(false);
        setCurrentSliceIdx((prev) => Math.min(totalSlices - 1, prev + 1));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [totalSlices]);

  if (!analysisData || !slices || slices.length === 0) {
    return null;
  }

  const handlePrevSlice = () => {
    setIsPlayingCine(false);
    setCurrentSliceIdx((prev) => Math.max(0, prev - 1));
  };

  const handleNextSlice = () => {
    setIsPlayingCine(false);
    setCurrentSliceIdx((prev) => Math.min(totalSlices - 1, prev + 1));
  };

  const handleMaskFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadGroundTruth(e.target.files[0]);
    }
    // Reset file input value so selecting the same mask triggers change cleanly
    e.target.value = '';
  };

  const getForegroundImage = () => {
    if (!currentSlice) return '';
    if (viewMode === 'gt' && currentSlice.gt_overlay_image) {
      return currentSlice.gt_overlay_image;
    }
    if (viewMode === 'composite' && currentSlice.composite_image) {
      return currentSlice.composite_image;
    }
    if (viewMode === 'contour' && currentSlice.contour_image) {
      return currentSlice.contour_image;
    }
    return currentSlice.overlay_image || currentSlice.raw_image;
  };

  const getCanvasTitle = () => {
    switch (viewMode) {
      case 'gt':
        return 'Ground Truth Annotation (Doctor)';
      case 'composite':
        return 'Direct Overlap Comparison';
      case 'contour':
        return 'Level Set Boundary Contour';
      case 'overlay':
      default:
        return 'AI Lesion Prediction (Overlay)';
    }
  };

  const formatMetric = (val) => {
    if (val === undefined || val === null) return '0.0';
    const num = Number(val);
    if (!isFinite(num)) return '0.0';
    return num.toFixed(1);
  };

  const displayAccuracy = currentSlice?.slice_ai_accuracy ?? currentSlice?.slice_dice ?? analysisData.ai_accuracy ?? analysisData.overall_dice;

  return (
    <section
      ref={viewerContainerRef}
      className="viewer-card"
      aria-label="Synchronized Radiology Image Viewer"
    >
      {/* Viewer Header Toolbar */}
      <div className="viewer-header">
        <div className="viewer-header-left">
          <div className="viewer-header-title">
            <Layers size={18} color="var(--color-brand-secondary)" strokeWidth={2} aria-hidden="true" />
            <span>{isPreview ? 'CT Scan Inspection' : 'Synchronized Radiology Canvas'}</span>
          </div>

          {isPreview && isValidCt && (
            <span
              style={{
                fontSize: 'var(--text-xs)',
                fontWeight: 600,
                padding: '0.2rem 0.6rem',
                borderRadius: 'var(--radius-pill)',
                background: 'var(--color-action-tint)',
                color: 'var(--color-action-blue)',
                border: '1px solid rgba(10, 132, 255, 0.28)',
              }}
            >
              Preview Mode (Unanalyzed)
            </span>
          )}

          {isPreview && !isValidCt && (
            <span
              style={{
                fontSize: 'var(--text-xs)',
                fontWeight: 600,
                padding: '0.2rem 0.6rem',
                borderRadius: 'var(--radius-pill)',
                background: 'var(--color-status-severe-bg)',
                color: 'var(--color-status-severe)',
                border: '1px solid var(--color-status-severe-border)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.35rem'
              }}
            >
              <AlertCircle size={12} strokeWidth={2} aria-hidden="true" />
              Invalid Image Input
            </span>
          )}

          {hasGroundTruth && !isPreview && (
            <span
              style={{
                fontSize: 'var(--text-xs)',
                fontWeight: 600,
                padding: '0.2rem 0.6rem',
                borderRadius: 'var(--radius-pill)',
                background: 'var(--color-status-clear-bg)',
                color: 'var(--color-status-clear)',
                border: '1px solid var(--color-status-clear-border)',
              }}
            >
              Accuracy: {formatMetric(displayAccuracy)}%
            </span>
          )}
        </div>

        <div className="viewer-header-right">
          {/* Segmented View Mode Switcher (Analyzed Mode) */}
          {!isPreview ? (
            <div
              className="segmented-control"
              role="radiogroup"
              aria-label="Radiology Layer Presentation"
            >
              <button
                type="button"
                className={`segmented-option ${viewMode === 'overlay' ? 'active' : ''}`}
                onClick={() => setViewMode('overlay')}
                role="radio"
                aria-checked={viewMode === 'overlay'}
              >
                <span className="segmented-dot" style={{ backgroundColor: '#EF4444' }} />
                <span>AI Prediction</span>
              </button>

              {hasGroundTruth && (
                <>
                  <button
                    type="button"
                    className={`segmented-option ${viewMode === 'gt' ? 'active' : ''}`}
                    onClick={() => setViewMode('gt')}
                    role="radio"
                    aria-checked={viewMode === 'gt'}
                  >
                    <span className="segmented-dot" style={{ backgroundColor: '#10B981' }} />
                    <span>Ground Truth</span>
                  </button>

                  <button
                    type="button"
                    className={`segmented-option ${viewMode === 'composite' ? 'active' : ''}`}
                    onClick={() => setViewMode('composite')}
                    role="radio"
                    aria-checked={viewMode === 'composite'}
                  >
                    <span className="segmented-dot" style={{ backgroundColor: '#F59E0B' }} />
                    <span>Overlap</span>
                  </button>
                </>
              )}

              <button
                type="button"
                className={`segmented-option ${viewMode === 'contour' ? 'active' : ''}`}
                onClick={() => setViewMode('contour')}
                role="radio"
                aria-checked={viewMode === 'contour'}
              >
                <span className="segmented-dot" style={{ backgroundColor: '#0EA5E9' }} />
                <span>Boundary</span>
              </button>
            </div>
          ) : isValidCt ? (
            <button
              type="button"
              className="btn-primary"
              onClick={onAnalyze}
              disabled={isAnalyzing}
              aria-label="Run Dual-Task CT Lesion Analysis"
            >
              {isAnalyzing ? (
                <>
                  <div className="spinner" aria-hidden="true" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Sparkles size={14} strokeWidth={2} aria-hidden="true" />
                  <span>Run AI Lesion Analysis</span>
                </>
              )}
            </button>
          ) : (
            <button
              type="button"
              className="btn-secondary"
              disabled={true}
              style={{
                opacity: 0.6,
                color: 'var(--color-status-severe)',
                borderColor: 'var(--color-status-severe-border)',
                cursor: 'not-allowed'
              }}
              title={analysisData.validation_reason || 'Invalid scan morphology.'}
            >
              <AlertCircle size={14} strokeWidth={2} aria-hidden="true" />
              <span>Analysis Locked</span>
            </button>
          )}

          {/* 1.5x Zoom Inspection Toggle */}
          <button
            type="button"
            className="btn-icon"
            onClick={() => setZoomLevel((z) => (z === 1 ? 1.5 : 1))}
            title={zoomLevel === 1 ? 'Zoom In (1.5x)' : 'Reset Zoom'}
            aria-label={zoomLevel === 1 ? 'Magnify scan to 1.5x' : 'Reset magnification to default'}
          >
            {zoomLevel === 1 ? (
              <Maximize2 size={15} strokeWidth={1.8} aria-hidden="true" />
            ) : (
              <Minimize2 size={15} strokeWidth={1.8} aria-hidden="true" />
            )}
          </button>
        </div>
      </div>

      {/* Non-CT Guardrail Warning Banner */}
      {isPreview && !isValidCt && (
        <div
          role="alert"
          style={{
            padding: '1rem 1.25rem',
            background: 'var(--color-status-severe-bg)',
            border: '1px solid var(--color-status-severe-border)',
            borderRadius: 'var(--radius-card)',
            color: 'var(--color-text-primary)',
            fontSize: 'var(--text-sm)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.875rem'
          }}
        >
          <AlertCircle
            size={22}
            color="var(--color-status-severe)"
            strokeWidth={2}
            style={{ flexShrink: 0, marginTop: '0.1rem' }}
            aria-hidden="true"
          />
          <div>
            <div style={{ fontWeight: 600, color: 'var(--color-status-severe)', marginBottom: '0.2rem' }}>
              Non-Thoracic or Non-Medical Image Detected
            </div>
            <div style={{ color: 'var(--color-text-primary)', lineHeight: 1.45 }}>
              {analysisData.validation_reason || 'The uploaded file does not match thoracic CT scan morphology.'}
            </div>
            <div style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--text-xs)', marginTop: '0.35rem' }}>
              Diagnostic lesion segmentation is locked to prevent clinical errors. Please upload an axial chest CT scan (.dcm, .nii, .npy, or grayscale CT slice).
            </div>
          </div>
        </div>
      )}

      {/* Synchronized Side-by-Side Dual Viewport Canvases */}
      <div className="canvases-container" style={isPreview ? { gridTemplateColumns: '1fr' } : {}}>
        {/* Left Viewport: Raw Base CT Scan */}
        <div
          className="canvas-wrapper"
          style={isPreview ? { maxHeight: '520px', aspectRatio: 'unset' } : {}}
        >
          <div className="canvas-label-badge">
            <Eye size={12} strokeWidth={2} aria-hidden="true" />
            <span>
              {isPreview
                ? `CT Scan Inspection — Slice ${currentSliceIdx + 1} of ${totalSlices}`
                : `Raw CT Scan (Slice ${currentSliceIdx + 1}/${totalSlices})`}
            </span>
          </div>

          <img
            src={currentSlice?.raw_image}
            alt={`Raw CT Scan slice ${currentSliceIdx + 1} of ${totalSlices}`}
            className="canvas-img"
            style={{
              transform: `scale(${zoomLevel})`,
              transformOrigin: 'center center',
              transition: 'transform var(--duration-fast) var(--ease-apple)',
              willChange: 'transform',
              maxHeight: isPreview ? '520px' : '100%'
            }}
          />
        </div>

        {/* Right Viewport: AI Prediction / Doctor GT / Overlap Layer (Shown once analyzed) */}
        {!isPreview && (
          <div className="canvas-wrapper">
            <div className="canvas-label-badge">
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor:
                    viewMode === 'gt'
                      ? '#10B981'
                      : viewMode === 'composite'
                        ? '#F59E0B'
                        : viewMode === 'contour'
                          ? '#0EA5E9'
                          : '#EF4444'
                }}
              />
              <span>{getCanvasTitle()}</span>
            </div>

            {/* Base CT image underlay - purely decorative background for overlay blending */}
            <img
              src={currentSlice?.raw_image}
              alt=""
              aria-hidden="true"
              className="canvas-img"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'center center',
                transition: 'transform var(--duration-fast) var(--ease-apple)',
                willChange: 'transform'
              }}
            />

            {/* Diagnostic Foreground Overlay with controlled opacity */}
            <img
              src={getForegroundImage()}
              alt={`${getCanvasTitle()} for slice ${currentSliceIdx + 1}`}
              className="canvas-img"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                opacity: overlayOpacity,
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'center center',
                transition: 'opacity var(--duration-fast) ease, transform var(--duration-fast) var(--ease-apple)',
                willChange: 'transform'
              }}
            />

            {currentSlice?.has_lesion && viewMode === 'overlay' && (
              <div
                className="canvas-indicator-badge tabular-nums"
                style={{ background: 'rgba(239, 68, 68, 0.92)' }}
              >
                Lesion Involvement: {formatMetric(currentSlice.involvement_pct)}%
              </div>
            )}

            {hasGroundTruth && viewMode === 'composite' && (
              <div
                className="canvas-indicator-badge"
                style={{
                  background: 'rgba(17, 22, 34, 0.95)',
                  borderColor: 'rgba(255, 255, 255, 0.18)'
                }}
              >
                🟡 Overlap | 🟢 Doctor GT | 🔴 AI Prediction
              </div>
            )}
          </div>
        )}
      </div>

      {/* Ground Truth Doctor Annotation Card (Available after analysis) */}
      {!isPreview && (
        <div className={`gt-panel ${hasGroundTruth ? 'loaded' : ''}`}>
          <input
            type="file"
            ref={maskInputRef}
            onChange={handleMaskFileChange}
            style={{ display: 'none' }}
            accept=".npy,.png,.jpg,.jpeg,.dcm,.nii,.nii.gz"
            tabIndex={-1}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
            <div className="gt-icon-circle" aria-hidden="true">
              {hasGroundTruth ? (
                <CheckCircle2 size={18} strokeWidth={2} />
              ) : (
                <UploadCloud size={18} strokeWidth={2} />
              )}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 'var(--text-sm)', color: 'var(--color-text-primary)' }}>
                {hasGroundTruth
                  ? `Ground Truth Loaded: ${analysisData.ground_truth_filename}`
                  : 'Upload Ground Truth Annotation (.npy / .png / .nii)'}
              </div>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)', marginTop: '0.15rem' }}>
                {hasGroundTruth
                  ? `Overall Agreement: ${formatMetric(analysisData.overall_dice)}% Dice | ${formatMetric(analysisData.overall_iou)}% IoU | Sensitivity: ${formatMetric(analysisData.overall_sensitivity)}%`
                  : 'Compare the deep learning model predictions against official radiologist contour masks.'}
              </div>
            </div>
          </div>

          <button
            type="button"
            className="btn-secondary"
            onClick={() => maskInputRef.current?.click()}
            disabled={isComparingGt}
            aria-label="Upload ground truth mask to compare with AI predictions"
          >
            {isComparingGt ? (
              <>
                <div className="spinner" aria-hidden="true" />
                <span>Comparing Mask...</span>
              </>
            ) : hasGroundTruth ? (
              'Replace Mask'
            ) : (
              'Upload Mask & Compare'
            )}
          </button>
        </div>
      )}

      {/* Interactive Telemetry & Navigation Controls Bar */}
      <div className="viewer-controls-bar">
        {/* Layer Opacity Blending Slider (Shown when analyzed) */}
        {!isPreview && (
          <div className="control-row">
            <label htmlFor="overlay-opacity-slider" className="control-label">
              <Sliders size={15} strokeWidth={1.8} aria-hidden="true" />
              <span>Layer Blend Opacity</span>
            </label>

            <div className="slider-group">
              <input
                id="overlay-opacity-slider"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="apple-range"
                aria-label="Adjust AI overlay transparency"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(overlayOpacity * 100)}
              />
              <span className="slider-val-badge tabular-nums">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>
          </div>
        )}

        {/* 3D Volumetric Axial Slice Navigator */}
        {totalSlices > 1 && (
          <div
            className="control-row"
            style={!isPreview ? { borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' } : {}}
          >
            <div className="control-label">
              <Eye size={15} strokeWidth={1.8} aria-hidden="true" />
              <span>
                Axial Slice ({currentSliceIdx + 1} of {totalSlices})
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', flex: 1, maxWidth: 420 }}>
              {/* Play/Pause Cine Loop */}
              <button
                type="button"
                className="btn-icon"
                onClick={() => setIsPlayingCine(!isPlayingCine)}
                title={isPlayingCine ? 'Pause Cine Loop' : 'Play 3D Cine Loop'}
                aria-label={isPlayingCine ? 'Pause slice animation' : 'Play cine slice sequence'}
              >
                {isPlayingCine ? (
                  <Pause size={14} strokeWidth={2} aria-hidden="true" />
                ) : (
                  <Play size={14} fill="currentColor" strokeWidth={1.8} aria-hidden="true" />
                )}
              </button>

              {/* Prev Slice */}
              <button
                type="button"
                className="btn-icon"
                onClick={handlePrevSlice}
                disabled={currentSliceIdx === 0}
                title="Previous Axial Slice (Left Arrow)"
                aria-label="Go to previous axial CT slice"
              >
                <ChevronLeft size={16} strokeWidth={2} aria-hidden="true" />
              </button>

              {/* Slice Range Scrubber */}
              <input
                type="range"
                min="0"
                max={totalSlices - 1}
                step="1"
                value={currentSliceIdx}
                onChange={(e) => {
                  setIsPlayingCine(false);
                  setCurrentSliceIdx(parseInt(e.target.value, 10));
                }}
                className="apple-range"
                aria-label="Navigate through axial CT slices"
                aria-valuemin={1}
                aria-valuemax={totalSlices}
                aria-valuenow={currentSliceIdx + 1}
              />

              {/* Next Slice */}
              <button
                type="button"
                className="btn-icon"
                onClick={handleNextSlice}
                disabled={currentSliceIdx === totalSlices - 1}
                title="Next Axial Slice (Right Arrow)"
                aria-label="Go to next axial CT slice"
              >
                <ChevronRight size={16} strokeWidth={2} aria-hidden="true" />
              </button>

              <span className="slider-val-badge tabular-nums">
                {currentSliceIdx + 1}/{totalSlices}
              </span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
