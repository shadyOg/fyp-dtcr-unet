import React, { useState, useEffect, useRef } from 'react';
import { Eye, Layers, ChevronLeft, ChevronRight, Sliders, Maximize2, UploadCloud, CheckCircle2, Play, Sparkles } from 'lucide-react';

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
  const maskInputRef = useRef(null);

  useEffect(() => {
    setCurrentSliceIdx(0);
    setZoomLevel(1);
  }, [analysisData?.metadata?.filename]);

  if (!analysisData || !analysisData.slices || analysisData.slices.length === 0) {
    return null;
  }

  const isPreview = Boolean(analysisData.is_preview);
  const slices = analysisData.slices;
  const currentSlice = slices[currentSliceIdx] || slices[0];
  const totalSlices = slices.length;
  const hasGroundTruth = Boolean(analysisData.has_ground_truth);

  const handlePrevSlice = () => {
    setCurrentSliceIdx((prev) => Math.max(0, prev - 1));
  };

  const handleNextSlice = () => {
    setCurrentSliceIdx((prev) => Math.min(totalSlices - 1, prev + 1));
  };

  const handleMaskFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadGroundTruth(e.target.files[0]);
    }
  };

  const getForegroundImage = () => {
    if (viewMode === 'gt' && currentSlice.gt_overlay_image) {
      return currentSlice.gt_overlay_image;
    }
    if (viewMode === 'composite' && currentSlice.composite_image) {
      return currentSlice.composite_image;
    }
    if (viewMode === 'contour') {
      return currentSlice.contour_image;
    }
    return currentSlice.overlay_image;
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
        return 'AI Model Lesion Prediction';
    }
  };

  const getCanvasBadgeColor = () => {
    switch (viewMode) {
      case 'gt':
        return 'rgba(23, 191, 40, 0.4)';
      case 'composite':
        return 'rgba(236, 148, 44, 0.4)';
      case 'contour':
        return 'rgba(37, 180, 248, 0.4)';
      case 'overlay':
      default:
        return 'rgba(168, 101, 200, 0.4)';
    }
  };

  return (
    <div className="viewer-card">
      <div className="viewer-header">
        <div className="viewer-header-title">
          <Layers size={18} color="var(--color-brand-secondary)" />
          <span>{isPreview ? 'CT Scan Inspection Viewer' : 'Synchronized Radiology Canvas'}</span>
          {isPreview && (
            <span style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '0.2rem 0.6rem',
              borderRadius: 'var(--radius-chip)',
              background: 'rgba(37, 180, 248, 0.15)',
              color: 'var(--color-accent-sky)',
              border: '1px solid rgba(37, 180, 248, 0.3)',
              marginLeft: '0.5rem'
            }}>
              Preview Mode (Unanalyzed)
            </span>
          )}
          {hasGroundTruth && !isPreview && (
            <span style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '0.2rem 0.6rem',
              borderRadius: 'var(--radius-chip)',
              background: 'rgba(23, 191, 40, 0.15)',
              color: 'var(--color-status-clear)',
              border: '1px solid rgba(23, 191, 40, 0.3)',
              marginLeft: '0.5rem'
            }}>
              Dice Overlap: {currentSlice.slice_dice ?? analysisData.overall_dice}%
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {!isPreview ? (
            /* View Mode Switcher */
            <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.05)', borderRadius: 'var(--radius-chip)', padding: 2 }}>
              <button
                type="button"
                className="btn-secondary"
                style={{
                  padding: '0.25rem 0.65rem',
                  fontSize: '0.75rem',
                  border: 'none',
                  background: viewMode === 'overlay' ? 'var(--color-brand-primary)' : 'transparent',
                  color: viewMode === 'overlay' ? '#fff' : 'var(--color-text-muted)',
                }}
                onClick={() => setViewMode('overlay')}
              >
                AI Prediction (Red)
              </button>

              {hasGroundTruth && (
                <>
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{
                      padding: '0.25rem 0.65rem',
                      fontSize: '0.75rem',
                      border: 'none',
                      background: viewMode === 'gt' ? 'var(--color-status-clear)' : 'transparent',
                      color: viewMode === 'gt' ? '#fff' : 'var(--color-text-muted)',
                    }}
                    onClick={() => setViewMode('gt')}
                  >
                    Ground Truth (Green)
                  </button>

                  <button
                    type="button"
                    className="btn-secondary"
                    style={{
                      padding: '0.25rem 0.65rem',
                      fontSize: '0.75rem',
                      border: 'none',
                      background: viewMode === 'composite' ? 'var(--color-status-moderate)' : 'transparent',
                      color: viewMode === 'composite' ? '#fff' : 'var(--color-text-muted)',
                    }}
                    onClick={() => setViewMode('composite')}
                  >
                    Overlap Comparison
                  </button>
                </>
              )}

              <button
                type="button"
                className="btn-secondary"
                style={{
                  padding: '0.25rem 0.65rem',
                  fontSize: '0.75rem',
                  border: 'none',
                  background: viewMode === 'contour' ? 'var(--color-action-blue)' : 'transparent',
                  color: viewMode === 'contour' ? '#fff' : 'var(--color-text-muted)',
                }}
                onClick={() => setViewMode('contour')}
              >
                Level Set Boundary
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="btn-primary"
              style={{ padding: '0.35rem 0.85rem', fontSize: '0.8rem' }}
              onClick={onAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Sparkles size={14} />
                  <span>Run AI Lesion Analysis</span>
                </>
              )}
            </button>
          )}

          <button
            type="button"
            className="btn-secondary"
            style={{ padding: '0.35rem 0.6rem' }}
            onClick={() => setZoomLevel((z) => (z === 1 ? 1.5 : 1))}
            title="Toggle 1.5x Zoom"
          >
            <Maximize2 size={14} />
          </button>
        </div>
      </div>

      {/* Side-by-Side Synchronized Canvases */}
      <div className="canvases-container" style={isPreview ? { gridTemplateColumns: '1fr' } : {}}>
        {/* Left: Original CT */}
        <div className="canvas-wrapper" style={isPreview ? { maxHeight: '520px', aspectRatio: 'unset' } : {}}>
          <div className="canvas-label-badge">
            {isPreview ? `CT Scan Inspection — Slice ${currentSliceIdx + 1} of ${totalSlices}` : 'Raw CT Scan'}
          </div>
          <img
            src={currentSlice.raw_image}
            alt={`CT Slice ${currentSliceIdx + 1}`}
            className="canvas-img"
            style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.2s ease', maxHeight: '500px' }}
          />
        </div>

        {/* Right: Selected Analysis Layer (Visible once analyzed) */}
        {!isPreview && (
          <div className="canvas-wrapper">
            <div className="canvas-label-badge" style={{ borderColor: getCanvasBadgeColor() }}>
              {getCanvasTitle()}
            </div>

            {/* Background raw slice */}
            <img
              src={currentSlice.raw_image}
              alt="Base CT"
              className="canvas-img"
              style={{ position: 'absolute', top: 0, left: 0, transform: `scale(${zoomLevel})` }}
            />

            {/* Foreground overlay layer */}
            <img
              src={getForegroundImage()}
              alt="Layer"
              className="canvas-img"
              style={{
                position: 'relative',
                opacity: overlayOpacity,
                transform: `scale(${zoomLevel})`,
                transition: 'transform 0.2s ease',
              }}
            />

            {currentSlice.has_lesion && viewMode === 'overlay' && (
              <div className="canvas-indicator-badge">
                Infection: {currentSlice.involvement_pct}%
              </div>
            )}

            {hasGroundTruth && viewMode === 'composite' && (
              <div className="canvas-indicator-badge" style={{ background: 'rgba(1, 22, 50, 0.85)', border: '1px solid rgba(255,255,255,0.2)' }}>
                🟡 Overlap | 🟢 GT Only | 🔴 AI Only
              </div>
            )}
          </div>
        )}
      </div>

      {/* Ground Truth Upload Card (Available after analysis) */}
      {!isPreview && (
        <div style={{
          background: hasGroundTruth ? 'rgba(23, 191, 40, 0.06)' : 'rgba(255, 255, 255, 0.02)',
          border: `1px dashed ${hasGroundTruth ? 'rgba(23, 191, 40, 0.35)' : 'var(--color-card-border)'}`,
          borderRadius: 'var(--radius-module)',
          padding: '0.85rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
        }}>
          <input
            type="file"
            ref={maskInputRef}
            onChange={handleMaskFileChange}
            style={{ display: 'none' }}
            accept=".npy,.png,.jpg,.jpeg,.dcm,.nii,.nii.gz"
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: 36,
              height: 36,
              borderRadius: '50%',
              background: hasGroundTruth ? 'rgba(23, 191, 40, 0.15)' : 'rgba(112, 0, 165, 0.15)',
              color: hasGroundTruth ? 'var(--color-status-clear)' : 'var(--color-brand-secondary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {hasGroundTruth ? <CheckCircle2 size={20} /> : <UploadCloud size={20} />}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--color-text-bright)' }}>
                {hasGroundTruth ? `Ground Truth Loaded: ${analysisData.ground_truth_filename}` : 'Upload Ground Truth Mask (.npy / .png / .nii)'}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)' }}>
                {hasGroundTruth
                  ? `Overall Agreement: ${analysisData.overall_dice}% Dice | ${analysisData.overall_iou}% IoU | Sensitivity: ${analysisData.overall_sensitivity}%`
                  : 'Upload the radiologist\'s ground truth mask to compare what the model predicted vs. actual truth.'}
              </div>
            </div>
          </div>

          <div>
            <button
              type="button"
              className="btn-secondary"
              style={{
                borderColor: hasGroundTruth ? 'rgba(23, 191, 40, 0.4)' : 'var(--color-border-subtle)',
                color: hasGroundTruth ? 'var(--color-status-clear)' : 'var(--color-text-main)',
              }}
              onClick={() => maskInputRef.current?.click()}
              disabled={isComparingGt}
            >
              {isComparingGt ? (
                <>
                  <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  <span>Comparing...</span>
                </>
              ) : hasGroundTruth ? (
                'Upload Different Mask'
              ) : (
                'Upload Mask & Compare'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Interactive Controls Bar */}
      <div className="viewer-controls-bar">
        {/* Opacity Blending Slider (Shown when analyzed) */}
        {!isPreview && (
          <div className="control-row">
            <div className="control-label">
              <Sliders size={16} />
              <span>Layer Blend Opacity</span>
            </div>
            <div className="slider-group">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="custom-range"
              />
              <span className="slider-val-badge">{Math.round(overlayOpacity * 100)}%</span>
            </div>
          </div>
        )}

        {/* 3D Volumetric Slice Navigator (Works in both Preview and Analyzed modes!) */}
        {totalSlices > 1 && (
          <div className="control-row" style={!isPreview ? { borderTop: '1px solid var(--color-border-subtle)', paddingTop: '0.75rem' } : {}}>
            <div className="control-label">
              <Eye size={16} />
              <span>Axial Slice ({currentSliceIdx + 1} of {totalSlices})</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: 1, maxWidth: 340 }}>
              <button
                type="button"
                className="nav-btn"
                onClick={handlePrevSlice}
                disabled={currentSliceIdx === 0}
                title="Previous Slice"
              >
                <ChevronLeft size={18} />
              </button>

              <input
                type="range"
                min="0"
                max={totalSlices - 1}
                step="1"
                value={currentSliceIdx}
                onChange={(e) => setCurrentSliceIdx(parseInt(e.target.value, 10))}
                className="custom-range"
              />

              <button
                type="button"
                className="nav-btn"
                onClick={handleNextSlice}
                disabled={currentSliceIdx === totalSlices - 1}
                title="Next Slice"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
