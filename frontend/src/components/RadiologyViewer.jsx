import React, { useState, useEffect } from 'react';
import { Eye, Layers, ChevronLeft, ChevronRight, Sliders, Maximize2, RotateCcw } from 'lucide-react';

export default function RadiologyViewer({ analysisData }) {
  const [currentSliceIdx, setCurrentSliceIdx] = useState(0);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'contour'
  const [zoomLevel, setZoomLevel] = useState(1);

  useEffect(() => {
    // Reset to slice 0 if new analysis data is received
    setCurrentSliceIdx(0);
    setZoomLevel(1);
  }, [analysisData]);

  if (!analysisData || !analysisData.slices || analysisData.slices.length === 0) {
    return null;
  }

  const slices = analysisData.slices;
  const currentSlice = slices[currentSliceIdx] || slices[0];
  const totalSlices = slices.length;

  const handlePrevSlice = () => {
    setCurrentSliceIdx((prev) => Math.max(0, prev - 1));
  };

  const handleNextSlice = () => {
    setCurrentSliceIdx((prev) => Math.min(totalSlices - 1, prev + 1));
  };

  return (
    <div className="viewer-card">
      <div className="viewer-header">
        <div className="viewer-header-title">
          <Layers size={18} color="var(--color-brand-secondary)" />
          <span>Synchronized Radiology Canvas</span>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
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
              Lesion Highlight
            </button>
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
      <div className="canvases-container">
        {/* Left: Original CT */}
        <div className="canvas-wrapper">
          <div className="canvas-label-badge">Raw CT Scan</div>
          <img
            src={currentSlice.raw_image}
            alt={`Raw CT Slice ${currentSliceIdx + 1}`}
            className="canvas-img"
            style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.2s ease' }}
          />
        </div>

        {/* Right: AI Segmented Overlay */}
        <div className="canvas-wrapper">
          <div className="canvas-label-badge" style={{ borderColor: 'rgba(168, 101, 200, 0.4)' }}>
            AI Lesion Overlay
          </div>

          {/* Background raw slice */}
          <img
            src={currentSlice.raw_image}
            alt="Base CT"
            className="canvas-img"
            style={{ position: 'absolute', top: 0, left: 0, transform: `scale(${zoomLevel})` }}
          />

          {/* Foreground highlight layer blended with opacity */}
          <img
            src={viewMode === 'overlay' ? currentSlice.overlay_image : currentSlice.contour_image}
            alt="AI Highlight"
            className="canvas-img"
            style={{
              position: 'relative',
              opacity: overlayOpacity,
              transform: `scale(${zoomLevel})`,
              transition: 'transform 0.2s ease',
            }}
          />

          {currentSlice.has_lesion && (
            <div className="canvas-indicator-badge">
              Infection: {currentSlice.involvement_pct}%
            </div>
          )}
        </div>
      </div>

      {/* Interactive Controls Bar */}
      <div className="viewer-controls-bar">
        {/* Opacity Blending Slider */}
        <div className="control-row">
          <div className="control-label">
            <Sliders size={16} />
            <span>Overlay Opacity</span>
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

        {/* 3D Volumetric Slice Navigator */}
        {totalSlices > 1 && (
          <div className="control-row" style={{ borderTop: '1px solid var(--color-border-subtle)', paddingTop: '0.75rem' }}>
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
