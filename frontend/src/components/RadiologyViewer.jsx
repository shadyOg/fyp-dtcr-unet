import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  Columns,
  Split,
  SunMedium,
  HelpCircle,
  Maximize
} from 'lucide-react';

const WL_PRESETS = {
  default: {
    id: 'default',
    label: 'Default',
    name: 'Standard CT Dynamic Range',
    filter: 'none',
  },
  lung: {
    id: 'lung',
    label: 'Lung Window',
    name: 'Lung Parenchyma (WW 1500 / WL -600)',
    filter: 'contrast(1.4) brightness(1.15)',
  },
  mediastinum: {
    id: 'mediastinum',
    label: 'Soft Tissue',
    name: 'Mediastinum & Soft Tissue (WW 350 / WL 40)',
    filter: 'contrast(1.65) brightness(0.85)',
  },
  bone: {
    id: 'bone',
    label: 'Bone / Hi-Con',
    name: 'Bone & Dense Infiltrates (WW 2000 / WL 500)',
    filter: 'contrast(2.2) brightness(1.25)',
  },
};

export default function RadiologyViewer({
  analysisData,
  onUploadGroundTruth,
  isComparingGt,
  onAnalyze,
  isAnalyzing,
  onOpenShortcuts,
}) {
  const [currentSliceIdx, setCurrentSliceIdx] = useState(0);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'gt' | 'composite' | 'contour'
  const [displayLayout, setDisplayLayout] = useState('dual'); // 'dual' | 'curtain'
  const [wlPreset, setWlPreset] = useState('default'); // 'default' | 'lung' | 'mediastinum' | 'bone'
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isPlayingCine, setIsPlayingCine] = useState(false);
  const [isTheaterMode, setIsTheaterMode] = useState(false);
  const [splitPos, setSplitPos] = useState(50);
  const [isDraggingCurtain, setIsDraggingCurtain] = useState(false);
  const [isScrubbingCurve, setIsScrubbingCurve] = useState(false);

  const maskInputRef = useRef(null);
  const viewerContainerRef = useRef(null);
  const canvasesRef = useRef(null);
  const curtainRef = useRef(null);

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

  // Toggle Diagnostic Theater / Fullscreen with browser API synchronization
  const toggleTheaterMode = useCallback(() => {
    setIsTheaterMode((prev) => {
      const next = !prev;
      if (next) {
        try {
          if (!document.fullscreenElement && viewerContainerRef.current?.requestFullscreen) {
            viewerContainerRef.current.requestFullscreen().catch(() => { });
          }
        } catch (err) { }
      } else {
        try {
          if (document.fullscreenElement && document.exitFullscreen) {
            document.exitFullscreen().catch(() => { });
          }
        } catch (err) { }
      }
      return next;
    });
  }, []);

  // Sync state if user exits browser native fullscreen (via Esc or OS gesture)
  useEffect(() => {
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement && isTheaterMode) {
        setIsTheaterMode(false);
      }
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, [isTheaterMode]);

  // Lock body overflow in Diagnostic Theater mode
  useEffect(() => {
    if (isTheaterMode) {
      const prevOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = prevOverflow;
      };
    }
  }, [isTheaterMode]);

  // Mouse wheel slice scrubbing on either canvas / viewer with smooth decay
  useEffect(() => {
    const el = canvasesRef.current;
    if (!el || totalSlices <= 1) return;

    let accumulatedDelta = 0;
    let decayTimer = null;
    const WHEEL_THRESHOLD = 25;

    const handleWheel = (e) => {
      if (e.ctrlKey) return; // Allow browser zoom
      e.preventDefault();
      setIsPlayingCine(false);

      if (decayTimer) clearTimeout(decayTimer);
      decayTimer = setTimeout(() => {
        accumulatedDelta = 0;
      }, 150);

      accumulatedDelta += e.deltaY;
      if (Math.abs(accumulatedDelta) >= WHEEL_THRESHOLD) {
        const direction = accumulatedDelta > 0 ? 1 : -1;
        setCurrentSliceIdx((prev) => Math.max(0, Math.min(totalSlices - 1, prev + direction)));
        accumulatedDelta = 0;
      }
    };

    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      if (decayTimer) clearTimeout(decayTimer);
      el.removeEventListener('wheel', handleWheel);
    };
  }, [totalSlices]);

  // Keyboard navigation & clinician shortcuts
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

      if (isInputActive) return;

      // Escape key exits theater mode
      if (e.key === 'Escape' && isTheaterMode && !isModalOpen) {
        e.preventDefault();
        toggleTheaterMode();
        return;
      }

      if (isModalOpen) return;

      if (e.key === 'ArrowLeft') {
        if (totalSlices <= 1) return;
        e.preventDefault();
        setIsPlayingCine(false);
        setCurrentSliceIdx((prev) => Math.max(0, prev - 1));
      } else if (e.key === 'ArrowRight') {
        if (totalSlices <= 1) return;
        e.preventDefault();
        setIsPlayingCine(false);
        setCurrentSliceIdx((prev) => Math.min(totalSlices - 1, prev + 1));
      } else if (e.key === ' ') {
        if (totalSlices <= 1) return;
        e.preventDefault();
        setIsPlayingCine((prev) => !prev);
      } else if (e.key.toLowerCase() === 'w') {
        e.preventDefault();
        const presetKeys = ['default', 'lung', 'mediastinum', 'bone'];
        setWlPreset((prev) => {
          const idx = presetKeys.indexOf(prev);
          return presetKeys[(idx + 1) % presetKeys.length];
        });
      } else if (e.key.toLowerCase() === 's' && !isPreview) {
        e.preventDefault();
        setDisplayLayout((prev) => (prev === 'dual' ? 'curtain' : 'dual'));
      } else if (e.key.toLowerCase() === 'z') {
        e.preventDefault();
        setZoomLevel((z) => (z === 1 ? 1.5 : 1));
      } else if (e.key.toLowerCase() === 'f') {
        e.preventDefault();
        toggleTheaterMode();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [totalSlices, isPreview, isTheaterMode, toggleTheaterMode]);

  // Pointer dragging handler for split curtain wipe
  const updateSplitFromPointer = useCallback((clientX) => {
    const el = curtainRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    const clamped = Math.max(2, Math.min(98, pct));
    setSplitPos(clamped);
  }, []);

  const handleCurtainPointerDown = (e) => {
    setIsDraggingCurtain(true);
    updateSplitFromPointer(e.clientX);
    try {
      e.currentTarget.setPointerCapture?.(e.pointerId);
    } catch (err) { }
  };

  const handleCurtainPointerMove = (e) => {
    if (!isDraggingCurtain) return;
    updateSplitFromPointer(e.clientX);
  };

  const handleCurtainPointerUp = (e) => {
    if (isDraggingCurtain) {
      setIsDraggingCurtain(false);
      try {
        e.currentTarget.releasePointerCapture?.(e.pointerId);
      } catch (err) { }
    }
  };

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

  const displayAccuracy =
    currentSlice?.slice_ai_accuracy ??
    currentSlice?.slice_dice ??
    analysisData.ai_accuracy ??
    analysisData.overall_dice;

  // Z-Axis Lesion Burden Curve Calculations
  const sliceBurdens = slices.map((s, idx) => ({
    sliceIdx: idx,
    burden: Number(s.involvement_pct) || 0,
    hasLesion: Boolean(s.has_lesion) || (Number(s.involvement_pct) || 0) > 0,
  }));
  const maxBurden = Math.max(...sliceBurdens.map((d) => d.burden), 1);
  const peakIndex = sliceBurdens.reduce(
    (maxIdx, curr, idx, arr) => (curr.burden > arr[maxIdx].burden ? idx : maxIdx),
    0
  );
  const peakBurden = sliceBurdens[peakIndex]?.burden || 0;

  // Generate SVG coordinates for 1000 x 44 viewBox
  const curvePoints = sliceBurdens.map((d, i) => {
    const x = totalSlices > 1 ? (i / (totalSlices - 1)) * 1000 : 500;
    const y = 40 - (d.burden / maxBurden) * 34;
    return { x, y, ...d };
  });

  const lineD = curvePoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(' ');
  const areaD = `${lineD} L 1000 44 L 0 44 Z`;

  const currSliceX = totalSlices > 1 ? (currentSliceIdx / (totalSlices - 1)) * 1000 : 500;
  const currSliceY = curvePoints[currentSliceIdx]?.y || 40;

  const handleCurveScrub = useCallback((clientX, svgEl) => {
    if (!svgEl || totalSlices <= 1) return;
    const rect = svgEl.getBoundingClientRect();
    if (rect.width === 0) return;
    const clickRatio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const targetSlice = Math.min(totalSlices - 1, Math.round(clickRatio * (totalSlices - 1)));
    setIsPlayingCine(false);
    setCurrentSliceIdx(targetSlice);
  }, [totalSlices]);

  const handleCurvePointerDown = (e) => {
    setIsScrubbingCurve(true);
    handleCurveScrub(e.clientX, e.currentTarget);
    try {
      e.currentTarget.setPointerCapture?.(e.pointerId);
    } catch (err) { }
  };

  const handleCurvePointerMove = (e) => {
    if (!isScrubbingCurve) return;
    handleCurveScrub(e.clientX, e.currentTarget);
  };

  const handleCurvePointerUp = (e) => {
    if (isScrubbingCurve) {
      setIsScrubbingCurve(false);
      try {
        e.currentTarget.releasePointerCapture?.(e.pointerId);
      } catch (err) { }
    }
  };

  const handleCurveClick = (e) => {
    handleCurveScrub(e.clientX, e.currentTarget);
  };

  const activeFilter = WL_PRESETS[wlPreset]?.filter || 'none';

  return (
    <section
      ref={viewerContainerRef}
      className={`viewer-card ${isTheaterMode ? 'diagnostic-theater' : ''}`}
      aria-label="Synchronized Radiology Image Viewer"
    >
      {/* Viewer Header Toolbar */}
      <div className="viewer-header">
        <div className="viewer-header-left">
          <div className="viewer-header-title">
            <Layers size={18} color="var(--color-brand-secondary)" strokeWidth={2} aria-hidden="true" />
            <span>
              {isTheaterMode
                ? 'Diagnostic Theater'
                : isPreview
                  ? 'CT Scan Inspection'
                  : 'Synchronized Radiology Canvas'}
            </span>
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
          {/* Window / Level (WW / WL) Preset Selector */}
          <div className="wl-group" role="group" aria-label="Window / Level CT Presets">
            <span className="wl-label" title="Window / Level Presets (Hotkey: W)">
              <SunMedium size={13} strokeWidth={2} aria-hidden="true" />
              <span>W/L</span>
            </span>
            <div className="segmented-control wl-segmented">
              {Object.values(WL_PRESETS).map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  className={`segmented-option ${wlPreset === preset.id ? 'active' : ''}`}
                  onClick={() => setWlPreset(preset.id)}
                  title={preset.name}
                  aria-pressed={wlPreset === preset.id}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* View Mode Switcher or Comparison Layout (Analyzed Mode) */}
          {!isPreview ? (
            <>
              {/* Display Layout Switcher: Side-by-Side vs Split Curtain */}
              <div className="segmented-control" role="group" aria-label="Canvas Layout Mode">
                <button
                  type="button"
                  className={`segmented-option ${displayLayout === 'dual' ? 'active' : ''}`}
                  onClick={() => setDisplayLayout('dual')}
                  title="Dual Canvas Side-by-Side (Hotkey: S)"
                  aria-pressed={displayLayout === 'dual'}
                >
                  <Columns size={13} strokeWidth={2} aria-hidden="true" />
                  <span>Dual</span>
                </button>
                <button
                  type="button"
                  className={`segmented-option ${displayLayout === 'curtain' ? 'active' : ''}`}
                  onClick={() => setDisplayLayout('curtain')}
                  title="Split Curtain Wipe Comparison (Hotkey: S)"
                  aria-pressed={displayLayout === 'curtain'}
                >
                  <Split size={13} strokeWidth={2} aria-hidden="true" />
                  <span>Split Wipe</span>
                </button>
              </div>

              {/* Segmented View Mode Switcher */}
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
                  <span>AI Overlay</span>
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
            </>
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
            title={zoomLevel === 1 ? 'Zoom In (1.5x) [Hotkey: Z]' : 'Reset Zoom (1x) [Hotkey: Z]'}
            aria-label={zoomLevel === 1 ? 'Magnify scan to 1.5x' : 'Reset magnification to default'}
          >
            {zoomLevel === 1 ? (
              <Maximize2 size={15} strokeWidth={1.8} aria-hidden="true" />
            ) : (
              <Minimize2 size={15} strokeWidth={1.8} aria-hidden="true" />
            )}
          </button>

          {/* Diagnostic Theater / Fullscreen Mode Toggle */}
          <button
            type="button"
            className={`btn-icon ${isTheaterMode ? 'btn-theater-active' : ''}`}
            onClick={toggleTheaterMode}
            title={isTheaterMode ? 'Exit Diagnostic Theater (Esc / F)' : 'Diagnostic Theater Fullscreen (Hotkey: F)'}
            aria-label={isTheaterMode ? 'Exit Diagnostic Theater' : 'Enter Diagnostic Theater Fullscreen'}
          >
            {isTheaterMode ? (
              <Minimize2 size={15} strokeWidth={1.8} aria-hidden="true" />
            ) : (
              <Maximize size={15} strokeWidth={1.8} aria-hidden="true" />
            )}
          </button>

          {/* Keyboard Shortcuts Help */}
          <button
            type="button"
            className="btn-icon"
            onClick={onOpenShortcuts}
            title="Keyboard Shortcuts Cheatsheet (?)"
            aria-label="Radiology Workstation Keyboard Shortcuts"
          >
            <HelpCircle size={15} strokeWidth={1.8} aria-hidden="true" />
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

      {/* Main Radiology Viewport (Mouse Wheel Scrubbing Enabled) */}
      <div
        ref={canvasesRef}
        className="viewer-viewports-container"
        tabIndex={0}
        role="region"
        aria-label="CT slice interactive canvas viewport. Scroll mouse wheel to scrub slices."
      >
        {displayLayout === 'curtain' && !isPreview ? (
          /* Interactive Split Curtain / Wipe Mode */
          <div
            ref={curtainRef}
            className="curtain-container"
            onPointerDown={handleCurtainPointerDown}
            onPointerMove={handleCurtainPointerMove}
            onPointerUp={handleCurtainPointerUp}
            onPointerCancel={handleCurtainPointerUp}
            role="slider"
            aria-label="Split curtain comparison wipe"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(splitPos)}
          >
            {/* Base CT Underlay */}
            <img
              src={currentSlice?.raw_image}
              alt={`Raw CT slice ${currentSliceIdx + 1}`}
              className="canvas-img"
              draggable={false}
              onDragStart={(e) => e.preventDefault()}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                filter: activeFilter,
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'center center',
                transition: 'transform var(--duration-fast) var(--ease-apple)',
                pointerEvents: 'none',
                userSelect: 'none',
                WebkitUserDrag: 'none',
              }}
            />

            {/* Diagnostic Foreground Overlay (Clipped by splitPos) */}
            <img
              src={getForegroundImage()}
              alt={`${getCanvasTitle()} for slice ${currentSliceIdx + 1}`}
              className="canvas-img curtain-foreground"
              draggable={false}
              onDragStart={(e) => e.preventDefault()}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                opacity: overlayOpacity,
                filter: activeFilter,
                clipPath: `inset(0 0 0 ${splitPos}%)`,
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'center center',
                transition: 'opacity var(--duration-fast) ease, transform var(--duration-fast) var(--ease-apple)',
                pointerEvents: 'none',
                userSelect: 'none',
                WebkitUserDrag: 'none',
              }}
            />

            {/* Draggable Vertical Hairline Divider */}
            <div
              className="curtain-hairline"
              style={{ left: `${splitPos}%` }}
              aria-hidden="true"
            >
              <div className="curtain-handle" title="Drag to adjust split curtain comparison">
                <Split size={14} strokeWidth={2} />
              </div>
            </div>

            {/* HUD Orientation Badges */}
            <div className="curtain-badge left-badge">
              <span>RAW CT SCAN</span>
            </div>

            <div className="curtain-badge right-badge">
              <span
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  backgroundColor:
                    viewMode === 'gt'
                      ? '#10B981'
                      : viewMode === 'composite'
                        ? '#F59E0B'
                        : viewMode === 'contour'
                          ? '#0EA5E9'
                          : '#EF4444',
                  display: 'inline-block'
                }}
              />
              <span>{getCanvasTitle()} ({Math.round(100 - splitPos)}% Visible)</span>
            </div>

            {currentSlice?.has_lesion && (
              <div className="canvas-indicator-badge tabular-nums" style={{ background: 'rgba(239, 68, 68, 0.92)' }}>
                Lesion Involvement: {formatMetric(currentSlice.involvement_pct)}%
              </div>
            )}
          </div>
        ) : (
          /* Standard Dual Canvas Viewport (Side-by-Side) */
          <div className="canvases-container" style={isPreview ? { gridTemplateColumns: '1fr' } : {}}>
            {/* Left Viewport: Raw Base CT Scan */}
            <div
              className="canvas-wrapper"
              style={isPreview ? { maxHeight: isTheaterMode ? 'calc(100vh - 190px)' : '520px', aspectRatio: 'unset' } : {}}
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
                draggable={false}
                onDragStart={(e) => e.preventDefault()}
                style={{
                  filter: activeFilter,
                  transform: `scale(${zoomLevel})`,
                  transformOrigin: 'center center',
                  transition: 'transform var(--duration-fast) var(--ease-apple)',
                  willChange: 'transform',
                  userSelect: 'none',
                  maxHeight: isPreview ? (isTheaterMode ? 'calc(100vh - 190px)' : '520px') : '100%'
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

                {/* Base CT image underlay */}
                <img
                  src={currentSlice?.raw_image}
                  alt=""
                  aria-hidden="true"
                  className="canvas-img"
                  draggable={false}
                  onDragStart={(e) => e.preventDefault()}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    filter: activeFilter,
                    transform: `scale(${zoomLevel})`,
                    transformOrigin: 'center center',
                    transition: 'transform var(--duration-fast) var(--ease-apple)',
                    willChange: 'transform',
                    userSelect: 'none'
                  }}
                />

                {/* Diagnostic Foreground Overlay with controlled opacity */}
                <img
                  src={getForegroundImage()}
                  alt={`${getCanvasTitle()} for slice ${currentSliceIdx + 1}`}
                  className="canvas-img"
                  draggable={false}
                  onDragStart={(e) => e.preventDefault()}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    opacity: overlayOpacity,
                    filter: activeFilter,
                    transform: `scale(${zoomLevel})`,
                    transformOrigin: 'center center',
                    transition: 'opacity var(--duration-fast) ease, transform var(--duration-fast) var(--ease-apple)',
                    willChange: 'transform',
                    userSelect: 'none'
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

        {/* 3D Volumetric Axial Slice Navigator & Z-Axis Lesion Burden Curve */}
        {totalSlices > 1 && (
          <div
            className="volume-navigation-block"
            style={!isPreview ? { borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' } : {}}
          >
            {/* Z-Axis Lesion Burden Curve Micro-Histogram */}
            {!isPreview && (
              <div className="z-axis-burden-container">
                <div className="z-axis-header">
                  <span className="z-axis-title">
                    Z-Axis Lesion Burden Curve
                  </span>
                  {peakBurden > 0 && (
                    <button
                      type="button"
                      className="peak-jump-btn"
                      onClick={() => {
                        setIsPlayingCine(false);
                        setCurrentSliceIdx(peakIndex);
                      }}
                      title={`Jump directly to slice with maximum lesion burden (${peakBurden.toFixed(1)}%)`}
                    >
                      <span className="peak-dot" />
                      <span>Max Disease: <strong>{peakBurden.toFixed(1)}%</strong> (Slice {peakIndex + 1})</span>
                    </button>
                  )}
                </div>

                <div className="z-axis-curve-wrapper">
                  <svg
                    className="z-axis-curve-svg"
                    viewBox="0 0 1000 44"
                    preserveAspectRatio="none"
                    onPointerDown={handleCurvePointerDown}
                    onPointerMove={handleCurvePointerMove}
                    onPointerUp={handleCurvePointerUp}
                    onPointerCancel={handleCurvePointerUp}
                    onClick={handleCurveClick}
                    title="Click or drag anywhere on the curve to jump to that slice"
                    aria-label="Z-Axis lesion involvement distribution graph across axial volume"
                  >
                    <defs>
                      <linearGradient id="zBurdenGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#FF453A" stopOpacity="0.5" />
                        <stop offset="70%" stopColor="#FF453A" stopOpacity="0.12" />
                        <stop offset="100%" stopColor="#FF453A" stopOpacity="0.0" />
                      </linearGradient>
                    </defs>

                    {/* Shaded Area under curve */}
                    <path d={areaD} fill="url(#zBurdenGradient)" />

                    {/* Smooth Lesion Burden Line */}
                    <path
                      d={lineD}
                      fill="none"
                      stroke="#FF453A"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />

                    {/* Peak Highlight Circle */}
                    {peakBurden > 0 && (
                      <circle
                        cx={curvePoints[peakIndex]?.x || 0}
                        cy={curvePoints[peakIndex]?.y || 40}
                        r="4"
                        fill="#FF453A"
                        stroke="#FFFFFF"
                        strokeWidth="1.5"
                      />
                    )}

                    {/* Current Slice Cursor Line & Marker */}
                    <line
                      x1={currSliceX}
                      y1="0"
                      x2={currSliceX}
                      y2="44"
                      stroke="#0A84FF"
                      strokeWidth="2"
                      strokeDasharray="2 2"
                    />
                    <circle
                      cx={currSliceX}
                      cy={currSliceY}
                      r="3.5"
                      fill="#0A84FF"
                      stroke="#FFFFFF"
                      strokeWidth="1.2"
                    />
                  </svg>
                </div>
              </div>
            )}

            {/* Scrubber Controls with Lesion Presence Tick Marks */}
            <div className="control-row slice-scrubber-row">
              <div className="control-label">
                <Eye size={15} strokeWidth={1.8} aria-hidden="true" />
                <span>
                  Axial Slice ({currentSliceIdx + 1} of {totalSlices})
                </span>
              </div>

              <div className="slider-group-with-ticks">
                {/* Play/Pause Cine Loop */}
                <button
                  type="button"
                  className="btn-icon"
                  onClick={() => setIsPlayingCine(!isPlayingCine)}
                  title={isPlayingCine ? 'Pause Cine Loop (Space)' : 'Play 3D Cine Loop (Space)'}
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

                {/* Range Scrubber Track with Lesion Presence Tick Marks */}
                <div className="slider-track-container">
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

                  {/* Lesion Presence Tick Marks */}
                  <div className="slider-ticks-track" aria-hidden="true">
                    {slices.map((s, idx) => {
                      if (!s.has_lesion && (Number(s.involvement_pct) || 0) <= 0) return null;
                      const leftPct = (idx / (totalSlices - 1)) * 100;
                      const isPeak = peakBurden > 0 && idx === peakIndex;
                      return (
                        <span
                          key={idx}
                          className={`slider-lesion-tick ${isPeak ? 'peak' : ''}`}
                          style={{ left: `${leftPct}%` }}
                        />
                      );
                    })}
                  </div>
                </div>

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
          </div>
        )}
      </div>
    </section>
  );
}
