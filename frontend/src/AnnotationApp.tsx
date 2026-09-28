import { useCallback, useEffect, useState } from 'react';
import ExpertAnnotationPanel from './components/ExpertAnnotationPanel';
import ExpertValidationPanel from './components/ExpertValidationPanel';
import ResultsPanel from './components/ResultsPanel';
import StructuralAuditViewer from './components/StructuralAuditViewer';
import AnnotationCanvas from './components/annotation/AnnotationCanvas';
import ReviewFoundationPanel from './components/ReviewFoundationPanel';
import AssociationReviewPanel from './components/AssociationReviewPanel';
import SurfacePilotPanel from './components/SurfacePilotPanel';
import LegacySurfaceRecordsPanel from './components/LegacySurfaceRecordsPanel';
import AnatomicalOverlayPanel from './components/AnatomicalOverlayPanel';
import GroundTruthPilotPanel from './components/GroundTruthPilotPanel';
import { getHistory, registerSource, type ReviewMode, type ReviewState } from './lib/reviewFoundation';
import type { AnnotationDraft } from './types';

const blankAnnotation: AnnotationDraft = {
  status: 'Uncertain',
  teeth: [],
  severity: 'Cannot determine',
  findings: [],
  notes: '',
  region: 'Bounding box',
  spatial: { method: 'Bounding box', boundingBox: null, polygon: [] },
  validated: false,
  validationStatus: 'Pending',
};

type LoadedImage = { file: File; src: string; width: number; height: number; source: 'library' | 'upload' };
type LibraryCase = { source_partition: 'Training' | 'Validation'; source_image_id: string; stable_image_id: string; arch: string; site: string; visible_tooth_count: number };
const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || 'http://127.0.0.1:8000';

function readImage(file: File, source: LoadedImage['source']): Promise<LoadedImage> {
  return new Promise((resolve, reject) => {
    const src = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => resolve({ file, src, width: image.naturalWidth, height: image.naturalHeight, source });
    image.onerror = () => { URL.revokeObjectURL(src); reject(new Error(`Could not decode ${file.name}.`)); };
    image.src = src;
  });
}

export default function AnnotationApp() {
  const [surfacePilotMode, setSurfacePilotMode] = useState(() => ['surface-v1', 'merged-anatomical-review', 'stage2c', 'stage2c-v2'].includes(new URLSearchParams(window.location.search).get('pilot') || ''));
  const [groundTruthPilotMode, setGroundTruthPilotMode] = useState(() => new URLSearchParams(window.location.search).get('pilot') === 'ground-truth-v1');
  const [images, setImages] = useState<LoadedImage[]>([]);
  const [imageIndex, setImageIndex] = useState(0);
  const [annotation, setAnnotation] = useState<AnnotationDraft>(blankAnnotation);
  const [error, setError] = useState<string | null>(null);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [validationMode, setValidationMode] = useState(false);
  const [resultsMode, setResultsMode] = useState(false);
  const [structuralAuditMode, setStructuralAuditMode] = useState(false);
  const [spatialMode, setSpatialMode] = useState(false);
  const [versionedMode, setVersionedMode] = useState(false);
  const [associationMode, setAssociationMode] = useState(false);
  const [legacySurfaceMode, setLegacySurfaceMode] = useState(false);
  const [anatomicalOverlayMode, setAnatomicalOverlayMode] = useState(false);
  const [annotationMode, setAnnotationMode] = useState(false);
  const [caseLibrary, setCaseLibrary] = useState<LibraryCase[]>([]);
  const [caseFilter, setCaseFilter] = useState<'All' | 'Training' | 'Validation'>('All');
  const [caseLoading, setCaseLoading] = useState('');
  const [intakeReviewMode, setIntakeReviewMode] = useState<ReviewMode>('ai_assisted');
  const [imageReviewModes, setImageReviewModes] = useState<Record<string, ReviewMode>>({});
  const [reviewStates, setReviewStates] = useState<Record<string, ReviewState>>({});
  const updateReviewState = useCallback((state: ReviewState) => setReviewStates(current => ({ ...current, [state.image_id]: state })), []);
  const currentImage = images[imageIndex];
  useEffect(() => {
    const pilot = new URLSearchParams(window.location.search).get('pilot');
    if (pilot === 'surface-v1' || pilot === 'stage2c' || pilot === 'stage2c-v2') {
      window.history.replaceState(null, '', `${window.location.pathname}?pilot=merged-anatomical-review`);
    }
  }, []);
  useEffect(() => { fetch(`${API}/api/stage2c/queue`).then(response => response.ok ? response.json() : Promise.reject(new Error('Case library unavailable'))).then(data => setCaseLibrary(data.entries || [])).catch(() => setCaseLibrary([])); }, []);

  function showPilot(sourceImageId?: string) {
    const query = new URLSearchParams({ pilot: 'merged-anatomical-review' });
    if (sourceImageId) query.set('image', sourceImageId);
    window.history.replaceState(null, '', `${window.location.pathname}?${query}`);
    setSurfacePilotMode(true);
  }

  if (surfacePilotMode) return <SurfacePilotPanel initialSourceImageId={new URLSearchParams(window.location.search).get('image') || undefined} onExit={() => { window.history.replaceState(null, '', window.location.pathname); setSurfacePilotMode(false); }} />;
  if (groundTruthPilotMode) return <GroundTruthPilotPanel onExit={() => { window.history.replaceState(null, '', window.location.pathname); setGroundTruthPilotMode(false); }} />;

  async function loadFiles(files: File[], reviewMode: ReviewMode, source: LoadedImage['source']) {
    if (!files.length) { setError('Choose one or more image files.'); return; }
    try {
      const loaded = await Promise.all(files.map(file => readImage(file, source)));
      let registered: Array<readonly [string, ReviewState]> = [];
      let registrationError: string | null = null;
      try {
        registered = await Promise.all(loaded.map(async image => {
          const id = imageId(image.file);
          await registerSource(id, image.file, reviewMode, 'case library or exploratory intake');
          return [id, (await getHistory(id)).review_state] as const;
        }));
      } catch (failure) {
        registrationError = failure instanceof Error ? failure.message : 'Versioned source registration failed.';
      }
      const registeredAll = !registrationError;
      setImageReviewModes(registeredAll
        ? Object.fromEntries(registered.map(([id, state]) => [id, state.mode]))
        : Object.fromEntries(loaded.map(image => [imageId(image.file), reviewMode])));
      setReviewStates(registeredAll ? Object.fromEntries(registered) : {});
      images.forEach((image) => URL.revokeObjectURL(image.src));
      setImages(loaded);
      setImageIndex(0);
      setAnnotation(blankAnnotation);
      setValidationMode(false);
      setResultsMode(false);
      setStructuralAuditMode(false);
      setSpatialMode(false);
      setAnatomicalOverlayMode(false);
      setAnnotationMode(false);
      setVersionedMode(source === 'library' && registeredAll);
      setSavedMessage(null);
      setError(registrationError ? `Versioned registration failed: ${registrationError}. Local legacy viewing remains available; AI controls are disabled.` : null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'One of the selected images could not be read.');
    }
  }

  async function handleFiles(fileList: FileList | null) {
    if (!fileList) return;
    await loadFiles(Array.from(fileList).filter(file => file.type.startsWith('image/')), intakeReviewMode, 'upload');
  }

  async function openLibraryCase(entry: LibraryCase) {
    setCaseLoading(entry.stable_image_id); setError(null);
    try {
      const imageEndpoint = entry.source_partition === 'Validation'
        ? `${API}/structural-audit/${entry.source_image_id}/image`
        : `${API}/api/stage2c/images/${entry.source_partition}/${entry.source_image_id}`;
      const response = await fetch(imageEndpoint);
      if (!response.ok) throw new Error('The selected case image could not be loaded.');
      const blob = await response.blob();
        // Preserve the source image id. Downstream review and audit endpoints
        // use this id (for example, Validation case 496), so adding the
        // partition prefix here would turn it into an invalid id.
        const file = new File([blob], `${entry.source_image_id}.jpg`, { type: blob.type || 'image/jpeg' });
      await loadFiles([file], intakeReviewMode, 'library');
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Case could not be opened.'); }
    finally { setCaseLoading(''); }
  }

  function selectImage(index: number) {
    setImageIndex(index);
    setAnnotation(blankAnnotation);
    setValidationMode(false);
    setResultsMode(false);
    setStructuralAuditMode(false);
    setSpatialMode(false);
    setAnatomicalOverlayMode(false);
    setAnnotationMode(false);
    setVersionedMode(images[index].source === 'library' && Boolean(reviewStates[imageId(images[index].file)]));
    setSavedMessage(null);
  }

  function imageId(file: File): string {
    return file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]+/g, '-');
  }

  function updateAnnotation(patch: Partial<AnnotationDraft>) {
    setAnnotation((current) => ({ ...current, ...patch }));
  }

  function openTool(tool: 'annotation' | 'versioned' | 'legacy-surface' | 'anatomy' | 'association' | 'spatial' | 'segmentation' | 'audit') {
    setVersionedMode(tool === 'versioned');
    setLegacySurfaceMode(tool === 'legacy-surface');
    setAnatomicalOverlayMode(tool === 'anatomy');
    setAnnotationMode(tool === 'annotation');
    setAssociationMode(tool === 'association');
    setSpatialMode(tool === 'spatial');
    setResultsMode(tool === 'segmentation');
    setStructuralAuditMode(tool === 'audit');
    setValidationMode(false);
  }

  return <><a className="skip-link" href="#workspace-main">Skip to main workspace</a><div className="app-shell annotation-app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">P</div><div><strong>PERIO<span>LAB</span></strong><small>Expert annotation</small></div></div>
      <div className="study-card"><span className="eyebrow">ACTIVE STUDY</span><strong>Localized periodontal disease</strong><span className="study-id">RESEARCH PROTOTYPE</span></div>
      <nav aria-label="Research workflows" className="workspace-nav"><span className="eyebrow">WORKFLOWS</span><button className="nav-item active" onClick={() => { setImages([]); setImageIndex(0); }}><span className="nav-index">01</span><span>Image review</span></button><button className="nav-item" onClick={() => showPilot()}><span className="nav-index">02</span><span>Unified anatomical review</span><small>Stage 2C + Surface · same image</small></button><button className="nav-item" onClick={() => { window.history.replaceState(null, '', `${window.location.pathname}?pilot=ground-truth-v1`); setGroundTruthPilotMode(true); }}><span className="nav-index">03</span><span>BRAR verification pilot</span></button></nav>
      <div className="sidebar-foot"><span className="status-dot" />Local prototype <small>No clinical connection</small></div>
    </aside>
    <main className="main-area" id="workspace-main">
      <header className="topbar"><div><span className="breadcrumb">RESEARCH WORKSPACE / STRUCTURAL REVIEW</span><h1>Oral assessment and treatment planning</h1></div><span className="pill neutral">RESEARCH ONLY</span></header>
      <div className="content">
        {!currentImage ? <section className="annotation-intake case-library">
          <span className="eyebrow">CASE LIBRARY</span><h2>Choose a review workflow</h2>
          <p>Open a predefined, versioned study case. Model output remains hidden during independent evaluation. Uploaded files are kept separate as exploratory work.</p>
          <div className="review-mode-cards" role="radiogroup" aria-label="Review mode"><button className={intakeReviewMode==='independent_evaluation'?'selected':''} onClick={()=>setIntakeReviewMode('independent_evaluation')}><strong>Independent evaluation</strong><span>Review the source image before any AI result is revealed.</span></button><button className={intakeReviewMode==='ai_assisted'?'selected':''} onClick={()=>setIntakeReviewMode('ai_assisted')}><strong>Routine AI-assisted review</strong><span>Review with versioned AI support and expert correction.</span></button></div>
          <div className="case-library-heading"><div><h3>{intakeReviewMode==='independent_evaluation'?'Independent evaluation cases':'Routine AI-assisted cases'}</h3><span>{caseLibrary.length} cases from the frozen manifest</span></div><label>Partition<select value={caseFilter} onChange={e=>setCaseFilter(e.target.value as typeof caseFilter)}><option>All</option><option>Training</option><option>Validation</option></select></label></div>
          <div className="library-case-grid">{caseLibrary.filter(x=>caseFilter==='All'||x.source_partition===caseFilter).map((entry,i)=><button key={entry.stable_image_id} disabled={!!caseLoading} onClick={()=>void openLibraryCase(entry)}><strong>Case {i+1}</strong><span>{entry.arch} · {entry.site}</span><span>{entry.visible_tooth_count} visible teeth</span><small>{caseLoading===entry.stable_image_id?'Opening…':'Open case'}</small></button>)}</div>
          <details className="exploratory-upload"><summary>Exploratory tools: upload your own image</summary><p>Uploads are not part of the frozen study queue or pilot acceptance results.</p><label className="dropzone"><strong>Choose exploratory radiographs</strong><small>Multiple images supported</small><input type="file" accept="image/*" multiple onChange={event => void handleFiles(event.target.files)} /></label></details>
          {error && <div className="save-error" role="alert">{error}</div>}
        </section> : <>
          <div className="annotation-casebar"><div><span className="eyebrow">CASE {imageIndex + 1} OF {images.length}</span><h2>{currentImage.file.name}</h2><span>{currentImage.width} x {currentImage.height} px · {imageId(currentImage.file)}</span></div>
            <div className={`casebar-actions ${currentImage.source==='upload'?'image-actions':'workspace-tools'}`} role="navigation" aria-label={currentImage.source==='upload'?'Uploaded image actions':'Case tools'}>
              {currentImage.source==='library'?<>
                <button className="secondary-btn" disabled={!reviewStates[imageId(currentImage.file)]} onClick={() => openTool('versioned')}>Versioned review</button>
                <button className="secondary-btn" onClick={() => showPilot(imageId(currentImage.file))}>Unified anatomical review</button>
                <button className="secondary-btn" onClick={() => openTool('annotation')}>Expert annotation</button>
                <button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('spatial')}>Spatial annotation</button>
                <button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('segmentation')}>Tooth segmentation</button>
                <button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('audit')}>Structural audit</button>
                <label className="secondary-btn upload-more">Add images<input type="file" accept="image/*" multiple onChange={event => void handleFiles(event.target.files)} /></label>
                <details className="legacy-tools"><summary>Advanced / Legacy tools</summary><div><button className="secondary-btn" onClick={() => openTool('legacy-surface')}>Historical surface records (read-only)</button><button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('association')}>Legacy association review</button></div></details>
              </>:<>
              <label className="secondary-btn upload-more">Add images<input type="file" accept="image/*" multiple onChange={event => void handleFiles(event.target.files)} /></label>
              <details className="legacy-tools advanced-tools"><summary>Advanced tools</summary><div>
                <button className="secondary-btn" onClick={() => showPilot(imageId(currentImage.file))}>Unified anatomical review</button>
                <button className="secondary-btn" disabled={!reviewStates[imageId(currentImage.file)]} onClick={() => openTool('versioned')}>Versioned review</button>
                <button className="secondary-btn" onClick={() => openTool('annotation')}>Expert annotation</button>
                <button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('spatial')}>Spatial annotation</button>
                <button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('segmentation')}>Tooth segmentation</button>
                <button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('audit')}>Structural audit</button>
                <button className="secondary-btn" onClick={() => openTool('legacy-surface')}>Historical surface records (read-only)</button>
                <button className="secondary-btn" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('association')}>Legacy association review</button>
              </div></details>
              </>}
            </div></div>
          {savedMessage && <div className="notice teal"><span>✓</span><p>{savedMessage}</p></div>}
          {error && <div className="save-error" role="alert">{error}</div>}
          {anatomicalOverlayMode ? <AnatomicalOverlayPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} imageSrc={currentImage.src} imageFile={currentImage.file} onBack={() => setAnatomicalOverlayMode(false)} /> : legacySurfaceMode ? <LegacySurfaceRecordsPanel imageId={imageId(currentImage.file)} onBack={() => setLegacySurfaceMode(false)} /> : associationMode ? <AssociationReviewPanel imageId={imageId(currentImage.file)} imageSrc={currentImage.src} onBack={() => setAssociationMode(false)} /> : versionedMode ? <ReviewFoundationPanel imageId={imageId(currentImage.file)} imageSrc={currentImage.src} imageFile={currentImage.file} width={currentImage.width} height={currentImage.height} mode={imageReviewModes[imageId(currentImage.file)] || intakeReviewMode} onState={updateReviewState} onBack={() => setVersionedMode(false)} />
            : spatialMode ? <AnnotationCanvas imageId={imageId(currentImage.file)} imageSrc={currentImage.src} imageFile={currentImage.file} width={currentImage.width} height={currentImage.height} onBack={() => setSpatialMode(false)} />
            : structuralAuditMode ? <StructuralAuditViewer imageId={imageId(currentImage.file)} imageSrc={currentImage.src} onBack={() => setStructuralAuditMode(false)} />
            : resultsMode ? <ResultsPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} imageSrc={currentImage.src} imageFile={currentImage.file} annotation={annotation} onBack={() => setResultsMode(false)} />
            : validationMode ? <ExpertValidationPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} originalSrc={currentImage.src} annotation={annotation} onChange={updateAnnotation} onEdit={() => setValidationMode(false)} onSaved={(timestamp, status) => { updateAnnotation({ validationStatus: status, validationTimestamp: timestamp, validated: status === 'Validated', savedAt: timestamp }); setSavedMessage(`Validation saved as ${status}. Ground truth remains attributable to expert review.`); setResultsMode(imageReviewModes[imageId(currentImage.file)] === 'ai_assisted' || !!reviewStates[imageId(currentImage.file)]?.revealed); }} />
            : annotationMode ? <ExpertAnnotationPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} imageSrc={currentImage.src} annotation={annotation} onChange={updateAnnotation} onSaved={(savedAt) => { updateAnnotation({ savedAt, validationStatus: 'Pending' }); setSavedMessage('Annotation saved to the FastAPI research backend. Preliminary analysis remains separate from ground truth.'); }} onContinue={() => setValidationMode(true)} canPrevious={imageIndex > 0} canNext={imageIndex < images.length - 1} imageIndex={imageIndex} imageCount={images.length} onPrevious={() => selectImage(imageIndex - 1)} onNext={() => selectImage(imageIndex + 1)} />
            : currentImage.source==='upload'?<section className="image-review-preview"><div className="image-review-frame"><img src={currentImage.src} alt={`Uploaded radiograph ${currentImage.file.name}`}/></div><div className="image-review-prompt"><span className="eyebrow">IMAGE READY</span><h2>Generate the anatomical overlay</h2><p>Run the experimental models to display predicted CEJ points, bone level, and apex key points on this radiograph.</p><button className="primary-action" disabled={imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed} onClick={() => openTool('anatomy')}>Generate Anatomical Overlay</button>{imageReviewModes[imageId(currentImage.file)] === 'independent_evaluation' && !reviewStates[imageId(currentImage.file)]?.revealed&&<small>Complete and reveal the independent review before viewing AI output.</small>}</div></section>:<ExpertAnnotationPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} imageSrc={currentImage.src} annotation={annotation} onChange={updateAnnotation} onSaved={(savedAt) => { updateAnnotation({ savedAt, validationStatus: 'Pending' }); setSavedMessage('Annotation saved to the FastAPI research backend. Preliminary analysis remains separate from ground truth.'); }} onContinue={() => setValidationMode(true)} canPrevious={imageIndex > 0} canNext={imageIndex < images.length - 1} imageIndex={imageIndex} imageCount={images.length} onPrevious={() => selectImage(imageIndex - 1)} onNext={() => selectImage(imageIndex + 1)}/>}
        </>}
      </div>
    </main>
  </div></>;
}
