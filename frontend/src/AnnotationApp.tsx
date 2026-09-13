import { useState } from 'react';
import ExpertAnnotationPanel from './components/ExpertAnnotationPanel';
import ExpertValidationPanel from './components/ExpertValidationPanel';
import ResultsPanel from './components/ResultsPanel';
import StructuralAuditViewer from './components/StructuralAuditViewer';
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

type LoadedImage = { file: File; src: string; width: number; height: number };

function readImage(file: File): Promise<LoadedImage> {
  return new Promise((resolve, reject) => {
    const src = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => resolve({ file, src, width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => { URL.revokeObjectURL(src); reject(new Error(`Could not decode ${file.name}.`)); };
    image.src = src;
  });
}

export default function AnnotationApp() {
  const [images, setImages] = useState<LoadedImage[]>([]);
  const [imageIndex, setImageIndex] = useState(0);
  const [annotation, setAnnotation] = useState<AnnotationDraft>(blankAnnotation);
  const [error, setError] = useState<string | null>(null);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [validationMode, setValidationMode] = useState(false);
  const [resultsMode, setResultsMode] = useState(false);
  const [structuralAuditMode, setStructuralAuditMode] = useState(false);
  const currentImage = images[imageIndex];

  async function handleFiles(fileList: FileList | null) {
    if (!fileList) return;
    const files = Array.from(fileList).filter((file) => file.type.startsWith('image/'));
    if (!files.length) { setError('Choose one or more image files.'); return; }
    try {
      const loaded = await Promise.all(files.map(readImage));
      images.forEach((image) => URL.revokeObjectURL(image.src));
      setImages(loaded);
      setImageIndex(0);
      setAnnotation(blankAnnotation);
      setValidationMode(false);
      setResultsMode(false);
      setStructuralAuditMode(false);
      setSavedMessage(null);
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'One of the selected images could not be read.');
    }
  }

  function selectImage(index: number) {
    setImageIndex(index);
    setAnnotation(blankAnnotation);
    setValidationMode(false);
    setResultsMode(false);
    setStructuralAuditMode(false);
    setSavedMessage(null);
  }

  function imageId(file: File): string {
    return file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]+/g, '-');
  }

  function updateAnnotation(patch: Partial<AnnotationDraft>) {
    setAnnotation((current) => ({ ...current, ...patch }));
  }

  return <div className="app-shell annotation-app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">P</div><div><strong>PERIO<span>LAB</span></strong><small>Expert annotation</small></div></div>
      <div className="study-card"><span className="eyebrow">ACTIVE STUDY</span><strong>Localized periodontal disease</strong><span className="study-id">RESEARCH PROTOTYPE</span></div>
      <div className="sidebar-foot"><span className="status-dot" />Local prototype <small>No clinical connection</small></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div><span className="breadcrumb">RESEARCH WORKSPACE / STRUCTURAL REVIEW</span><h1>Oral assessment and treatment planning</h1></div><span className="pill neutral">RESEARCH ONLY</span></header>
      <div className="content">
        {!currentImage ? <section className="annotation-intake"><span className="eyebrow">IMAGE INTAKE</span><h2>Review DenPAR structural annotations</h2><p>Load a radiograph to inspect source-aligned tooth masks, boxes, landmarks, and bone lines. Structural annotations do not establish disease labels.</p><label className="dropzone"><span className="upload-icon">↑</span><strong>Choose radiographs</strong><small>Multiple images supported for navigation</small><input type="file" accept="image/*" multiple onChange={(event) => void handleFiles(event.target.files)} /></label>{error && <div className="save-error" role="alert">{error}</div>}</section> : <>
          <div className="annotation-casebar"><div><span className="eyebrow">CASE {imageIndex + 1} OF {images.length}</span><h2>{currentImage.file.name}</h2><span>{currentImage.width} x {currentImage.height} px · {imageId(currentImage.file)}</span></div><div className="casebar-actions"><button className="secondary-btn" onClick={() => { setStructuralAuditMode(false); setResultsMode(true); }}>Tooth segmentation</button><button className="secondary-btn" onClick={() => setStructuralAuditMode(true)}>Structural audit</button><label className="secondary-btn upload-more">Add images<input type="file" accept="image/*" multiple onChange={(event) => void handleFiles(event.target.files)} /></label></div></div>
          {savedMessage && <div className="notice teal"><span>✓</span><p>{savedMessage}</p></div>}
          {structuralAuditMode ? <StructuralAuditViewer imageId={imageId(currentImage.file)} imageSrc={currentImage.src} onBack={() => setStructuralAuditMode(false)} /> : resultsMode ? <ResultsPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} imageSrc={currentImage.src} imageFile={currentImage.file} annotation={annotation} onBack={() => setResultsMode(false)} /> : validationMode ? <ExpertValidationPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} originalSrc={currentImage.src} annotation={annotation} onChange={updateAnnotation} onEdit={() => setValidationMode(false)} onSaved={(timestamp, status) => { updateAnnotation({ validationStatus: status, validationTimestamp: timestamp, validated: status === 'Validated', savedAt: timestamp }); setSavedMessage(`Validation saved as ${status}. Ground truth remains attributable to expert review.`); setResultsMode(true); }} /> : <ExpertAnnotationPanel imageId={imageId(currentImage.file)} imageName={currentImage.file.name} imageSrc={currentImage.src} annotation={annotation} onChange={updateAnnotation} onSaved={(savedAt) => { updateAnnotation({ savedAt, validationStatus: 'Pending' }); setSavedMessage('Annotation saved to the FastAPI research backend. Preliminary analysis remains separate from ground truth.'); }} onContinue={() => setValidationMode(true)} canPrevious={imageIndex > 0} canNext={imageIndex < images.length - 1} imageIndex={imageIndex} imageCount={images.length} onPrevious={() => selectImage(imageIndex - 1)} onNext={() => selectImage(imageIndex + 1)} />}
        </>}
      </div>
    </main>
  </div>;
}
