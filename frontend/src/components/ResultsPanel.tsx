import { useEffect, useState } from 'react';
import { getToothSegmentation } from '../lib/annotations';
import type { AnnotationDraft, ToothSegmentationResponse } from '../types';

interface ResultsPanelProps {
  imageId: string; imageName: string; imageSrc: string; imageFile: File;
  annotation: AnnotationDraft; onBack: () => void;
}

export default function ResultsPanel({ imageId, imageName, imageSrc, imageFile, onBack }: ResultsPanelProps) {
  const [result, setResult] = useState<ToothSegmentationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [masks, setMasks] = useState(true);
  const [boxes, setBoxes] = useState(true);
  const [labels, setLabels] = useState(true);
  const [mode, setMode] = useState('Model Prediction');
  useEffect(() => {
    const controller = new AbortController();
    setResult(null); setError(null); setMode('Model Prediction');
    void getToothSegmentation(imageId, imageFile, controller.signal).then(value => {
      if (!controller.signal.aborted) setResult(value);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Segmentation unavailable.');
    });
    return () => controller.abort();
  }, [imageId, imageFile]);
  const prediction = mode !== 'Ground Truth';
  const truth = mode !== 'Model Prediction';
  return <section className="results-page">
    <div className="results-heading"><div><h2>MODEL TOOTH SEGMENTATION</h2><p>{imageName}</p></div><button className="secondary-btn" onClick={onBack}>Back to review</button></div>
    <div className="research-disclaimer">This system is a research prototype intended to support oral assessment and treatment planning research. Its outputs are not clinically validated and should not be used as a standalone diagnosis or treatment recommendation.</div>
    <p className="research-disclaimer">Research prototype. Tooth segmentation output does not indicate periodontal disease, severity, or treatment need.</p>
    {!result && !error && <p role="status">Running tooth segmentation...</p>}
    {error && <p role="alert">{error}</p>}
    {result?.model_status === 'unavailable' && <p role="status">Tooth segmentation model unavailable. No predictions generated.</p>}
    {result?.model_status === 'available' && <>
      <p data-testid="instance-count">{result.instances.length} detected tooth instances / {result.model_version} / confidence threshold {result.confidence_threshold}</p>
      <div className="segmentation-controls">
        <label><input type="checkbox" checked={masks} onChange={e => setMasks(e.target.checked)} /> Show masks</label>
        <label><input type="checkbox" checked={boxes} onChange={e => setBoxes(e.target.checked)} /> Show bounding boxes</label>
        <label><input type="checkbox" checked={labels} onChange={e => setLabels(e.target.checked)} /> Show confidence labels</label>
        {result.ground_truth && <label>Comparison mode <select value={mode} onChange={e => setMode(e.target.value)}>{['Model Prediction', 'Ground Truth', 'Overlay Comparison'].map(value => <option key={value}>{value}</option>)}</select></label>}
      </div>
      <p>Prediction: green. Ground truth: magenta. Instance numbers are local detection IDs, not tooth identities.</p>
      {truth && result.ground_truth && <p>{result.ground_truth.split} ground truth: {result.ground_truth.source}</p>}
    </>}
    {result?.model_status === 'available' ? <svg data-testid="segmentation-viewer" viewBox={`0 0 ${result.width} ${result.height}`} style={{ width: '100%', height: 'auto', display: 'block', background: '#111' }} aria-label="Original radiograph with tooth segmentation overlays" role="img">
      <defs><filter id="prediction-color"><feFlood floodColor="#34ed92" /><feComposite in2="SourceAlpha" operator="in" /></filter><filter id="truth-color"><feFlood floodColor="#fa65e2" /><feComposite in2="SourceAlpha" operator="in" /></filter></defs>
      <image data-testid="original-image" href={imageSrc} width={result.width} height={result.height} />
      {prediction && result.instances.map(instance => <g key={instance.instance_id} data-instance={instance.instance_id}>
        {masks && <image data-testid="prediction-mask" href={instance.mask_url} width={result.width} height={result.height} opacity="0.4" filter="url(#prediction-color)" />}
        {boxes && <rect data-testid="prediction-box" x={instance.bbox[0]} y={instance.bbox[1]} width={instance.bbox[2]-instance.bbox[0]} height={instance.bbox[3]-instance.bbox[1]} fill="none" stroke="#34ed92" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
        {labels && <text data-testid="confidence-label" x={instance.bbox[0]} y={Math.max(18, instance.bbox[1])} fill="white" stroke="black" strokeWidth="3" paintOrder="stroke" fontSize={Math.max(16, result.width / 65)}>{instance.instance_id}: {(instance.confidence*100).toFixed(2)}%</text>}
      </g>)}
      {truth && masks && result.ground_truth?.mask_urls.map((url, index) => <image data-testid="truth-mask" key={index} href={url} width={result.width} height={result.height} opacity="0.4" filter="url(#truth-color)" />)}
    </svg> : <img src={imageSrc} alt={`Original radiograph ${imageName}`} style={{ width: '100%', height: 'auto' }} />}
    {result?.model_status === 'available' && <ul>{result.instances.map(instance => <li key={instance.instance_id}>Instance {instance.instance_id}: confidence {(instance.confidence*100).toFixed(2)}%</li>)}</ul>}
  </section>;
}
