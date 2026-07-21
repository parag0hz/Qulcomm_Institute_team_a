import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { AnalysisResponse, DesignParameters, ParameterSchema, PredictionResponse, Recommendation, Variant } from "../types";
import { CopilotPanel } from "./CopilotPanel";

type ResultTab = "performance" | "drivers" | "confidence" | "copilot";

interface StatusShape {
  model_status: string;
  model_metrics: { r2?: number; mae?: number };
  providers?: { vertex?: { available?: boolean; location?: string } };
}

interface ResultsPanelProps {
  design: DesignParameters;
  schema: ParameterSchema;
  status: StatusShape;
  prediction: PredictionResponse | null;
  analysis: AnalysisResponse | null;
  baselineCd: number | null;
  locks: string[];
  variants: Variant[];
  predictionMode?: "parameters" | "stl";
  resultBusy?: boolean;
  onFocusParameter: (name: string) => void;
  onSetBaseline: () => void;
  onPreview: (recommendation: Recommendation) => void;
  onApply: (recommendation: Recommendation) => void;
  onCancelPreview: () => void;
  onLoadVariant: (id: string) => void;
  onDeleteVariant: (id: string) => void;
  onClearWorkspace: () => void;
  onPrint: () => void;
}

export function ResultsPanel(props: ResultsPanelProps) {
  const [tab, setTab] = useState<ResultTab>("performance");
  const [targetCd, setTargetCd] = useState(0.24);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [optimizing, setOptimizing] = useState(false);
  const [vertexResult, setVertexResult] = useState("");
  const [compareLeft, setCompareLeft] = useState("");
  const [compareRight, setCompareRight] = useState("");
  const [optimizationBase, setOptimizationBase] = useState("");
  const optimizeAbortRef = useRef<AbortController | null>(null);
  const currentDesignRef = useRef(props.design);
  currentDesignRef.current = props.design;
  const predictionMode = props.predictionMode ?? "parameters";
  const tabs: ResultTab[] = predictionMode === "stl"
    ? ["performance", "confidence"]
    : ["performance", "drivers", "confidence", "copilot"];
  const prediction = props.prediction;
  const domain = prediction?.domain_status || "inside";
  const warnings = prediction?.warnings || [];
  const delta = prediction && props.baselineCd != null ? prediction.cd - props.baselineCd : null;
  const deltaPercent = delta != null && props.baselineCd ? delta / props.baselineCd * 100 : null;
  const designFingerprint = JSON.stringify(props.design);

  useEffect(() => {
    if (!tabs.includes(tab)) setTab("performance");
  }, [predictionMode, tab]);

  useEffect(() => {
    if (!optimizationBase) return;
    const belongsToSearch = designFingerprint === optimizationBase
      || recommendations.some((item) => JSON.stringify(item.parameters) === designFingerprint);
    if (!belongsToSearch) {
      optimizeAbortRef.current?.abort();
      setRecommendations([]);
      setOptimizationBase("");
      setOptimizing(false);
    }
  }, [designFingerprint, optimizationBase, recommendations]);

  useEffect(() => () => optimizeAbortRef.current?.abort(), []);

  const comparison = useMemo(() => {
    const left = props.variants.find((item) => item.id === compareLeft);
    const right = props.variants.find((item) => item.id === compareRight);
    if (!left || !right) return null;
    const changes = props.schema.parameters.filter((parameter) => Number(left.design[parameter.name]) !== Number(right.design[parameter.name]));
    return { left, right, changes };
  }, [compareLeft, compareRight, props.schema.parameters, props.variants]);

  const optimize = async () => {
    optimizeAbortRef.current?.abort();
    const controller = new AbortController();
    optimizeAbortRef.current = controller;
    const base = JSON.stringify(props.design);
    setOptimizationBase(base);
    setRecommendations([]);
    setOptimizing(true);
    try {
      const response = await api.optimize(
        { parameters: props.design, target_cd: targetCd, locked: props.locks },
        controller.signal,
      );
      if (!controller.signal.aborted && JSON.stringify(currentDesignRef.current) === base) {
        setRecommendations(response.recommendations);
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setRecommendations([]);
        window.alert(error instanceof Error ? error.message : "Optimization failed.");
      }
    } finally {
      if (!controller.signal.aborted) setOptimizing(false);
    }
  };

  const testVertex = async () => {
    setVertexResult("Testing endpoint…");
    try {
      const result = await api.testVertex({ parameters: props.design });
      setVertexResult(`Connected · Cd ${Number(result.prediction).toFixed(4)} · ${result.latency_ms} ms`);
    } catch (error) { setVertexResult(error instanceof Error ? error.message : "Vertex connection failed."); }
  };

  return (
    <aside className="performance-panel" data-risk={domain}>
      <div className="performance-head"><div><p className="eyebrow">Performance</p><h2>Aerodynamic result</h2></div><span className={`confidence-badge ${domain === "inside" ? "good" : ""}`}>{prediction ? `${domain} domain` : "Model loading"}</span></div>
      <div className="result-tabs" role="tablist" style={{ gridTemplateColumns: `repeat(${tabs.length}, 1fr)` }}>{tabs.map((item) => <button key={item} className={`result-tab ${tab === item ? "active" : ""}`} type="button" onClick={() => setTab(item)}>{item[0].toUpperCase() + item.slice(1)}</button>)}</div>
      {(domain !== "inside" || warnings.length > 0) && <div className={`risk-banner ${domain}`}>{warnings[0] || "Review this result before selecting the concept."}</div>}

      {tab === "performance" && <div className="result-pane active">
        <section className="cd-hero"><div className="cd-value"><span>Estimated Cd</span><strong>{prediction?.cd.toFixed(4) ?? "—"}</strong></div>
          <div className={`delta-badge ${delta == null || Math.abs(delta) < .00005 ? "neutral" : delta < 0 ? "better" : "worse"}`}>{delta == null ? "Waiting for baseline" : `${delta <= 0 ? "↓" : "↑"} ${Math.abs(deltaPercent || 0).toFixed(1)}% vs baseline · ${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`}</div>
          <p>{prediction?.comparison || "The model will compare this concept with the DrivAerNet design population."}</p>
        </section>
        <section className="benchmark-card"><div className="benchmark-head"><span>Dataset position</span><strong>{prediction ? `P${Math.round(prediction.percentile)}` : "—"}</strong></div><div className="range-track"><span className="range-zone good" /><span className="range-zone average" /><span className="range-zone high" /><span className="cd-marker" style={{ left: `${prediction?.percentile ?? 50}%` }} /></div><div className="range-labels"><span>{prediction?.dataset.cd_min.toFixed(3) ?? "0.200"}</span><span>Median {prediction?.dataset.cd_median.toFixed(3) ?? "0.252"}</span><span>{prediction?.dataset.cd_max.toFixed(3) ?? "0.320"}</span></div></section>
        {predictionMode === "parameters" ? <section className="model-card"><div className="card-title"><span>Prediction quality</span><small>{props.status.model_status}</small></div><dl><div><dt>Validation R²</dt><dd>{props.status.model_metrics.r2?.toFixed(3) ?? "—"}</dd></div><div><dt>Validation MAE</dt><dd>{props.status.model_metrics.mae?.toFixed(4) ?? "—"}</dd></div><div><dt>Features used</dt><dd>25</dd></div></dl><p>{domain === "inside" ? "Inputs have a nearby sample in the observed training domain." : warnings[0] || "Review domain confidence."}</p></section>
          : <section className="model-card"><div className="card-title"><span>Prediction quality</span><small>Geometry fallback</small></div><dl><div><dt>Training</dt><dd>None</dd></div><div><dt>Input</dt><dd>STL</dd></div><div><dt>Use</dt><dd>Screening</dd></div></dl><p>This mesh estimate is based on geometric proportions, not the trained 25-feature surrogate.</p></section>}
        {predictionMode === "parameters" && <button className="baseline-button" type="button" disabled={props.resultBusy || !prediction} onClick={props.onSetBaseline}>Set current design as baseline</button>}
      </div>}

      {tab === "drivers" && <div className="result-pane active">
        <section className="driver-card"><div className="card-title"><span>Local sensitivity</span><small>±5% design range</small></div><p className="helper-copy">Select a driver to jump to its control and highlight the 3D region.</p><div className="driver-chart">{props.analysis?.drivers.slice(0, 8).map((driver) => { const max = props.analysis?.drivers[0]?.impact || 1; const direction = Math.abs(driver.minus_delta) > Math.abs(driver.plus_delta) ? driver.minus_delta : driver.plus_delta; return <button key={driver.name} className="driver-row" type="button" onClick={() => props.onFocusParameter(driver.name)}><span>{driver.label}</span><span className="driver-bar"><i style={{ width: `${driver.impact / max * 100}%` }} /></span><strong className={direction < 0 ? "benefit" : "penalty"}>{direction >= 0 ? "+" : ""}{direction.toFixed(4)}</strong></button>; }) || <p className="empty-state">Calculating sensitivity…</p>}</div></section>
        <section className="optimizer-card"><div className="card-title"><span>Target Cd assistant</span><small>Surrogate guidance</small></div><label>Target Cd<input type="number" min="0.18" max="0.36" step="0.001" value={targetCd} onChange={(event) => setTargetCd(Number(event.target.value))} /></label><p className="helper-copy">Locked parameters remain unchanged.</p><button className="primary-button" type="button" disabled={optimizing} onClick={() => void optimize()}>{optimizing ? "Searching…" : "Find improvement directions"}</button><div className="recommendations">{recommendations.map((item, index) => <article className="recommendation" key={`${item.cd}-${index}`}><div><strong>Direction {index + 1} · Cd {item.cd.toFixed(4)}</strong><span>{item.improvement >= 0 ? "−" : "+"}{Math.abs(item.improvement).toFixed(4)} · {item.domain_status}</span></div><p>{item.changes.slice(0, 3).map((change) => change.label).join(", ") || "No change"}</p><button type="button" onClick={() => props.onPreview(item)}>Preview</button><button type="button" onClick={() => props.onApply(item)}>Apply</button></article>)}</div>{recommendations.length > 0 && <button className="cancel-preview" type="button" onClick={props.onCancelPreview}>Cancel preview</button>}</section>
      </div>}

      {tab === "confidence" && <div className="result-pane active"><section className="confidence-card"><div className="confidence-grid"><div><span>Provider</span><strong>{prediction?.provider || "—"}</strong></div><div><span>Domain</span><strong>{domain}</strong></div><div><span>Nearest distance</span><strong>{Number.isFinite(prediction?.nearest_sample_distance) ? prediction?.nearest_sample_distance.toFixed(3) : "—"}</strong></div><div><span>95% interval</span><strong>{prediction?.uncertainty ? `${prediction.uncertainty.lower.toFixed(3)}–${prediction.uncertainty.upper.toFixed(3)}` : "—"}</strong></div></div><div className="warning-list">{warnings.length ? warnings.map((warning) => <p key={warning}>! {warning}</p>) : <p>✓ Inputs are represented by nearby training samples.</p>}</div></section>{predictionMode === "parameters" ? <section className="model-card confidence-model"><div className="card-title"><span>Provider readiness</span><small>Local + Vertex AI</small></div><p>{props.status.providers?.vertex?.available ? `Vertex AI configured in ${props.status.providers.vertex.location}.` : "Local model active; Vertex is not configured."}</p><button className="baseline-button" type="button" onClick={() => void testVertex()}>Test Vertex connection</button>{vertexResult && <div className={`connection-result ${vertexResult.startsWith("Connected") ? "success" : "error"}`}>{vertexResult}</div>}</section> : <section className="model-card"><div className="card-title"><span>STL confidence</span><small>Low</small></div><p>No training-domain distance or statistical uncertainty is available for the geometric fallback. Use this result only for early screening.</p></section>}</div>}
      {tab === "copilot" && <div className="result-pane active"><CopilotPanel design={props.design} /></div>}

      {predictionMode === "parameters" && <section className="variant-workspace"><div className="card-title"><span>Design variants</span><small>{props.variants.length}/5 saved</small></div><div className="variant-list">{props.variants.length ? props.variants.map((variant) => <article className="variant-card" key={variant.id}>{variant.thumbnail ? <img src={variant.thumbnail} alt={`${variant.name} thumbnail`} /> : <div />}<div><strong>{variant.name}</strong><span>Cd {variant.cd?.toFixed(4) ?? "—"} · {variant.design.CarRear}</span></div><button type="button" onClick={() => props.onLoadVariant(variant.id)}>Load</button><button type="button" onClick={() => props.onDeleteVariant(variant.id)}>×</button></article>) : <p className="empty-state">Save a concept to compare alternatives.</p>}</div><div className="compare-controls"><select value={compareLeft} onChange={(event) => setCompareLeft(event.target.value)}><option value="">Select variant</option>{props.variants.map((variant) => <option value={variant.id} key={variant.id}>{variant.name}</option>)}</select><span>vs</span><select value={compareRight} onChange={(event) => setCompareRight(event.target.value)}><option value="">Select variant</option>{props.variants.map((variant) => <option value={variant.id} key={variant.id}>{variant.name}</option>)}</select></div>{comparison && <div className="comparison-table"><div className="compare-summary"><strong>{comparison.left.name} {comparison.left.cd?.toFixed(4)}</strong><span>↔</span><strong>{comparison.right.name} {comparison.right.cd?.toFixed(4)}</strong></div>{comparison.changes.slice(0, 8).map((parameter) => <div key={parameter.name}><span>{parameter.label}</span><strong>{Number(comparison.left.design[parameter.name]).toFixed(2)} → {Number(comparison.right.design[parameter.name]).toFixed(2)}</strong></div>)}</div>}<button className="clear-workspace" type="button" onClick={props.onClearWorkspace}>Clear all variants and history</button></section>}
      <div className="report-actions"><button className="ghost-button" type="button" onClick={props.onPrint}>Print / Save PDF</button></div>
    </aside>
  );
}
