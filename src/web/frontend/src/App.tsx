import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";

import "./styles.css";
import { api } from "./api";
import { DesignControls } from "./components/DesignControls";
import { Header } from "./components/Header";
import { HoldoutBenchmark } from "./components/HoldoutBenchmark";
import { ResultsPanel } from "./components/ResultsPanel";
import type { VehicleViewerHandle } from "./components/VehicleViewer";
import { ViewerErrorBoundary } from "./components/ViewerErrorBoundary";
import {
  selectCanRedo,
  selectCanUndo,
  useWorkspaceStore,
} from "./store";
import type {
  AnalysisResponse,
  CarRear,
  CloudPredictionResponse,
  DatasetSummary,
  DesignParameters,
  NumericParameterName,
  ParameterSchema,
  PredictionResponse,
  Recommendation,
  StatusResponse,
  StlPredictionResponse,
  WheelTreatment,
} from "./types";

type InputMode = "parameters" | "stl";
type CameraView = "perspective" | "side" | "front" | "top";

const VehicleViewer = lazy(() => import("./components/VehicleViewer").then((module) => ({
  default: module.VehicleViewer,
})));

const htmlEscape = (value: unknown): string => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

function defaultDesign(schema: ParameterSchema): DesignParameters {
  const numeric = Object.fromEntries(
    schema.parameters.map((parameter) => [parameter.name, parameter.default]),
  ) as Pick<DesignParameters, NumericParameterName>;
  const configuration = schema.valid_combinations.find(
    (item) => item.CarRear === "Fastback" && item.Wheels === "Closed smooth",
  ) ?? schema.valid_combinations[0];
  return {
    ...numeric,
    CarRear: configuration?.CarRear ?? schema.categories.CarRear[0] ?? "Fastback",
    Wheels: configuration?.Wheels ?? schema.categories.Wheels[0] ?? "Open detailed",
  } as DesignParameters;
}

function downloadJson(name: string, payload: unknown): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function designFingerprint(design: DesignParameters): string {
  return JSON.stringify(design);
}

function stlDisplayPrediction(result: StlPredictionResponse): PredictionResponse {
  return {
    cd: result.cd,
    percentile: result.percentile,
    level: result.level,
    comparison: result.comparison,
    provider: "Geometry fallback",
    domain_status: "outside",
    nearest_sample_distance: Number.NaN,
    uncertainty: null,
    warnings: [
      "Imported STL uses an untrained geometric-proportion estimate. Validate the concept with CFD.",
    ],
    model: result.model,
    dataset: result.dataset,
  };
}

// 업로드한 .paddle_tensor를 PointNet(학습된 형상 모델)이 예측한 결과를 스튜디오의
// 공통 결과 표시에 맞춰 변환한다. STL 휴리스틱과 달리 이건 실제 대체모델이므로
// 라벨을 "trained surrogate"로 둔다. 백분위는 데이터셋 범위로 근사한다.
function cloudDisplayPrediction(
  result: CloudPredictionResponse,
  dataset?: DatasetSummary,
): PredictionResponse {
  const cd = result.cd ?? result.raw_cd;
  const lo = dataset?.cd_min ?? 0.2;
  const hi = dataset?.cd_max ?? 0.36;
  const percentile = Math.max(0, Math.min(100, ((cd - lo) / (hi - lo || 1)) * 100));
  const level: PredictionResponse["level"] =
    cd < (dataset?.cd_p25 ?? 0.24) ? "low" : cd > (dataset?.cd_p75 ?? 0.3) ? "high" : "medium";
  return {
    cd,
    percentile,
    level,
    comparison: "PointNet estimated this coefficient directly from the uploaded point cloud.",
    provider: "PointNet (point cloud)",
    domain_status: result.trusted ? "inside" : "outside",
    nearest_sample_distance: Number.NaN,
    uncertainty: null,
    warnings: result.warnings,
    model: { name: "PointNet", status: "connected", confidence: "high" },
    dataset:
      dataset ?? {
        sample_count: 0,
        cd_min: 0.2,
        cd_max: 0.36,
        cd_mean: 0.256,
        cd_median: 0.252,
        cd_p25: 0.24,
        cd_p75: 0.3,
        feature_count: 3,
      },
  };
}

// VehicleViewer는 정규화된 STL preview([-1,1])에 맞춰 카메라·점 크기가 고정돼 있고
// 위치만 geometry.center()로 맞춘다(스케일 보정 없음). 미터 스케일 원본 점군을 그대로
// 주면 2배 이상 커 보이므로, STL의 normalize_preview_points와 동일하게 정규화한다.
function normalizePreviewPoints(points: number[][]): number[][] {
  if (!points.length) return points;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const [x, y, z] of points) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-9);
  return points.map(([x, y, z]) => [((x - cx) / span) * 2, ((y - cy) / span) * 2, ((z - cz) / span) * 2]);
}

export default function App() {
  useEffect(() => {
    document.body.classList.add("paragon-studio");
    return () => document.body.classList.remove("paragon-studio");
  }, []);

  const [benchmarkOpen, setBenchmarkOpen] = useState(false);
  const [schema, setSchema] = useState<ParameterSchema | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [predictionFingerprint, setPredictionFingerprint] = useState("");
  const [analysisFingerprint, setAnalysisFingerprint] = useState("");
  const [draftDesign, setDraftDesign] = useState<DesignParameters | null>(null);
  const [activeParameter, setActiveParameter] = useState<NumericParameterName>("A_Car_Length");
  const [mode, setMode] = useState<InputMode>("parameters");
  const [cameraView, setCameraView] = useState<CameraView>("perspective");
  const [dimensionsVisible, setDimensionsVisible] = useState(true);
  const [predictionBusy, setPredictionBusy] = useState(false);
  const [stlBusy, setStlBusy] = useState(false);
  const [stlResult, setStlResult] = useState<StlPredictionResponse | null>(null);
  const [cloudResult, setCloudResult] = useState<CloudPredictionResponse | null>(null);
  const [startupError, setStartupError] = useState("");
  const [requestError, setRequestError] = useState("");
  const [toast, setToast] = useState("");

  const viewerRef = useRef<VehicleViewerHandle>(null);
  const historyCommitRef = useRef<number | null>(null);
  const draftDesignRef = useRef<DesignParameters | null>(null);
  const storedCurrent = useWorkspaceStore((state) => state.current);
  const baseline = useWorkspaceStore((state) => state.baseline);
  const locks = useWorkspaceStore((state) => state.locks);
  const variants = useWorkspaceStore((state) => state.variants);
  const designName = useWorkspaceStore((state) => state.designName);
  const canUndo = useWorkspaceStore(selectCanUndo);
  const canRedo = useWorkspaceStore(selectCanRedo);
  const initializeDesign = useWorkspaceStore((state) => state.initializeDesign);
  const setCurrentDesign = useWorkspaceStore((state) => state.setCurrentDesign);
  const setBaseline = useWorkspaceStore((state) => state.setBaseline);
  const setBaselineCd = useWorkspaceStore((state) => state.setBaselineCd);
  const setDesignName = useWorkspaceStore((state) => state.setDesignName);
  const undo = useWorkspaceStore((state) => state.undo);
  const redo = useWorkspaceStore((state) => state.redo);
  const toggleLock = useWorkspaceStore((state) => state.toggleLock);
  const saveVariant = useWorkspaceStore((state) => state.saveVariant);
  const deleteVariant = useWorkspaceStore((state) => state.deleteVariant);
  const loadVariant = useWorkspaceStore((state) => state.loadVariant);
  const previewRecommendation = useWorkspaceStore((state) => state.previewRecommendation);
  const applyRecommendation = useWorkspaceStore((state) => state.applyRecommendation);
  const cancelRecommendation = useWorkspaceStore((state) => state.cancelRecommendation);
  const clearWorkspace = useWorkspaceStore((state) => state.clearWorkspace);
  const current = draftDesign ?? storedCurrent;

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([api.getParameters(controller.signal), api.getStatus(controller.signal)])
      .then(([parameterSchema, serviceStatus]) => {
        setSchema(parameterSchema);
        setStatus(serviceStatus);
        initializeDesign(defaultDesign(parameterSchema));
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setStartupError(error instanceof Error ? error.message : "Unable to initialize Paragon.");
        }
      });
    return () => controller.abort();
  }, [initializeDesign]);

  useEffect(() => () => {
    if (historyCommitRef.current !== null) window.clearTimeout(historyCommitRef.current);
  }, []);

  useEffect(() => {
    draftDesignRef.current = null;
    setDraftDesign(null);
  }, [storedCurrent]);

  useEffect(() => {
    if (!current || mode !== "parameters") {
      setPredictionBusy(false);
      return;
    }
    const controller = new AbortController();
    const fingerprint = designFingerprint(current);
    setPredictionBusy(true);
    setRequestError("");
    const timer = window.setTimeout(() => {
      api.predict(current, controller.signal)
        .then((result) => {
          if (!controller.signal.aborted) {
            setPrediction(result);
            setPredictionFingerprint(fingerprint);
          }
        })
        .catch((error: unknown) => {
          if (!controller.signal.aborted) {
            setRequestError(error instanceof Error ? error.message : "Prediction failed.");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setPredictionBusy(false);
        });
    }, 220);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [current, mode]);

  useEffect(() => {
    if (!current || mode !== "parameters") return;
    const controller = new AbortController();
    const fingerprint = designFingerprint(current);
    const timer = window.setTimeout(() => {
      api.analyze(current, controller.signal)
        .then((result) => {
          if (!controller.signal.aborted) {
            setAnalysis(result);
            setAnalysisFingerprint(fingerprint);
          }
        })
        .catch(() => {
          if (!controller.signal.aborted) setAnalysis(null);
        });
    }, 520);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [current, mode]);

  useEffect(() => {
    if (!baseline || baseline.cd !== null) return;
    const controller = new AbortController();
    api.predict(baseline.design, controller.signal)
      .then((result) => setBaselineCd(result.cd))
      .catch(() => undefined);
    return () => controller.abort();
  }, [baseline, setBaselineCd]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const baseDefaults = useMemo(() => schema ? defaultDesign(schema) : null, [schema]);
  const activeDefinition = schema?.parameters.find((parameter) => parameter.name === activeParameter);
  const baselineCd = baseline?.cd ?? null;
  const currentFingerprint = current ? designFingerprint(current) : "";
  const currentPrediction = predictionFingerprint === currentFingerprint ? prediction : null;
  const currentAnalysis = analysisFingerprint === currentFingerprint ? analysis : null;
  const displayPrediction = mode === "stl"
    ? (cloudResult
        ? cloudDisplayPrediction(cloudResult, status?.dataset)
        : stlResult
          ? stlDisplayPrediction(stlResult)
          : null)
    : currentPrediction;
  const displayAnalysis = mode === "parameters" ? currentAnalysis : null;
  const resultBusy = mode === "stl"
    ? stlBusy
    : predictionBusy || currentPrediction === null;

  const clearHistoryCommit = () => {
    if (historyCommitRef.current !== null) window.clearTimeout(historyCommitRef.current);
    historyCommitRef.current = null;
  };

  const commitDraft = (): DesignParameters | null => {
    clearHistoryCommit();
    const latest = draftDesignRef.current;
    if (!latest) return storedCurrent;
    setCurrentDesign(latest, { record: true });
    draftDesignRef.current = null;
    setDraftDesign(null);
    return latest;
  };

  const commitSliderHistory = () => {
    clearHistoryCommit();
    historyCommitRef.current = window.setTimeout(() => {
      const latest = draftDesignRef.current;
      if (latest) {
        useWorkspaceStore.getState().setCurrentDesign(latest, { record: true });
        draftDesignRef.current = null;
        setDraftDesign(null);
      }
      historyCommitRef.current = null;
    }, 320);
  };

  const handleParameterChange = (name: string, value: number) => {
    if (!schema || !current) return;
    const definition = schema.parameters.find((parameter) => parameter.name === name);
    if (!definition || !Number.isFinite(value)) return;
    const bounded = Math.min(definition.max, Math.max(definition.min, value));
    const next = { ...current, [definition.name]: bounded };
    setActiveParameter(definition.name);
    draftDesignRef.current = next;
    setDraftDesign(next);
    commitSliderHistory();
  };

  const handleCategoryChange = (name: "CarRear" | "Wheels", value: string) => {
    if (!current) return;
    clearHistoryCommit();
    const next = name === "CarRear"
      ? { ...current, CarRear: value as CarRear }
      : { ...current, Wheels: value as WheelTreatment };
    draftDesignRef.current = null;
    setDraftDesign(null);
    setCurrentDesign(next, { record: true });
  };

  const focusParameter = (name: string) => {
    const definition = schema?.parameters.find((parameter) => parameter.name === name);
    if (!definition) return;
    setActiveParameter(definition.name);
    document.querySelector(`[data-control="${CSS.escape(name)}"]`)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  };

  const handlePreset = (id: string) => {
    const preset = schema?.presets.find((item) => item.id === id);
    if (!preset) return;
    clearHistoryCommit();
    draftDesignRef.current = null;
    setDraftDesign(null);
    setCurrentDesign(preset.design);
    setActiveParameter("A_Car_Length");
    setToast(`${preset.name} applied`);
  };

  const handleStlUpload = async (file: File) => {
    const isCloud = file.name.toLowerCase().endsWith(".paddle_tensor");
    setStlBusy(true);
    setRequestError("");
    try {
      if (isCloud) {
        const result = await api.uploadCloud(file);
        setCloudResult(result);
        setStlResult(null);
        setToast(`Read ${result.file.name}`);
      } else {
        const result = await api.uploadStl(file);
        setStlResult(result);
        setCloudResult(null);
        setToast(`Analyzed ${result.file.name}`);
      }
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setStlBusy(false);
    }
  };

  const saveCurrentVariant = () => {
    if (!current || mode !== "parameters") {
      setToast("STL analyses are exported as reports, not parameter variants");
      return;
    }
    if (resultBusy || !currentPrediction) {
      setToast("Wait for the current Cd prediction before saving");
      return;
    }
    commitDraft();
    const delta = baselineCd !== null ? currentPrediction.cd - baselineCd : null;
    const variant = saveVariant({
      cd: currentPrediction.cd,
      baselineDelta: delta,
      thumbnail: viewerRef.current?.captureThumbnail() ?? "",
    });
    setToast(variant ? `${variant.name} saved` : "You can save up to five variants");
  };

  const reportPayload = () => ({
    product: "Paragon Vehicle Design Studio",
    exported_at: new Date().toISOString(),
    design_name: designName,
    source_mode: mode,
    design: mode === "parameters" ? current : null,
    baseline: mode === "parameters" ? baseline : null,
    prediction: displayPrediction,
    sensitivity: displayAnalysis,
    stl_analysis: mode === "stl" ? stlResult : null,
    geometry: {
      reference: mode === "parameters" ? "DrivAer F_D_WM_WW_3532" : stlResult?.file.name,
      body_source: mode === "parameters" ? "dataset QEM reference mesh" : "user-provided STL preview",
      body_architecture_geometry: mode === "parameters" ? "Fastback reference" : "imported mesh",
      wheel_treatment: mode === "parameters" ? current?.Wheels : null,
      wheel_geometry_source: mode === "parameters" ? (current?.Wheels === "Open detailed" ? "dataset" : "procedural") : "imported STL",
      morphing: mode === "parameters" ? "approximate designer-preview morph; not CFD/CAD geometry" : "none",
    },
  });

  const exportDesign = () => {
    if ((mode === "parameters" && (resultBusy || !currentPrediction)) || (mode === "stl" && !stlResult)) {
      setToast("Wait for the current analysis before exporting");
      return;
    }
    if (mode === "parameters") commitDraft();
    downloadJson(`${designName.trim().replace(/[^a-z0-9]+/gi, "-").toLowerCase() || "paragon-design"}.json`, reportPayload());
    setToast("Design JSON exported");
  };

  const printReport = () => {
    if (!current || !schema) return;
    if ((mode === "parameters" && (resultBusy || !currentPrediction)) || (mode === "stl" && !stlResult)) {
      setToast("Wait for the current analysis before opening the report");
      return;
    }
    const report = window.open("", "paragon-report", "width=960,height=760");
    if (!report) {
      setToast("Allow pop-ups to open the report");
      return;
    }
    const rows = mode === "parameters" ? schema.parameters.map((parameter) => `
      <tr><td>${htmlEscape(parameter.label)}</td><td>${htmlEscape(current[parameter.name])}</td><td>${htmlEscape(baseline?.design[parameter.name] ?? "—")}</td></tr>
    `).join("") : "";
    const drivers = displayAnalysis?.drivers.slice(0, 8).map((driver) => `
      <tr><td>${htmlEscape(driver.label)}</td><td>${driver.minus_delta.toFixed(4)}</td><td>${driver.plus_delta.toFixed(4)}</td></tr>
    `).join("") ?? "";
    const configuration = mode === "parameters"
      ? `${htmlEscape(current.CarRear)} · ${htmlEscape(current.Wheels)} · ${current.Wheels === "Open detailed" ? "dataset wheel" : "procedural wheel visualization"}`
      : `${htmlEscape(stlResult?.file.name ?? "Imported STL")} · ${htmlEscape(stlResult?.mesh.triangle_count ?? "—")} triangles`;
    const detailSection = mode === "parameters"
      ? `<h2>Parameters</h2><table><thead><tr><th>Parameter</th><th>Selected</th><th>Baseline</th></tr></thead><tbody>${rows}</tbody></table><h2>Local sensitivity</h2><table><thead><tr><th>Driver</th><th>Decrease direction ΔCd</th><th>Increase direction ΔCd</th></tr></thead><tbody>${drivers}</tbody></table>`
      : `<h2>Imported mesh</h2><p>This estimate uses geometric proportions and is not a trained CFD surrogate. Validate the mesh with CFD before making a design decision.</p>`;
    report.document.write(`<!doctype html><html><head><title>${htmlEscape(designName)} · Paragon</title><style>
      body{font:14px/1.5 Inter,Arial,sans-serif;color:#1d2a32;margin:36px}header{border-bottom:3px solid #087f78;margin-bottom:24px}h1{margin:0}small{color:#667780}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.metrics div{padding:14px;background:#eef5f4;border-radius:8px}.metrics strong{display:block;font-size:22px}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border-bottom:1px solid #dce4e7;padding:8px;text-align:left}th{background:#f3f6f7}@media print{button{display:none}}
    </style></head><body><header><small>PARAGON VEHICLE DESIGN STUDIO</small><h1>${htmlEscape(designName)}</h1><p>${mode === "parameters" ? "Surrogate-model design report" : "Imported STL screening report"} · validate shortlisted concepts with CFD.</p></header>
    <section class="metrics"><div>Estimated Cd<strong>${displayPrediction?.cd.toFixed(4) ?? "—"}</strong></div><div>Baseline<strong>${mode === "parameters" ? baselineCd?.toFixed(4) ?? "—" : "N/A"}</strong></div><div>Provider<strong>${htmlEscape(displayPrediction?.provider ?? "—")}</strong></div><div>Domain<strong>${htmlEscape(displayPrediction?.domain_status ?? "—")}</strong></div></section>
    <h2>Configuration</h2><p>${configuration}</p>${detailSection}
    <h2>Confidence</h2><p>${htmlEscape(displayPrediction?.warnings.join(" ") || "Inputs have a nearby observed sample.")}</p><button onclick="window.print()">Print / Save PDF</button></body></html>`);
    report.document.close();
  };

  if (startupError) {
    return <main className="app-error"><h1>Paragon could not start</h1><p>{startupError}</p><p>Make sure FastAPI is running and the local model artifact is available.</p></main>;
  }
  if (!schema || !status || !current) return <main className="app-loading">Loading Paragon design workspace…</main>;

  const lengthDelta = current.A_Car_Length - (schema.parameters.find((item) => item.name === "A_Car_Length")?.default ?? current.A_Car_Length);
  const widthDelta = current.A_Car_Width - (schema.parameters.find((item) => item.name === "A_Car_Width")?.default ?? current.A_Car_Width);
  const heightDelta = current.A_Car_Roof_Height - (schema.parameters.find((item) => item.name === "A_Car_Roof_Height")?.default ?? current.A_Car_Roof_Height);
  const importedPoints = mode === "stl"
    ? (cloudResult ? normalizePreviewPoints(cloudResult.preview_points) : stlResult?.preview_points)
    : undefined;

  return (
    <>
      <Header
        designName={designName}
        modelLabel={displayPrediction?.provider ?? status.model_status}
        canUndo={Boolean(draftDesign) || canUndo}
        canRedo={!draftDesign && canRedo}
        canSaveVariant={mode === "parameters" && !resultBusy && currentPrediction !== null}
        onNameChange={setDesignName}
        onUndo={() => { commitDraft(); undo(); }}
        onRedo={() => { clearHistoryCommit(); draftDesignRef.current = null; setDraftDesign(null); redo(); }}
        onSaveVariant={saveCurrentVariant}
        onOpenBenchmark={() => setBenchmarkOpen(true)}
        onReset={() => {
          clearHistoryCommit();
          draftDesignRef.current = null;
          setDraftDesign(null);
          if (baseDefaults) setCurrentDesign(baseDefaults);
          setMode("parameters");
          setStlResult(null);
          setCloudResult(null);
          viewerRef.current?.showReference();
          viewerRef.current?.resetView();
          setToast("Current design reset");
        }}
        onExport={exportDesign}
      />
      <main className="studio-shell">
        <DesignControls
          schema={schema}
          design={current}
          activeParameter={activeParameter}
          locked={locks}
          mode={mode}
          stlBusy={stlBusy}
          stlSummary={cloudResult ? `${cloudResult.file.name} · PointNet · ${cloudResult.cd !== null ? `Cd ${cloudResult.cd.toFixed(4)}` : "out of distribution"}` : stlResult ? `${stlResult.file.name} · ${Number(stlResult.mesh.triangle_count ?? 0).toLocaleString()} triangles · geometry fallback Cd ${stlResult.cd.toFixed(4)}` : undefined}
          onModeChange={(nextMode) => {
            if (nextMode === "parameters") viewerRef.current?.showReference();
            setMode(nextMode);
          }}
          onParameterChange={handleParameterChange}
          onCategoryChange={handleCategoryChange}
          onPreset={handlePreset}
          onFocus={focusParameter}
          onToggleLock={(name) => toggleLock(name as NumericParameterName)}
          onStlUpload={(file) => void handleStlUpload(file)}
        />

        <section className="viewport-panel" aria-label="3D design preview">
          <div className="viewport-toolbar">
            <div className="view-tabs" aria-label="View controls">
              {(["perspective", "side", "front", "top"] as CameraView[]).map((view) => (
                <button key={view} type="button" className={`view-button ${cameraView === view ? "active" : ""}`} onClick={() => { setCameraView(view); viewerRef.current?.setView(view); }}>{view[0].toUpperCase() + view.slice(1)}</button>
              ))}
            </div>
            <div className="viewport-meta">
              <button className={`dimension-toggle ${dimensionsVisible && mode === "parameters" ? "active" : ""}`} type="button" aria-pressed={dimensionsVisible && mode === "parameters"} disabled={mode === "stl"} onClick={() => setDimensionsVisible((visible) => !visible)}>Dimensions</button>
              <span>{mode === "stl" && cloudResult ? "Imported point cloud" : mode === "stl" && stlResult ? "Imported STL preview" : "Reference DrivAer geometry"}</span>
              <span className="live-dot" />Live
            </div>
          </div>
          <div className="canvas-wrap">
            <ViewerErrorBoundary>
              <Suspense fallback={<div className="viewer-host"><div className="loading-overlay"><span />Loading 3D workspace</div></div>}>
                <VehicleViewer
                  ref={viewerRef}
                  values={current}
                  parameters={schema.parameters}
                  wheelTreatment={current.Wheels}
                  activeParameter={activeParameter}
                  dimensionsVisible={dimensionsVisible && mode === "parameters"}
                  importedPoints={importedPoints}
                />
              </Suspense>
            </ViewerErrorBoundary>
            <div className="canvas-overlay top-left">
              <span>{mode === "stl" && cloudResult ? cloudResult.file.name.toUpperCase() : mode === "stl" && stlResult ? stlResult.file.name.toUpperCase() : `DRIVAER FASTBACK · ${current.Wheels.toUpperCase()}`}</span>
              <small>Drag to rotate · scroll to zoom</small>
              {mode === "parameters" && current.CarRear !== "Fastback" && <small className="architecture-warning">{current.CarRear} affects Cd only; this viewport keeps the Fastback reference body.</small>}
              {mode === "stl" && cloudResult && <small className="architecture-warning">PointNet prediction from your uploaded point cloud (2,048 sampled points).</small>}
              {mode === "stl" && stlResult && <small className="architecture-warning">Imported point preview · geometry fallback, not a trained CFD surrogate.</small>}
            </div>
            <div className="axis-widget" aria-hidden="true"><span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z</span></div>
            {(predictionBusy || stlBusy) && <div className="loading-overlay"><span />{stlBusy ? "Analyzing upload" : "Updating prediction"}</div>}
          </div>
          {mode === "parameters" ? <div className="design-strip">
            <div><span>Length study</span><strong>Δ {signed(lengthDelta)} mm</strong></div>
            <div><span>Width study</span><strong>Δ {signed(widthDelta)} mm</strong></div>
            <div><span>Roof study</span><strong>Δ {signed(heightDelta)} mm</strong></div>
            <div><span>Diffuser</span><strong>{current.B_Diffusor_Angle.toFixed(2)}°</strong></div>
            <div className="concept-disclaimer"><span>Geometry status</span><strong>{current.Wheels === "Open detailed" ? "QEM body · dataset wheels" : `QEM body · ${current.Wheels} procedural wheels`}</strong></div>
          </div> : cloudResult ? <div className="design-strip stl-design-strip">
            <div><span>File</span><strong>{cloudResult.file.name}</strong></div>
            <div><span>Points read</span><strong>{cloudResult.n_points_model.toLocaleString()} / {cloudResult.n_points_input.toLocaleString()}</strong></div>
            <div><span>Estimate</span><strong>{cloudResult.cd !== null ? `Cd ${cloudResult.cd.toFixed(4)}` : "Out of distribution"}</strong></div>
            <div className="concept-disclaimer"><span>Model</span><strong>PointNet · trained surrogate</strong></div>
          </div> : <div className="design-strip stl-design-strip">
            <div><span>File</span><strong>{stlResult?.file.name ?? "Choose an STL"}</strong></div>
            <div><span>Triangles</span><strong>{Number(stlResult?.mesh.triangle_count ?? 0).toLocaleString()}</strong></div>
            <div><span>Preview points</span><strong>{stlResult?.preview_points.length.toLocaleString() ?? "—"}</strong></div>
            <div><span>Estimate</span><strong>{stlResult ? `Cd ${stlResult.cd.toFixed(4)}` : "—"}</strong></div>
            <div className="concept-disclaimer"><span>Geometry status</span><strong>Imported STL · untrained fallback</strong></div>
          </div>}
        </section>

        <ResultsPanel
          design={current}
          schema={schema}
          status={status}
          prediction={displayPrediction}
          analysis={displayAnalysis}
          baselineCd={mode === "parameters" ? baselineCd : null}
          locks={locks}
          variants={variants}
          predictionMode={mode}
          resultBusy={resultBusy}
          onFocusParameter={focusParameter}
          onSetBaseline={() => {
            if (!currentPrediction || resultBusy) return;
            commitDraft();
            setBaseline(current, currentPrediction.cd);
            setToast("Baseline updated");
          }}
          onPreview={(recommendation) => { commitDraft(); previewRecommendation(recommendation); }}
          onApply={(recommendation: Recommendation) => { applyRecommendation(recommendation); setToast("Recommendation applied"); }}
          onCancelPreview={cancelRecommendation}
          onLoadVariant={(id) => { clearHistoryCommit(); setMode("parameters"); setStlResult(null); setCloudResult(null); loadVariant(id); }}
          onDeleteVariant={deleteVariant}
          onClearWorkspace={() => {
            clearHistoryCommit();
            draftDesignRef.current = null;
            setDraftDesign(null);
            setMode("parameters");
            setStlResult(null);
            setCloudResult(null);
            clearWorkspace(baseDefaults ?? undefined);
            setToast("Workspace cleared");
          }}
          onPrint={printReport}
        />
      </main>
      {requestError && <div className="request-error" role="alert"><span>{requestError}</span><button type="button" onClick={() => setRequestError("")}>×</button></div>}
      {activeDefinition && <span className="sr-only" aria-live="polite">Editing {activeDefinition.label}</span>}
      {toast && <div className="toast visible" role="status">{toast}</div>}
      <HoldoutBenchmark open={benchmarkOpen} onClose={() => setBenchmarkOpen(false)} />
    </>
  );
}
