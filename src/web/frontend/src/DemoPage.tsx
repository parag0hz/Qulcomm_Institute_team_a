import { useCallback, useEffect, useRef, useState } from "react";
import "./demo.css";
import { api } from "./api";
import { BodyTypeChart } from "./components/BodyTypeChart";
import { PointCloudViewer } from "./components/PointCloudViewer";
import type { DemoCar, DemoInference } from "./types";

const STUDIO_URL = "/";

/**
 * 데모의 본체.
 *
 * 흐름을 일부러 한 화면에 묶었다: 모델이 보는 것(점군)을 먼저 띄우고 →
 * 버튼을 누르면 → 실제로 추론이 돌고 → 정답과 나란히 놓는다. 이 순서를
 * 눈으로 따라가면 "이게 뭐 하는 물건인지"가 설명 없이 전달된다.
 */
function LiveDemo() {
  const [cars, setCars] = useState<DemoCar[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [points, setPoints] = useState<number[][] | null>(null);
  const [loadingCloud, setLoadingCloud] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [result, setResult] = useState<DemoInference | null>(null);
  const [level, setLevel] = useState(2048);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const message = (cause: unknown) =>
    cause instanceof Error && cause.message ? cause.message : "Could not reach the server.";

  // 목록을 받아 첫 차량을 자동 선택한다.
  useEffect(() => {
    const controller = new AbortController();
    api
      .getDemoCars(controller.signal)
      .then((payload) => {
        if (controller.signal.aborted) return;
        setCars(payload.cars);
        if (payload.cars.length) setActiveId(payload.cars[0].id);
      })
      .catch((cause) => {
        if (!controller.signal.aborted) setError(message(cause));
      });
    return () => controller.abort();
  }, []);

  // 선택이 바뀌면 점군을 새로 불러온다.
  useEffect(() => {
    if (!activeId) return;
    const controller = new AbortController();
    setLoadingCloud(true);
    setResult(null);
    setError(null);
    api
      .getDemoCloud(activeId, controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) setPoints(payload.points);
      })
      .catch((cause) => {
        if (!controller.signal.aborted) setError(message(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingCloud(false);
      });
    return () => controller.abort();
  }, [activeId]);

  const infer = useCallback(async () => {
    if (!activeId) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setInferring(true);
    setError(null);
    setResult(null);
    try {
      const payload = await api.inferDemoCar(activeId, level, controller.signal);
      if (!controller.signal.aborted) setResult(payload);
    } catch (cause) {
      if (!controller.signal.aborted) setError(message(cause));
    } finally {
      if (!controller.signal.aborted) setInferring(false);
    }
  }, [activeId, level]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const active = cars.find((car) => car.id === activeId) ?? null;
  const available = active?.point_count ?? 2048;
  // FPS 순서가 보존돼 있어 앞에서 K개를 자르면 그대로 FPS-K 샘플이 된다.
  const levels = [1, 128, 512, 1024, 2048, 4096].filter((n) => n <= available);
  const shown = points ? points.slice(0, Math.min(level, points.length)) : null;

  return (
    <div className="demo-stage">
      <div className="stage-viewer">
        <PointCloudViewer points={shown} busy={loadingCloud || inferring} />
        {inferring && (
          <div className="stage-overlay" role="status">
            <span className="spinner" aria-hidden="true" />
            Running inference…
          </div>
        )}
      </div>

      <aside className="stage-side">
        <div className="stage-block">
          <p className="stage-label">Never seen during training</p>
          <div className="car-chips">
            {cars.map((car) => (
              <button
                key={car.id}
                type="button"
                className={`chip ${car.id === activeId ? "active" : ""}`}
                onClick={() => setActiveId(car.id)}
              >
                {car.body_type}
                <small>{car.id.split("_").pop()}</small>
              </button>
            ))}
            {!cars.length && !error && <span className="stage-muted">Loading…</span>}
          </div>
        </div>

        <div className="stage-block">
          <p className="stage-label">
            Points fed to the model
            <span className="level-value">{level.toLocaleString()}</span>
          </p>
          <div className="level-steps">
            {levels.map((n) => (
              <button
                key={n}
                type="button"
                className={`step ${n === level ? "active" : ""}`}
                onClick={() => {
                  setLevel(n);
                  setResult(null);
                }}
              >
                {n.toLocaleString()}
                {n === 2048 && <em>trained</em>}
              </button>
            ))}
          </div>
        </div>

        <button
          className="pill primary block"
          type="button"
          disabled={!activeId || inferring || loadingCloud}
          onClick={() => void infer()}
        >
          {inferring ? "Running inference…" : "Run inference on this shape"}
        </button>

        {error && <div className="stage-error">{error}</div>}

        {result ? (
          <div className="stage-result">
            <div className="result-row">
              <span>Model prediction</span>
              <strong>{result.trusted ? result.cd?.toFixed(4) : "Out of distribution"}</strong>
            </div>
            <div className="result-row">
              <span>Actual (CFD)</span>
              <strong className="muted">{result.true_cd?.toFixed(4)}</strong>
            </div>
            <div className="result-row highlight">
              <span>Error</span>
              <strong className={(result.error_counts ?? 0) < 5 ? "good" : "warn"}>
                {result.error_counts?.toFixed(2)} counts
              </strong>
            </div>
            <p className="stage-foot">
              {result.n_points.toLocaleString()} points · {result.inference_ms.toFixed(1)} ms
              {result.n_points !== result.trained_points && (
                <> · the model was trained on {result.trained_points.toLocaleString()} points, so
                this is a stress test</>
              )}
              . 1 count = 0.001 Cd; under 5 counts is the accuracy a surrogate is expected to reach.
            </p>
          </div>
        ) : (
          <p className="stage-foot">
            {active
              ? `${active.body_type} body. Step the point count up and watch the shape — and the prediction — resolve.`
              : "Pick a vehicle to see the point cloud the model reads."}
          </p>
        )}
      </aside>
    </div>
  );
}

export function DemoPage() {
  return (
    <div className="demo-root">
      <nav className="demo-nav">
        <a className="brand" href={STUDIO_URL}>
          <span className="brand-mark">P</span>
          <span>Paragon</span>
        </a>
        <a className="pill dark sm" href={STUDIO_URL}>
          Open the studio
        </a>
      </nav>

      {/* 데모를 맨 위에. 설명보다 먼저 보여준다. */}
      <header className="demo-section demo-hero">
        <div className="demo-wrap">
          <p className="demo-eyebrow">Drag prediction from 3D shape</p>
          <h1 className="demo-h1">It reads drag from nothing but these points.</h1>
          <p className="demo-lede">
            A CFD run takes days to weeks per design. Paragon estimates the same coefficient from
            2,048 points sampled off the body surface, in milliseconds.
          </p>
          <LiveDemo />
        </div>
      </header>

      {/* 방금 무슨 일이 있었는지 */}
      <section className="demo-section tinted">
        <div className="demo-wrap">
          <p className="demo-eyebrow">What just happened</p>
          <h2 className="demo-h2">Nothing above was precomputed.</h2>
          <div className="demo-grid cols-3">
            <div className="demo-card step-card">
              <span className="step-no">1</span>
              <h3>A car it has never seen</h3>
              <p>
                These five were held out of training and validation from the very start. The model met
                them for the first time just now.
              </p>
            </div>
            <div className="demo-card step-card">
              <span className="step-no">2</span>
              <h3>2,048 points, nothing else</h3>
              <p>
                The coordinates on screen go straight into the network. No images, no drawings, no design
                parameters.
              </p>
            </div>
            <div className="demo-card step-card">
              <span className="step-no">3</span>
              <h3>Checked against the truth</h3>
              <p>
                The reference value comes from a full CFD simulation. We show the gap as it is.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 왜 형상인가 */}
      <section className="demo-section">
        <div className="demo-wrap">
          <p className="demo-eyebrow">Why shape</p>
          <h2 className="demo-h2">Design numbers break down on wagons.</h2>
          <p className="demo-lede">
            Length, width and angles get you some of the way. But the moment the body style changes it
            wobbles — and on estate bodies it drops below simply guessing the average.
          </p>
          <BodyTypeChart />
        </div>
      </section>

      {/* 한계 */}
      <section className="demo-section tinted">
        <div className="demo-wrap">
          <p className="demo-eyebrow">Limits</p>
          <h2 className="demo-h2">What this tool will not answer.</h2>
          <div className="demo-grid cols-3">
            <div className="demo-card">
              <h3>Sedan derivatives only</h3>
              <p>
                It learned from DrivAer sedan variants. SUVs and trucks fall outside that, so it reports
                out-of-distribution instead of answering.
              </p>
            </div>
            <div className="demo-card">
              <h3>Very small differences</h3>
              <p>
                Below 5 counts even the direction of a change is close to a coin flip. Trust the ranking,
                not the last digit.
              </p>
            </div>
            <div className="demo-card">
              <h3>Not a certification</h3>
              <p>
                This screens candidates before CFD. It does not replace a wind tunnel or a high-fidelity
                solve.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="demo-section center">
        <div className="demo-wrap">
          <h2 className="demo-h2">Now change the shape yourself.</h2>
          <p className="demo-lede">
            In the studio, every one of the 23 parameters updates the 3D body and the predicted Cd as you
            move it.
          </p>
          <div className="demo-cta-row">
            <a className="pill primary" href={STUDIO_URL}>
              Open the studio
            </a>
          </div>
        </div>
      </section>

      <footer className="demo-footer">
        <div className="demo-wrap">
          Chosun University · Qualcomm Institute Team A — trained on{" "}
          <a href="https://github.com/Mohamedelrefaie/DrivAerNet" target="_blank" rel="noreferrer">
            DrivAerNet++
          </a>{" "}
          (Elrefaie et al., NeurIPS 2024).
        </div>
      </footer>
    </div>
  );
}
