import { useCallback, useEffect, useRef, useState } from "react";
import "./demo.css";
import { api } from "./api";
import { BodyTypeChart } from "./components/BodyTypeChart";
import type { PointNetDemoResponse } from "./types";

const STUDIO_URL = "/";

function errorTone(counts: number | undefined) {
  if (counts == null) return "";
  return counts < 5 ? "good" : counts < 10 ? "" : "poor";
}

/** 라이브 증명 — 이 페이지의 중심. 버튼을 누르면 실제로 모델이 돈다. */
function LiveProof() {
  const [result, setResult] = useState<PointNetDemoResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.getPointNetDemo(controller.signal));
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : "Prediction failed.");
      }
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const items = result?.items ?? [];

  return (
    <div className="live-panel">
      <div className="live-head">
        <div>
          <h3>Five cars the model has never seen</h3>
          <p>Held out of training and validation from the start. Nothing here is precomputed.</p>
        </div>
        <button className="pill primary" type="button" disabled={busy} onClick={() => void run()}>
          {busy ? "Predicting…" : result ? "Run again" : "Run the model"}
        </button>
      </div>

      {error && <div className="live-error">{error}</div>}
      {result?.available === false && <div className="live-error">{result.reason}</div>}

      {result?.available && (
        <>
          <div className="live-headline">
            <strong>
              {result.mean_error_counts == null ? "—" : result.mean_error_counts.toFixed(2)}
            </strong>
            <span>drag counts of average error · 1 count = 0.001 Cd</span>
          </div>

          <table className="live-table">
            <thead>
              <tr>
                <th scope="col">Design</th>
                <th scope="col">Body</th>
                <th scope="col">Actual Cd</th>
                <th scope="col">Predicted</th>
                <th scope="col">Error</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id}>
                  <td className="id">{item.id}</td>
                  <td className="id">{item.body_type}</td>
                  <td>{item.true_cd?.toFixed(4) ?? "—"}</td>
                  <td>
                    {item.trusted ? (
                      item.cd?.toFixed(4)
                    ) : (
                      <span title={item.warnings[0]}>out of distribution</span>
                    )}
                  </td>
                  <td className={`err ${errorTone(item.error_counts)}`}>
                    {item.error_counts?.toFixed(2) ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="live-foot">
            {result.point_count} points per car · all {items.length} shown, including the worst case.
            Under 5 counts is the accuracy literature accepts for screening.
          </p>
        </>
      )}
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

      {/* 1 — 무엇을 하는 물건인가 */}
      <header className="demo-section demo-hero center">
        <div className="demo-wrap">
          <p className="demo-eyebrow">Aerodynamic drag, predicted</p>
          <h1 className="demo-h1">
            Weeks of simulation,
            <br />
            answered in milliseconds.
          </h1>
          <p className="demo-lede">
            Paragon estimates a vehicle's drag coefficient the moment you change its shape —
            so you can compare fifty concepts before CFD has finished one.
          </p>
          <div className="demo-cta-row">
            <a className="pill primary" href="#live">
              See it predict live
            </a>
            <a className="pill ghost" href={STUDIO_URL}>
              Open the studio
            </a>
          </div>
          <div className="hero-readout">
            <small>Inference</small>
            <strong>1.8</strong>
            <em>milliseconds per design, on a CPU</em>
          </div>
        </div>
      </header>

      {/* 2 — 왜 필요한가 */}
      <section className="demo-section tinted">
        <div className="demo-wrap">
          <p className="demo-eyebrow">The bottleneck</p>
          <h2 className="demo-h2">Testing a shape costs more than drawing it.</h2>
          <p className="demo-lede">
            Computational fluid dynamics is the standard way to measure drag, and it is slow
            enough that early-stage designers simply never run it.
          </p>
          <div className="demo-grid cols-3">
            <div className="demo-card stat-card">
              <strong>Days–weeks</strong>
              <span>for one high-fidelity CFD run</span>
            </div>
            <div className="demo-card stat-card">
              <strong>2,880</strong>
              <span>CPU cores used to build the dataset we learn from</span>
            </div>
            <div className="demo-card stat-card">
              <strong>39 TB</strong>
              <span>of simulation output behind those labels</span>
            </div>
          </div>
        </div>
      </section>

      {/* 3 — 어떻게 동작하는가 */}
      <section className="demo-section">
        <div className="demo-wrap">
          <p className="demo-eyebrow">How it works</p>
          <h2 className="demo-h2">Two ways in, one number out.</h2>
          <p className="demo-lede">
            Give it design parameters or the 3D shape itself. Both paths end at the same
            answer: the drag coefficient, immediately.
          </p>
          <div className="demo-grid cols-3">
            <div className="demo-card step-card">
              <span className="step-no">1</span>
              <h3>Describe the car</h3>
              <p>
                Move 23 design sliders, or hand it a point cloud sampled from the body surface.
              </p>
            </div>
            <div className="demo-card step-card">
              <span className="step-no">2</span>
              <h3>The model reads the shape</h3>
              <p>
                A 0.8M-parameter network trained on 3,704 simulated vehicles. Small enough to
                run on a CPU, so there is no queue.
              </p>
            </div>
            <div className="demo-card step-card">
              <span className="step-no">3</span>
              <h3>Compare, don't guess</h3>
              <p>
                Rank concepts against each other and against the dataset, with the uncertainty
                stated rather than hidden.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4 — 증명 (이 페이지의 중심) */}
      <section className="demo-section tinted center" id="live">
        <div className="demo-wrap">
          <p className="demo-eyebrow">The proof</p>
          <h2 className="demo-h2">Don't take our word for it.</h2>
          <p className="demo-lede">
            Five vehicles were removed from the dataset before training ever started. Their true
            drag was computed by CFD. Press the button and watch the model meet them for the
            first time.
          </p>
          <LiveProof />
        </div>
      </section>

      {/* 5 — 왜 형상인가 */}
      <section className="demo-section">
        <div className="demo-wrap">
          <p className="demo-eyebrow">Why shape</p>
          <h2 className="demo-h2">Numbers describe a car. Shape explains it.</h2>
          <p className="demo-lede">
            Predicting from design parameters alone works until the body style changes. On
            estate bodies it collapses — one model does worse than simply guessing the average.
          </p>
          <BodyTypeChart />
        </div>
      </section>

      {/* 6 — 정직하게 */}
      <section className="demo-section tinted">
        <div className="demo-wrap">
          <p className="demo-eyebrow">Where it stops</p>
          <h2 className="demo-h2">What this tool will not tell you.</h2>
          <div className="demo-grid cols-3">
            <div className="demo-card">
              <h3>Sedans only</h3>
              <p>
                It learned from DrivAer sedan variants. SUVs and trucks are outside what it has
                seen, and it says so instead of answering confidently.
              </p>
            </div>
            <div className="demo-card">
              <h3>Small differences</h3>
              <p>
                Below about 5 drag counts, the sign of a predicted change is close to a coin
                flip. Trust the ranking, not the last decimal.
              </p>
            </div>
            <div className="demo-card">
              <h3>Not a certificate</h3>
              <p>
                This is a screening filter that runs before CFD, never a replacement for a wind
                tunnel.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 7 — 직접 해보기 */}
      <section className="demo-section center">
        <div className="demo-wrap">
          <h2 className="demo-h2">Now move a slider and watch it change.</h2>
          <p className="demo-lede">
            The studio gives you the 3D model, all 23 parameters, and a live prediction on every
            edit.
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
