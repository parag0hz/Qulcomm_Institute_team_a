import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { PointNetDemoResponse } from "../types";

interface HoldoutBenchmarkProps {
  open: boolean;
  onClose: () => void;
}

const formatCd = (value: number | null) => (value == null ? "—" : value.toFixed(4));

/** 오차 크기를 색으로 구분한다. 5 counts는 문헌이 대체모델에 허용하는 경계다. */
function errorTone(counts: number | undefined): string {
  if (counts == null) return "";
  if (counts < 5) return "good";
  if (counts < 10) return "fair";
  return "poor";
}

export function HoldoutBenchmark({ open, onClose }: HoldoutBenchmarkProps) {
  const [result, setResult] = useState<PointNetDemoResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);

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

  useEffect(() => {
    if (!open) return;
    void run();
    closeRef.current?.focus();
    return () => abortRef.current?.abort();
  }, [open, run]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const items = result?.items ?? [];

  return (
    <div className="benchmark-backdrop" role="presentation" onClick={onClose}>
      <section
        className="benchmark-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="benchmark-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="benchmark-head">
          <div>
            <p className="eyebrow">Point-cloud model</p>
            <h2 id="benchmark-title">Held-out benchmark</h2>
          </div>
          <button
            ref={closeRef}
            className="icon-button"
            type="button"
            aria-label="Close benchmark"
            onClick={onClose}
          >
            ✕
          </button>
        </header>

        <p className="benchmark-lede">
          These designs were <strong>permanently excluded</strong> from training and validation.
          Every run below calls the model live — nothing here is precomputed.
        </p>

        {result?.available === false && (
          <div className="risk-banner outside">{result.reason}</div>
        )}
        {error && <div className="risk-banner outside">{error}</div>}

        {result?.available && (
          <>
            <div className="benchmark-headline">
              <div>
                <span>Mean absolute error</span>
                <strong>
                  {result.mean_error_counts == null ? "—" : result.mean_error_counts.toFixed(2)}
                  <small> drag counts</small>
                </strong>
              </div>
              <p>
                1 count = 0.001 Cd. Literature treats under 5 counts as acceptable for a
                surrogate screening model.
              </p>
            </div>

            <div className="benchmark-table-scroll">
              <table className="benchmark-table">
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
                      <td className="mono">{item.id}</td>
                      <td>{item.body_type}</td>
                      <td className="mono">{formatCd(item.true_cd)}</td>
                      <td className="mono">
                        {item.trusted ? (
                          formatCd(item.cd)
                        ) : (
                          <span title={item.warnings[0]}>out of distribution</span>
                        )}
                      </td>
                      <td className={`mono error-cell ${errorTone(item.error_counts)}`}>
                        {item.error_counts == null ? "—" : `${item.error_counts.toFixed(2)}`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="benchmark-foot">
              {result.point_count} points per design · all {items.length} held-out cars shown,
              including the worst case.
            </p>
          </>
        )}

        <div className="benchmark-actions">
          <button className="dark-button" type="button" disabled={busy} onClick={() => void run()}>
            {busy ? "Predicting…" : "Run prediction again"}
          </button>
        </div>
      </section>
    </div>
  );
}
