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
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const message = (cause: unknown) =>
    cause instanceof Error && cause.message ? cause.message : "서버에 연결하지 못했습니다.";

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
      const payload = await api.inferDemoCar(activeId, controller.signal);
      if (!controller.signal.aborted) setResult(payload);
    } catch (cause) {
      if (!controller.signal.aborted) setError(message(cause));
    } finally {
      if (!controller.signal.aborted) setInferring(false);
    }
  }, [activeId]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const active = cars.find((car) => car.id === activeId) ?? null;

  return (
    <div className="demo-stage">
      <div className="stage-viewer">
        <PointCloudViewer points={points} busy={loadingCloud || inferring} />
        {inferring && (
          <div className="stage-overlay" role="status">
            <span className="spinner" aria-hidden="true" />
            추론 중입니다…
          </div>
        )}
      </div>

      <aside className="stage-side">
        <div className="stage-block">
          <p className="stage-label">학습에 쓰이지 않은 차량</p>
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
            {!cars.length && !error && <span className="stage-muted">불러오는 중…</span>}
          </div>
        </div>

        <button
          className="pill primary block"
          type="button"
          disabled={!activeId || inferring || loadingCloud}
          onClick={() => void infer()}
        >
          {inferring ? "추론 중입니다…" : "이 형상으로 추론하기"}
        </button>

        {error && <div className="stage-error">{error}</div>}

        {result ? (
          <div className="stage-result">
            <div className="result-row">
              <span>모델 예측</span>
              <strong>{result.trusted ? result.cd?.toFixed(4) : "분포 밖"}</strong>
            </div>
            <div className="result-row">
              <span>실제 CFD 값</span>
              <strong className="muted">{result.true_cd?.toFixed(4)}</strong>
            </div>
            <div className="result-row highlight">
              <span>오차</span>
              <strong className={(result.error_counts ?? 0) < 5 ? "good" : "warn"}>
                {result.error_counts?.toFixed(2)} counts
              </strong>
            </div>
            <p className="stage-foot">
              추론 {result.inference_ms.toFixed(1)}ms · 1 count = 0.001 Cd · 5 counts 이하면
              대체모델로 통용되는 정확도입니다.
            </p>
          </div>
        ) : (
          <p className="stage-foot">
            {active
              ? `${active.body_type} 차체 · ${active.point_count.toLocaleString()}개 점. 이 좌표만이 모델의 입력입니다.`
              : "차량을 고르면 모델이 보는 점군이 그대로 표시됩니다."}
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
          스튜디오 열기
        </a>
      </nav>

      {/* 데모를 맨 위에. 설명보다 먼저 보여준다. */}
      <header className="demo-section demo-hero">
        <div className="demo-wrap">
          <p className="demo-eyebrow">3D 형상으로 공기저항 예측</p>
          <h1 className="demo-h1">이 점들만 보고, 항력을 맞힙니다.</h1>
          <p className="demo-lede">
            CFD 시뮬레이션은 설계 하나에 며칠에서 몇 주가 걸립니다. Paragon은 차체 표면에서 뽑은
            2,048개의 점만으로 같은 값을 밀리초 만에 추정합니다.
          </p>
          <LiveDemo />
        </div>
      </header>

      {/* 방금 무슨 일이 있었는지 */}
      <section className="demo-section tinted">
        <div className="demo-wrap">
          <p className="demo-eyebrow">방금 일어난 일</p>
          <h2 className="demo-h2">사전에 계산해둔 값이 아닙니다.</h2>
          <div className="demo-grid cols-3">
            <div className="demo-card step-card">
              <span className="step-no">1</span>
              <h3>처음 보는 차량</h3>
              <p>
                이 5대는 학습과 검증 어디에도 쓰이지 않도록 처음부터 빼두었습니다. 모델에게는
                방금이 첫 대면입니다.
              </p>
            </div>
            <div className="demo-card step-card">
              <span className="step-no">2</span>
              <h3>점 2,048개가 입력의 전부</h3>
              <p>
                화면에 보이는 좌표를 그대로 신경망에 넣습니다. 사진도, 도면도, 설계 수치도 쓰지
                않습니다.
              </p>
            </div>
            <div className="demo-card step-card">
              <span className="step-no">3</span>
              <h3>정답과 나란히</h3>
              <p>
                비교 대상인 실제 Cd는 슈퍼컴퓨터로 돌린 CFD 결과입니다. 그 차이를 그대로
                보여줍니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 왜 형상인가 */}
      <section className="demo-section">
        <div className="demo-wrap">
          <p className="demo-eyebrow">왜 형상이어야 하는가</p>
          <h2 className="demo-h2">설계 수치만으로는 왜건에서 무너집니다.</h2>
          <p className="demo-lede">
            길이·폭·각도 같은 파라미터로도 예측은 됩니다. 다만 차체 형식이 바뀌면 흔들리고,
            에스테이트에서는 평균을 찍는 것보다 못한 지점까지 내려갑니다.
          </p>
          <BodyTypeChart />
        </div>
      </section>

      {/* 한계 */}
      <section className="demo-section tinted">
        <div className="demo-wrap">
          <p className="demo-eyebrow">한계</p>
          <h2 className="demo-h2">이 도구가 답하지 않는 것.</h2>
          <div className="demo-grid cols-3">
            <div className="demo-card">
              <h3>세단 계열만</h3>
              <p>
                DrivAer 세단 변형으로 학습했습니다. SUV·트럭은 학습 분포 밖이라, 값을 내는 대신
                분포 밖이라고 알립니다.
              </p>
            </div>
            <div className="demo-card">
              <h3>미세한 차이</h3>
              <p>
                5 counts 미만의 변화는 부호조차 동전 던지기에 가깝습니다. 순위를 믿되 마지막
                자리는 믿지 마세요.
              </p>
            </div>
            <div className="demo-card">
              <h3>인증이 아닙니다</h3>
              <p>
                CFD 앞단에서 후보를 걸러내는 도구입니다. 풍동과 고충실도 해석을 대체하지
                않습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="demo-section center">
        <div className="demo-wrap">
          <h2 className="demo-h2">이제 직접 형상을 바꿔보세요.</h2>
          <p className="demo-lede">
            스튜디오에서는 23개 파라미터를 움직일 때마다 3D 형상과 예측 Cd가 함께 갱신됩니다.
          </p>
          <div className="demo-cta-row">
            <a className="pill primary" href={STUDIO_URL}>
              스튜디오 열기
            </a>
          </div>
        </div>
      </section>

      <footer className="demo-footer">
        <div className="demo-wrap">
          조선대학교 · Qualcomm Institute Team A —{" "}
          <a href="https://github.com/Mohamedelrefaie/DrivAerNet" target="_blank" rel="noreferrer">
            DrivAerNet++
          </a>{" "}
          (Elrefaie et al., NeurIPS 2024) 로 학습했습니다.
        </div>
      </footer>
    </div>
  );
}
