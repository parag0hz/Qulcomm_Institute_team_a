/**
 * 차종별 예측 정확도(R²) — 형상 기반 vs 설계 파라미터 기반.
 *
 * 출처: ml/PROTOCOL_COMPARISON.md — 같은 3,704대, 같은 K=5 fold.
 * 이야기의 핵심은 Estate다: 파라미터 모델은 0 아래로 내려가 '평균 찍기보다
 * 못한' 영역에 들어가는데, 형상 모델은 세 차종에서 거의 평평하다.
 * 그래서 0선을 반드시 그리고, 음수 막대가 그 아래로 내려가게 둔다.
 */

interface Row {
  body: string;
  shape: number;
  params: number;
}

const ROWS: Row[] = [
  { body: "Fastback", shape: 0.783, params: 0.449 },
  { body: "Estate", shape: 0.786, params: -0.257 },
  { body: "Notchback", shape: 0.788, params: 0.606 },
];

const SHAPE = "#0f9a8d";
const PARAMS = "#b45309";

// 좌표계: viewBox 안에서만 계산하고 CSS로 늘린다.
const W = 720;
const H = 300;
const PAD = { top: 16, right: 16, bottom: 46, left: 44 };
const Y_MAX = 0.9;
const Y_MIN = -0.35;

const plotH = H - PAD.top - PAD.bottom;
const plotW = W - PAD.left - PAD.right;
const y = (v: number) => PAD.top + ((Y_MAX - v) / (Y_MAX - Y_MIN)) * plotH;
const zeroY = y(0);

const groupW = plotW / ROWS.length;
const BAR = 54;
const GAP = 10; // 인접 막대 사이 표면 간격

export function BodyTypeChart() {
  return (
    <div className="chart-block">
      <div className="chart-legend">
        <span>
          <i style={{ background: SHAPE }} aria-hidden="true" />
          Shape (point cloud)
        </span>
        <span>
          <i style={{ background: PARAMS }} aria-hidden="true" />
          Design parameters
        </span>
      </div>

      <svg
        className="chart-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Prediction accuracy by body type. Shape-based scores 0.78 on all three body types. Parameter-based scores 0.45 on fastback, 0.61 on notchback, and minus 0.26 on estate, which is worse than predicting the average."
      >
        {/* 눈금 — 뒤로 물러나 있어야 한다 */}
        {[0.8, 0.4, 0.0, -0.2].map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke={tick === 0 ? "#c7c7cc" : "#ececf0"}
              strokeWidth={tick === 0 ? 1.5 : 1}
            />
            <text
              x={PAD.left - 10}
              y={y(tick) + 4}
              textAnchor="end"
              fontSize="12"
              fill="#8a8a8e"
            >
              {tick.toFixed(1)}
            </text>
          </g>
        ))}

        {ROWS.map((row, index) => {
          const cx = PAD.left + groupW * index + groupW / 2;
          const bars = [
            { key: "shape", value: row.shape, color: SHAPE, x: cx - BAR - GAP / 2 },
            { key: "params", value: row.params, color: PARAMS, x: cx + GAP / 2 },
          ];
          return (
            <g key={row.body}>
              {bars.map((bar) => {
                const top = bar.value >= 0 ? y(bar.value) : zeroY;
                const height = Math.abs(y(bar.value) - zeroY);
                const negative = bar.value < 0;
                return (
                  <g key={bar.key}>
                    <rect
                      x={bar.x}
                      y={top}
                      width={BAR}
                      height={height}
                      rx="4"
                      fill={bar.color}
                    />
                    {/* 값 직접 표기 — 범례에만 의존하지 않게 한다 */}
                    <text
                      x={bar.x + BAR / 2}
                      y={negative ? top + height + 17 : top - 8}
                      textAnchor="middle"
                      fontSize="13"
                      fontWeight="600"
                      fill={negative ? "#c0392f" : "#1d1d1f"}
                    >
                      {bar.value > 0 ? "+" : ""}
                      {bar.value.toFixed(2)}
                    </text>
                  </g>
                );
              })}
              <text
                x={cx}
                y={H - PAD.bottom + 26}
                textAnchor="middle"
                fontSize="14"
                fill="#1d1d1f"
              >
                {row.body}
              </text>
            </g>
          );
        })}

        <text x={PAD.left - 10} y={PAD.top - 2} textAnchor="end" fontSize="11" fill="#8a8a8e">
          R²
        </text>
      </svg>

      <p className="chart-caption">
        Coefficient of determination on held-out folds — higher is better, and anything below
        zero is worse than predicting the dataset average. Same 3,704 vehicles and same 5-fold
        split for both methods.
      </p>
    </div>
  );
}
