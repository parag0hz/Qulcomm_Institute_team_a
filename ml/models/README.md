# models — 서빙용 모델

## pointnet_serving.onnx (1.8 MB)

포인트클라우드 → 항력계수(Cd) 예측. **전처리·후처리가 그래프에 내장**돼 있어
서빙 코드는 좌표만 넣으면 된다.

| | |
|---|---|
| 입력 | `points` — `(batch, 2048, 3)` float32, **미터 스케일 원본 좌표** |
| 출력 | `cd` — `(batch,)` float32, Cd 예측값 |
| opset | 17 (CPU만으로 동작, GPU 불필요) |

### ⚠️ 입력 규칙
- 좌표는 **미터 단위 원본 그대로** 넣는다. **센터링·정규화·unit-sphere 스케일링 금지**
  (모델 내부에서 학습 때와 동일한 상수를 적용한다. 밖에서 또 하면 예측이 망가진다).
- 좌표계는 지면 `z=0`, 좌우대칭축 `y=0`, `x`는 앞→뒤.
- 점 개수는 **FPS(farthest point sampling) 2,048점** 권장. batch/점 개수 축은 동적이라
  다른 값도 실행은 되지만, 학습 조건(2048)과 맞추는 게 정확하다.

### 사용 예
```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("pointnet_serving.onnx", providers=["CPUExecutionProvider"])
pts = fps(stl_to_pointcloud(stl_file), 2048).astype(np.float32)[None]   # (1, 2048, 3)
cd  = float(sess.run(["cd"], {"points": pts})[0][0])
```

### 성능 (정직한 수치)
| 지표 | 값 |
|---|---|
| **5-fold 교차검증 R²** | **0.865 ± 0.038** ← 대표 성능으로 인용할 것 |
| 5-fold MAE | 6.5 drag counts (1 count = 0.001 Cd) |
| 이 모델의 검증 R² | 0.892 (fold 1~4 학습, fold 5 검증) |
| 데모 5대 평균 오차 | 3.6 counts (표본 5대뿐 — 참고용) |

학습 데이터: DrivAerNet++ 중 파라미터 CSV ∩ 포인트클라우드 교집합 3,704대
(데모용 5대는 학습에서 영구 제외). 상세: [../ml/PROTOCOL_COMPARISON.md](../ml/PROTOCOL_COMPARISON.md)

### 범위 밖 (주의)
학습 데이터가 DrivAer 세단 계열이라 **SUV·트럭은 분포 밖**이다.
절대 Cd 측정값이 아니라 **상대적 설계 비교용**으로 쓰고, OOD 경고를 함께 노출할 것.

재생성: `python ml/scripts/export_pointnet_onnx.py`
