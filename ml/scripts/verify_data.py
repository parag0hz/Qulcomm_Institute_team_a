"""Phase 0 데이터 검증 — CFA_PROJECT.md §1-4 체크리스트를 실데이터로 확인한다.

실행:  python scripts/verify_data.py

.paddle_tensor 로딩 방법 (§1-5의 결론)
---------------------------------------
이 파일들은 PaddlePaddle이 **raw pickle**로 저장한 것이다.
`paddle.load()` / `pickle.load()`는 언피클링 중 임의 opcode를 실행하므로
서드파티 미러에서 받은 파일에 쓰면 안 된다 (공급망 리스크).
대신 `cd_common.safe_load_ndarray`를 쓴다:
  pickletools.genops 로 opcode 스트림만 훑고 → 가장 큰 bytes 버퍼를 찾아
  → dtype을 opcode에서 판별 → np.frombuffer 로 복원. 코드 실행 0회.
검증: 7,713개 전부 (100000, 3) float32 / 버퍼 1,200,000 B (전수 스캔 완료).

§1-4 가정 대비 실측 정정
------------------------
- "CSV 라벨 ~8,000개" → 실제 **7,713행** (파일명의 '8k'는 이름일 뿐).
- 진짜 함정은 개수가 아니라 **ID 표기 불일치**다: CSV는 패딩 없음(E_S_WWC_WM_1),
  split·파일명은 3자리 패딩(E_S_WWC_WM_001), 구 fastback 계열만 4자리 + 'DrivAer_' 접두사.
  raw 문자열 inner join은 **에러 없이 609개를 떨어뜨린다** (아래 §4에서 실측).
  반드시 cd_common.norm_id 로 정규화 후 조인할 것.
"""
from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, "/home/kwy00/qi")

import cd_common as C


def sep(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# ---------------------------------------------------------------- 1. 파일 개수
sep("1. .paddle_tensor 파일 개수")
files = C.file_index()                      # 디렉토리 스캔 (정규 키 -> 경로)
print(f"발견: {len(files)}개   (기대: {C.N_DESIGNS})")
assert len(files) == C.N_DESIGNS, "파일 개수 불일치"

# ------------------------------------------------------------ 2. 실제 shape
sep("2. 파일 하나 로드 -> 실제 shape/dtype (safe loader)")
sample_key = sorted(files)[0]
path = files[sample_key]
pts = C.safe_load_ndarray(path)
print(f"파일      : {path.split('/')[-1]}")
print(f"shape     : {pts.shape}")
print(f"dtype     : {pts.dtype}")
print(f"앞 3개 점 :\n{pts[:3]}")
print(f"축별 범위 : x[{pts[:,0].min():+.3f}, {pts[:,0].max():+.3f}] "
      f"y[{pts[:,1].min():+.3f}, {pts[:,1].max():+.3f}] "
      f"z[{pts[:,2].min():+.3f}, {pts[:,2].max():+.3f}]  (단위: m)")
assert pts.shape == (C.N_POINTS, 3), "shape 불일치"

# ------------------------------------------------- 3. ID 포맷 3원 비교 (눈으로)
sep("3. ID 표기 비교 — CSV vs 파일명 vs split")
import csv as _csv
import os

with open(C.CSV_PATH, encoding="utf-8-sig", newline="") as f:
    csv_raw = [row["Design"] for row in _csv.DictReader(f)]
file_raw = sorted(os.path.basename(p) for p in files.values())
with open(os.path.join(C.SPLIT_DIR, "train_design_ids.txt"), encoding="utf-8-sig") as f:
    split_raw = [l.strip() for l in f if l.strip()]

def pick(rows: list[str], token: str, n: int = 2) -> list[str]:
    return [r for r in rows if token in r][:n]

print(f"{'출처':<8}{'구 fastback 계열 (4자리)':<44}파라메트릭 계열 (E/F/N_S_*)")
print(f"{'CSV':<8}{str(pick(csv_raw, 'F_D_WM_WW')):<44}{pick(csv_raw, 'E_S_WWC_WM')}")
print(f"{'파일명':<7}{str(pick(file_raw, 'F_D_WM_WW')):<44}{pick(file_raw, 'E_S_WWC_WM')}")
print(f"{'split':<8}{str(pick(split_raw, 'F_D_WM_WW')):<44}{pick(split_raw, 'E_S_WWC_WM')}")

# ------------------------------------------------------- 4. inner join 실측
sep("4. inner join — raw 문자열 vs 정규화 키")
csv_set = set(csv_raw)
file_stem = {f[: -len(".paddle_tensor")] for f in file_raw}
raw_join = csv_set & file_stem
print(f"raw 문자열 join : {len(raw_join)}개 매칭  ->  {C.N_DESIGNS - len(raw_join)}개가 **조용히 증발**")

lost = sorted(file_stem - csv_set)
print(f"증발 예시        : {lost[:3]} ...")

cd = C.drag_table()                          # norm_id 로 정규화된 라벨 테이블
norm_join = set(files) & set(cd)
print(f"정규화 후 join   : {len(norm_join)}개 매칭  (기대: {C.N_DESIGNS})")
assert len(norm_join) == C.N_DESIGNS, "정규화 join 불일치"

# ------------------------------------------------------------- 5. split 검증
sep("5. split ID 포맷/멤버십")
sp = C.splits()
for s, ids in sp.items():
    missing = set(ids) - set(files)
    print(f"{s:<6}: {len(ids):>5}개  (기대 {C.SPLIT_SIZES[s]})   파일 없는 ID: {len(missing)}개")
body = Counter(C.body_type(k) for k in files)
print(f"차종 분포: {dict(body)}")

# ------------------------------------------------------------- 6. 종합 게이트
sep("6. cd_common.check_integrity() — 전체 무결성 게이트")
C.check_integrity()
print("OK — 파일 7,713 = 라벨 7,713 = split 합집합. 중복/누락/교차 없음.")
print("\nPhase 0 데이터 검증 통과 ✅")
