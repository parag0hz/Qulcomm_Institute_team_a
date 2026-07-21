"""데모 전용 홀드아웃 선정 — 학습/검증/테스트 전부에서 영구 제외 (피드백 #6).

방침:
  - CSV(설계 파라미터)와 포인트클라우드를 **둘 다 가진 교집합**에서 고른다.
    → 한 대로 슬라이더 경로와 형상 경로를 동시에 시연 가능하고,
      "CSV 5대 / 포인트클라우드 5대" 요구를 같은 5대로 충족한다.
  - 차종 3종을 모두 포함하고, 각 차종 안에서 Cd 분위수를 고르게 잡는다.
  - 난수 없이 **결정적**으로 선정 (분위수 기반) → 언제 다시 돌려도 같은 5대.

  python scripts/make_demo_holdout.py
결과: data/demo_holdout.json  (이후 모든 학습 스크립트가 이 ID를 제외해야 함)
"""
from __future__ import annotations

import csv as _csv
import json
import sys

import numpy as np

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C

CSV = "/home/kwy00/qi/data/DrivAerNet_ParametricData.csv"
FPS = "/home/kwy00/qi/data/fps2048.npz"
OUT = "/home/kwy00/qi/data/demo_holdout.json"

# 차종별 배정 (총 5대). Fastback이 최다수라 2대, 나머지 각각.
ALLOC = {"Fastback": 2, "Estate": 2, "Notchback": 1}


def main():
    C.check_integrity()
    d = np.load(FPS, allow_pickle=True)
    keys = np.array([str(k) for k in d["keys"]])
    cd, cls, split = d["cd"].astype(float), d["cls"], d["split"]

    # --- CSV에 존재하는 ID 집합 (정규화 키) ---
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rdr = _csv.DictReader(f)
        csv_keys = set()
        for r in rdr:
            p, i = C.norm_id(r["Experiment"])
            csv_keys.add(f"{p}_{i}")

    in_csv = np.array([k in csv_keys for k in keys])
    print(f"포인트클라우드 {len(keys):,}대 · CSV {len(csv_keys):,}행 → 교집합 {in_csv.sum():,}대")

    # --- 차종별 Cd 분위수로 결정적 선정 ---
    picked = []
    for body, n in ALLOC.items():
        m = in_csv & (cls == body)
        idx = np.where(m)[0]
        order = idx[np.argsort(cd[idx])]                      # Cd 오름차순
        # n개를 균등 분위수 위치에서 (n=1 -> 중앙, n=2 -> 20%/80%)
        qs = [0.5] if n == 1 else np.linspace(0.2, 0.8, n)
        for q in qs:
            pos = int(round(q * (len(order) - 1)))
            picked.append(int(order[pos]))
        print(f"  {body:<10} 교집합 {m.sum():>4}대 → {n}대 선정")

    picked = sorted(set(picked), key=lambda i: cd[i])
    assert len(picked) == sum(ALLOC.values()), "중복 선정 발생"

    files = C.file_index()
    rows = []
    for i in picked:
        p, ix = C.norm_id(keys[i])
        rows.append({
            "id": keys[i],
            "body_type": str(cls[i]),
            "true_cd": round(float(cd[i]), 5),
            "original_split": str(split[i]),
            "pointcloud_file": str(files[(p, ix)]).split("/")[-1],
            "has_csv": True,
            "has_pointcloud": True,
        })

    payload = {
        "purpose": "데모 전용. 학습/검증/테스트 어디에도 사용 금지 (피드백 #6)",
        "n": len(rows),
        "selection_rule": "CSV∩포인트클라우드 교집합에서 차종별 Cd 분위수 기반 결정적 선정",
        "alloc": ALLOC,
        "ids": [r["id"] for r in rows],
        "items": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    print(f"\n=== 데모 홀드아웃 {len(rows)}대 ===")
    print(f"{'ID':<24}{'차종':<12}{'실제 Cd':>9}{'원래 split':>12}")
    print("-" * 60)
    for r in rows:
        print(f"{r['id']:<24}{r['body_type']:<12}{r['true_cd']:>9.4f}{r['original_split']:>12}")
    print(f"\n저장: {OUT}")
    print(f"이후 학습 데이터: 교집합 {in_csv.sum():,} → {in_csv.sum()-len(rows):,}대 (데모 {len(rows)}대 제외)")


if __name__ == "__main__":
    main()
