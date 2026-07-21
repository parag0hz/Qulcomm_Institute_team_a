"""outputs/*.json -> R0 대비 비교표."""
import glob, json

R0 = {"All": 0.814, "Fastback": 0.881, "Estate": -0.519, "Notchback": -0.151}
R0_MAE = {"All": 4.6, "Fastback": 3.6, "Estate": 6.6, "Notchback": 6.7}
ORDER = ["All", "Fastback", "Estate", "Notchback"]

rows = []
for f in sorted(glob.glob("/home/kwy00/qi/outputs/*.json")):
    r = json.load(open(f))
    rows.append(r)
rows.sort(key=lambda r: -r["test"]["All"][0])

print(f"{'run':<32}{'params':>8}" + "".join(f"{c:>20}" for c in ORDER))
print(f"{'':<32}{'':>8}" + "".join(f"{'R2 (MAE%)':>20}" for _ in ORDER))
print("-" * 120)
print(f"{'R0: 치수 선형+2차 (기준)':<32}{'13':>8}" +
      "".join(f"{f'{R0[c]:+.3f} ({R0_MAE[c]:.1f}%)':>20}" for c in ORDER))
print(f"{'차종평균만 (형상 정보 0)':<32}{'0':>8}{'+0.226 (9.4%)':>20}")
print("-" * 120)
for r in rows:
    p = f"{r['params']/1e6:.2f}M"
    cells = "".join(f"{f'{r[chr(116)+chr(101)+chr(115)+chr(116)][c][0]:+.3f} ({r[chr(116)+chr(101)+chr(115)+chr(116)][c][1]:.1f}%)':>20}"
                    for c in ORDER)
    print(f"{r['tag']:<32}{p:>8}{cells}")

print("\nΔ vs R0 (R2 차이)")
print(f"{'run':<32}" + "".join(f"{c:>14}" for c in ORDER))
print("-" * 90)
for r in rows:
    print(f"{r['tag']:<32}" + "".join(f"{r['test'][c][0]-R0[c]:>+14.3f}" for c in ORDER))
