#!/bin/bash
# qi/ 의 코드·문서를 팀 저장소(dongwon 브랜치)의 ml/ 로 동기화한다.
#
#   bash scripts/sync_to_team_repo.sh "커밋 메시지"
#
# 데이터·모델·PDF·PPTX·이미지는 .gitignore로 자동 제외된다 (공개 저장소이므로 필수).
# 인증이 없으면 커밋까지만 하고 push 명령을 안내한다.
set -e

QI=/home/kwy00/qi
TR="${TEAM_REPO:-$HOME/team_repo}"        # 팀 저장소 클론 위치 (TEAM_REPO로 변경 가능)
REMOTE=https://github.com/parag0hz/Qulcomm_Institute_team_a
BRANCH=dongwon
MSG="${1:-Update ml/: 실험 결과 및 스크립트 갱신}"

# --- 팀 저장소 준비 ---
if [ ! -d "$TR/.git" ]; then
  echo "▶ 팀 저장소 클론: $TR"
  git clone -q -b "$BRANCH" "$REMOTE" "$TR"
fi
cd "$TR"
git fetch -q origin
git checkout -q "$BRANCH"
git merge -q --ff-only "origin/$BRANCH" 2>/dev/null || {
  echo "⚠ 로컬 $BRANCH 가 원격과 갈라졌다. 수동 확인 필요."; exit 1; }

# --- qi/ 에서 추적 대상 파일만 복사 ---
cd "$QI"
FILES=$(git ls-files; git ls-files --others --exclude-standard)   # 추적 + 미추적(무시 제외)
cd "$TR"
rm -rf ml_tmp && mkdir -p ml_tmp
cd "$QI"
echo "$FILES" | while read -r f; do
  [ -f "$f" ] || continue
  case "$f" in *.pdf|*.pptx|*.pkl|*.npz|*.pt|*.pth|*.tar|*.png|*.gif) continue;; esac
  mkdir -p "$TR/ml_tmp/$(dirname "$f")"
  cp "$f" "$TR/ml_tmp/$f"
done
# demo_holdout.json 은 data/ 이지만 설정이라 포함
[ -f "$QI/data/demo_holdout.json" ] && {
  mkdir -p "$TR/ml_tmp/data"; cp "$QI/data/demo_holdout.json" "$TR/ml_tmp/data/"; }

cd "$TR"
rsync -a --delete --exclude='data/' --exclude='outputs/' ml_tmp/ ml/
[ -f ml_tmp/data/demo_holdout.json ] && { mkdir -p ml/data; cp ml_tmp/data/demo_holdout.json ml/data/; }
rm -rf ml_tmp

# --- 안전 검사 ---
git add -A ml
git add -f ml/data/demo_holdout.json 2>/dev/null || true
BAD=$(git diff --cached --name-only | grep -iE '\.pdf$|\.pptx$|\.pkl$|\.npz$|\.pt$|\.pth$|\.tar$|\.png$|\.gif$' || true)
if [ -n "$BAD" ]; then
  echo "⚠ 금지 항목이 스테이징됨 — 중단:"; echo "$BAD"; git reset -q; exit 1
fi
BIG=$(git diff --cached --name-only | while read -r f; do
        if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 10485760 ]; then echo "$f"; fi
      done) || true
if [ -n "$BIG" ]; then
  echo "⚠ 10MB 초과 파일 — 중단:"; echo "$BIG"; git reset -q; exit 1
fi

if git diff --cached --quiet; then
  echo "✅ 변경사항 없음 (동기화 이미 최신)"; exit 0
fi

echo "▶ 변경 파일:"; git diff --cached --name-only | sed 's/^/    /'
git -c user.name="dongwon" -c user.email="dev@beavertalk.im" commit -q -m "$MSG"
echo "✅ 커밋 완료: $(git log --oneline -1)"

if git push -q origin "$BRANCH" 2>/dev/null; then
  echo "✅ 푸시 완료 → $REMOTE ($BRANCH)"
else
  echo ""
  echo "⚠ 인증이 없어 푸시 실패. 아래를 직접 실행하세요:"
  echo "    cd $TR && git push origin $BRANCH"
fi
