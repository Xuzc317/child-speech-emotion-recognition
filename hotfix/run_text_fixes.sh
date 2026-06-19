#!/usr/bin/env bash
# hotfix/run_text_fixes.sh
# Applies text fixes: Chunk A (numbers), C (B6 direction), D (B2 numbers), E (spelling).
# Backs up each target to <file>.bak (no-clobber) before editing, then verifies.
set -euo pipefail

# --- locate targets (no `find`; known repo-relative paths) ---
pick() { for p in "$@"; do [ -f "$p" ] && { printf '%s\n' "$p"; return 0; }; done; return 1; }

DOC1="$(pick "docs/current/补充实验方案_v1.md" "补充实验方案_v1.md")" \
  || { echo "ERROR: 补充实验方案_v1.md not found (run from repo root)"; exit 1; }
CLAUDEMD="$(pick "CLAUDE.md" "./CLAUDE.md")" \
  || { echo "ERROR: CLAUDE.md not found (run from repo root)"; exit 1; }
echo "Target 1: $DOC1"
echo "Target 2: $CLAUDEMD"

# --- backup (no-clobber: re-runs keep the true original) ---
backup() {
  if [ ! -f "$1.bak" ]; then cp "$1" "$1.bak"; echo "backup: $1 -> $1.bak";
  else echo "backup exists, kept: $1.bak"; fi
}
backup "$DOC1"
backup "$CLAUDEMD"

# === Chunk A (line-anchored numbers) + Chunk E (spelling) on DOC1 ===
# Chunk E masks the legitimate word "SelfAttention" so it is never corrupted,
# fixes the bare "SelfAtten" typo, then restores "SelfAttention".
sed -i \
  -e '15s/97\.13%/97.22%/' \
  -e '17s/67\.15%/63.98%/' \
  -e 's/SelfAttention/<<<KEEPATTN>>>/g' \
  -e 's/SelfAtten/SelfAttn/g' \
  -e 's/<<<KEEPATTN>>>/SelfAttention/g' \
  "$DOC1"

# === Chunk C (B6 direction) + Chunk D (B2 numbers) on CLAUDE.md ===
sed -i \
  -e 's/消融从全栈出发逐步移除模块（非逐步添加）/cumulative build-up design（累加式）/' \
  -e 's/去Adapter +0\.28pp/加Adapter +0.28pp/' \
  -e 's/换MeanPool +11pp/换SelfAttn +11pp/' \
  -e 's/去WF微弱+0\.05pp/加WF微弱+0.05pp/' \
  -e 's/去WF反而-0\.84pp/加Adapter反而-0.84pp/' \
  -e 's/去WF +0\.18pp/加Adapter +0.18pp/' \
  -e 's/Pooling 从 SA→MeanPool 是关键提升/Pooling 从 Mean→SelfAttn 是关键提升/' \
  -e 's/34-41%/19.17%-35.47%/g' \
  "$CLAUDEMD"

# === verification (short report; [WARN] = target string not found as expected) ===
echo
echo "== verification =="
have() { local n; n=$(grep -Fc -- "$2" "$1" 2>/dev/null || true); n=${n:-0}; \
  if [ "$n" -ge 1 ]; then echo "  [OK]   present: $2"; else echo "  [WARN] MISSING: $2"; fi; }
gone() { local n; n=$(grep -Fc -- "$2" "$1" 2>/dev/null || true); n=${n:-0}; \
  if [ "$n" -eq 0 ]; then echo "  [OK]   removed: $2"; else echo "  [WARN] REMAINS($n): $2"; fi; }

echo "[$DOC1]"
have "$DOC1" "97.22%"; gone "$DOC1" "97.13%"
have "$DOC1" "63.98%"; gone "$DOC1" "67.15%"
sa=$( { grep -Fo "SelfAtten"      "$DOC1" || true; } | wc -l | tr -d '[:space:]' )
sat=$( { grep -Fo "SelfAttention" "$DOC1" || true; } | wc -l | tr -d '[:space:]' )
san=$( { grep -Fo "SelfAttn"      "$DOC1" || true; } | wc -l | tr -d '[:space:]' )
echo "  counts: SelfAtten=$sa SelfAttention=$sat SelfAttn=$san"
if [ "$sa" -eq "$sat" ]; then echo "  [OK]   no bare 'SelfAtten' typo remains";
else echo "  [WARN] bare 'SelfAtten' typo may remain"; fi

echo "[$CLAUDEMD]"
have "$CLAUDEMD" "cumulative build-up design（累加式）"; gone "$CLAUDEMD" "消融从全栈出发逐步移除模块"
have "$CLAUDEMD" "加Adapter +0.28pp";        gone "$CLAUDEMD" "去Adapter +0.28pp"
have "$CLAUDEMD" "换SelfAttn +11pp";          gone "$CLAUDEMD" "换MeanPool +11pp"
have "$CLAUDEMD" "加WF微弱+0.05pp";           gone "$CLAUDEMD" "去WF微弱+0.05pp"
have "$CLAUDEMD" "加Adapter反而-0.84pp";      gone "$CLAUDEMD" "去WF反而-0.84pp"
have "$CLAUDEMD" "加Adapter +0.18pp";         gone "$CLAUDEMD" "去WF +0.18pp"
have "$CLAUDEMD" "Pooling 从 Mean→SelfAttn 是关键提升"; gone "$CLAUDEMD" "Pooling 从 SA→MeanPool"
have "$CLAUDEMD" "19.17%-35.47%";             gone "$CLAUDEMD" "34-41%"

echo
echo "Text fixes done. Backups: *.bak  (restore: mv <file>.bak <file>)"
