#!/usr/bin/env bash
# Wait for a Codex session to finish, then start the full-dataset indexing.
#
# Trigger detection: the session's rollout JSONL ends with a
# {"type":"event_msg","payload":{"type":"task_complete",...}} event and the
# file size stays stable for STABLE_MINUTES. Idempotent: after triggering
# once, a marker file prevents re-triggering. Safe to run from cron every
# few minutes, or as a long-lived background loop.
#
# Usage:
#   SESSION_ID=019fea0c-... ./scripts/wait_for_session_and_index.sh
set -u

SESSION_ID="${SESSION_ID:-}"
ROLLOUT="${ROLLOUT:-}"
MARKER="${MARKER:-/tmp/litrag_triggered_${SESSION_ID:-default}.marker}"
STABLE_MINUTES="${STABLE_MINUTES:-5}"
WORKTREE="${WORKTREE:-/home/shiqiany/AIagent/phomatagent-literature-rag}"
LITERATURE_ROOT="${LITERATURE_ROOT:-/home/shiqiany/AIagent/Photoelectric detection/dataset/paper}"
INDEX_DIR="${INDEX_DIR:-output/literature_index}"
LOG_FILE="${LOG_FILE:-output/full_index.log}"
STATUS_FILE="${STATUS_FILE:-output/full_index_status.json}"
PYTHON="${PYTHON:-$WORKTREE/.venv/bin/python}"

if [ -z "$SESSION_ID" ] && [ -z "$ROLLOUT" ]; then
  echo "ERROR: set SESSION_ID or ROLLOUT" >&2
  exit 2
fi

if [ -z "$ROLLOUT" ]; then
  ROLLOUT="/mnt/c/Users/牧之原/.codex/sessions/2026/08/10/rollout-*${SESSION_ID}.jsonl"
fi

if [ -f "$MARKER" ]; then
  echo "already triggered ($MARKER); exiting"
  exit 0
fi

ROLLOUT_FILE="$(ls -t $ROLLOUT 2>/dev/null | head -1)"
if [ -z "$ROLLOUT_FILE" ] || [ ! -f "$ROLLOUT_FILE" ]; then
  echo "rollout file not found for session ${SESSION_ID:-$ROLLOUT}"
  exit 0
fi
echo "watching: $ROLLOUT_FILE"

LAST_EVENT="$(tail -c 4000 "$ROLLOUT_FILE" | grep -o '"type":"task_complete"' | tail -1)"
if [ -z "$LAST_EVENT" ]; then
  echo "session still running (no task_complete yet); retry later"
  exit 0
fi

# Stability check: file size unchanged for STABLE_MINUTES.
SIZE1="$(stat -c %s "$ROLLOUT_FILE")"
sleep "${STABLE_MINUTES}m"
SIZE2="$(stat -c %s "$ROLLOUT_FILE" 2>/dev/null || echo "$SIZE1")"
if [ "$SIZE1" != "$SIZE2" ]; then
  echo "session still writing after task_complete; retry later"
  exit 0
fi

echo "session completed; launching full-dataset indexing"
touch "$MARKER"
cd "$WORKTREE" || exit 1
nohup env \
  PHOTOMATAGENT_LITERATURE_DIR="$LITERATURE_ROOT" \
  PHOTOMATAGENT_LITERATURE_INDEX_DIR="$INDEX_DIR" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  "$PYTHON" scripts/run_full_index.py \
  --log-file "$LOG_FILE" --status-file "$STATUS_FILE" \
  > "$WORKTREE/output/full_index_launcher.log" 2>&1 &
echo "indexing launched (pid $!); logs: $LOG_FILE"
