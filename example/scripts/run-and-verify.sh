#!/usr/bin/env bash
# Automated harness: relaunches the app, polls Documents/test-results.json,
# and reports pass/fail without any manual screenshot inspection.
#
# Usage:  ./scripts/run-and-verify.sh
set -euo pipefail

DEVICE="iPhone 17 Pro"
BUNDLE_ID="org.reactjs.native.example.RnMcpDemo"
APP_DATA_DIR=$(xcrun simctl get_app_container "$DEVICE" "$BUNDLE_ID" data)
RESULTS_FILE="$APP_DATA_DIR/Documents/test-results.json"
TIMEOUT=420

echo "▶ Removing prior results"
rm -f "$RESULTS_FILE"

echo "▶ Terminating app"
xcrun simctl terminate "$DEVICE" "$BUNDLE_ID" 2>&1 || true
sleep 1

echo "▶ Launching app — auto-harness will run on mount"
xcrun simctl launch "$DEVICE" "$BUNDLE_ID"

echo "▶ Polling $RESULTS_FILE (timeout ${TIMEOUT}s)"
START=$(date +%s)
LAST_STATUS=""
while true; do
  if [ -f "$RESULTS_FILE" ]; then
    STATUS=$(grep -o '"status": "[^"]*"' "$RESULTS_FILE" | head -1 | cut -d'"' -f4 || true)
    if [ "$STATUS" != "$LAST_STATUS" ]; then
      printf "  [%4ds] status=%s\n" "$(( $(date +%s) - START ))" "$STATUS"
      LAST_STATUS="$STATUS"
    fi
    if [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ]; then break; fi
  fi
  if [ $(( $(date +%s) - START )) -gt "$TIMEOUT" ]; then
    echo "✗ TIMEOUT after ${TIMEOUT}s"
    exit 124
  fi
  sleep 2
done

echo
echo "▶ Final results:"
cat "$RESULTS_FILE" | python3 -m json.tool
echo
PASSED=$(grep -c '"passed": true' "$RESULTS_FILE" || true)
FAILED=$(grep -c '"passed": false' "$RESULTS_FILE" || true)
echo "Summary: $PASSED passed, $FAILED failed."
[ "$FAILED" -eq 0 ]
