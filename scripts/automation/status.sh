#!/bin/bash
echo "=== Chain state ==="
cat /workspace/automation/flags/CHAIN.status 2>/dev/null || echo "(no chain flag)"
echo ""
echo "=== Individual run flags ==="
for f in /workspace/automation/flags/RUN-*.status; do
    [ -f "$f" ] && { echo "--- $(basename $f) ---"; cat "$f"; echo; }
done
echo ""
echo "=== Watchdog anomaly? ==="
if [ -f /workspace/automation/flags/WATCHDOG.status ]; then
    cat /workspace/automation/flags/WATCHDOG.status
else
    echo "No anomaly flagged."
fi
echo ""
echo "=== Chain log (last 20) ==="
tail -20 /workspace/automation/logs/chain.log
echo ""
echo "=== Watchdog log (last 5) ==="
tail -5 /workspace/automation/logs/watchdog.log
