#!/bin/bash
# Auto-launch breakdown benchmarks after speed phase completes
# Usage: bash artifacts/debug/022_linear_time_proportion_study/auto_launch_breakdown.sh

ARTIFACT_DIR="/root/wja/project/my/cospaq/fake/artifacts/debug/022_linear_time_proportion_study"
RUN_STUDY="$ARTIFACT_DIR/run_study.py"

echo "Waiting for speed benchmarks to complete..."
echo "Checking every 30s for completion..."

# Wait for all 3 speed logs to show "Done."
while true; do
    done_count=0
    for model in 2b 4b 9b; do
        if grep -q "Done\." "$ARTIFACT_DIR/logs/${model}_speed_all.log" 2>/dev/null; then
            done_count=$((done_count + 1))
        fi
    done
    if [ $done_count -eq 3 ]; then
        echo "All speed benchmarks done at $(date)!"
        break
    fi
    echo "  $(date): $done_count/3 done"
    sleep 30
done

echo ""
echo "=== Launching breakdown benchmarks ==="

CUDA_VISIBLE_DEVICES=5 python "$RUN_STUDY" --model 2B --phase breakdown --scenario prefill_only 2>&1 &
PID1=$!
echo "2B breakdown PID: $PID1"

CUDA_VISIBLE_DEVICES=6 python "$RUN_STUDY" --model 4B --phase breakdown --scenario prefill_only 2>&1 &
PID2=$!
echo "4B breakdown PID: $PID2"

CUDA_VISIBLE_DEVICES=7 python "$RUN_STUDY" --model 9B --phase breakdown --scenario prefill_only 2>&1 &
PID3=$!
echo "9B breakdown PID: $PID3"

wait $PID1 $PID2 $PID3
echo "All prefill_only breakdown benchmarks done."

echo ""
echo "=== Launching prefill-decode breakdown ==="

CUDA_VISIBLE_DEVICES=5 python "$RUN_STUDY" --model 2B --phase breakdown --scenario prefill_decode 2>&1 &
PID1=$!
echo "2B breakdown decode PID: $PID1"

CUDA_VISIBLE_DEVICES=6 python "$RUN_STUDY" --model 4B --phase breakdown --scenario prefill_decode 2>&1 &
PID2=$!
echo "4B breakdown decode PID: $PID2"

CUDA_VISIBLE_DEVICES=7 python "$RUN_STUDY" --model 9B --phase breakdown --scenario prefill_decode 2>&1 &
PID3=$!
echo "9B breakdown decode PID: $PID3"

wait $PID1 $PID2 $PID3
echo "All breakdown benchmarks done at $(date)!"
echo "Ready for analysis."