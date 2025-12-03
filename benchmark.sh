#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

TIME="/usr/bin/time -p"

echo "=============================================="
echo "   Spectral Co-clustering Benchmark Script"
echo "=============================================="
echo

# -----------------------------
# 1) CPU baseline (sklearn)
# -----------------------------
echo "[CPU] Running sklearn SpectralCoclustering ..."

CPU_TIME=$(
  { 
    ${TIME} python3 - <<'PY' 2>&1
from sklearn.cluster import SpectralCoclustering
import numpy as np

file = 'facebookG.txt'
X = np.array(np.loadtxt(file))
clustering = SpectralCoclustering(n_clusters=10, random_state=0).fit(X)
PY
  } 2>&1 | awk '/real/ {print $2}'
)

echo "[CPU] runtime = ${CPU_TIME} sec"
echo


# -----------------------------
# 2) GPU pipeline
# -----------------------------
echo "[GPU] Running CUDA coclustering executable ..."

GPU_TIME=$(
  { 
    ${TIME} ./cocluster facebookG.txt > /dev/null
  } 2>&1 | awk '/real/ {print $2}'
)

echo "[GPU] runtime = ${GPU_TIME} sec"
echo


# -----------------------------
# 3) Speedup
# -----------------------------
CPU_FLOAT=$(printf "%.6f" "$CPU_TIME")
GPU_FLOAT=$(printf "%.6f" "$GPU_TIME")

SPEEDUP=$(python3 - <<PY
cpu = float("$CPU_TIME")
gpu = float("$GPU_TIME")
print(cpu/gpu)
PY
)

echo "[RESULT] speedup = ${SPEEDUP}x faster"
echo


# -----------------------------
# Save result
# -----------------------------
mkdir -p results
echo "{"                                    > results/benchmark_facebookG.json
echo "  \"cpu_time\": $CPU_TIME,"          >> results/benchmark_facebookG.json
echo "  \"gpu_time\": $GPU_TIME,"          >> results/benchmark_facebookG.json
echo "  \"speedup\": $SPEEDUP"             >> results/benchmark_facebookG.json
echo "}"                                   >> results/benchmark_facebookG.json

echo "[INFO] saved results to results/benchmark_facebookG.json"
echo "Done."


# -----------------------------
# -----------------------------

#!/usr/bin/env bash

TIME="/usr/bin/time -p"

echo "=============================================="
echo "   Spectral Co-clustering Benchmark Script"
echo "=============================================="
echo

# -----------------------------
# 1) CPU baseline (sklearn)
# -----------------------------
echo "[CPU] Running sklearn SpectralCoclustering ..."

CPU_TIME=$(
  { 
    ${TIME} python3 - <<'PY' 2>&1
from sklearn.cluster import SpectralCoclustering
import numpy as np

file = 'Syn200G.txt'
X = np.array(np.loadtxt(file))
clustering = SpectralCoclustering(n_clusters=200, random_state=0).fit(X)
PY
  } 2>&1 | awk '/real/ {print $2}'
)

echo "[CPU] runtime = ${CPU_TIME} sec"
echo


# -----------------------------
# 2) GPU pipeline
# -----------------------------
echo "[GPU] Running CUDA coclustering executable ..."

GPU_TIME=$(
  { 
    ${TIME} ./cocluster Syn200G.txt > /dev/null
  } 2>&1 | awk '/real/ {print $2}'
)

echo "[GPU] runtime = ${GPU_TIME} sec"
echo


# -----------------------------
# 3) Speedup
# -----------------------------
CPU_FLOAT=$(printf "%.6f" "$CPU_TIME")
GPU_FLOAT=$(printf "%.6f" "$GPU_TIME")

SPEEDUP=$(python3 - <<PY
cpu = float("$CPU_TIME")
gpu = float("$GPU_TIME")
print(cpu/gpu)
PY
)

echo "[RESULT] speedup = ${SPEEDUP}x faster"
echo


# -----------------------------
# Save result
# -----------------------------
mkdir -p results
echo "{"                                    > results/benchmark_Syn200G.json
echo "  \"cpu_time\": $CPU_TIME,"          >> results/benchmark_Syn200G.json
echo "  \"gpu_time\": $GPU_TIME,"          >> results/benchmark_Syn200G.json
echo "  \"speedup\": $SPEEDUP"             >> results/benchmark_Syn200G.json
echo "}"                                   >> results/benchmark_Syn200G.json

echo "[INFO] saved results to results/benchmark_Syn200G.json"
echo "Done."
