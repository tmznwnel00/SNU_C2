#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "Benchmark script for SpectralCoclustering (sklearn) and CUDA sources"
echo

TIME_BIN="/usr/bin/time -p"

echo "1) sklearn SpectralCoclustering on facebookG.txt (n_clusters=10)"
${TIME_BIN} python3 - <<'PY'
from sklearn.cluster import SpectralCoclustering
import numpy as np
file = 'facebookG.txt'
X = np.array(np.loadtxt(file))
clustering = SpectralCoclustering(n_clusters=10, random_state=0).fit(X)
print(clustering)
PY

echo
echo "2) sklearn SpectralCoclustering on Syn200G.txt (n_clusters=200)"
${TIME_BIN} python3 - <<'PY'
from sklearn.cluster import SpectralCoclustering
import numpy as np
file = 'Syn200G.txt'
X = np.array(np.loadtxt(file))
clustering = SpectralCoclustering(n_clusters=200, random_state=0).fit(X)
print(clustering)
PY

echo
echo "3) Attempt to compile CUDA sources and measure compile time"
NVCC=$(command -v nvcc || true)
if [ -z "$NVCC" ]; then
  echo "nvcc not found. To compile CUDA sources install the CUDA toolkit (nvcc)."
  echo "On Debian/Ubuntu you can try: sudo apt install nvidia-cuda-toolkit (or install NVIDIA CUDA from NVIDIA)."
  echo "This script will skip CUDA compilation."
else
  for f in coclustering_1.cu coclustering_2.cu coclustering_3.cu coclustering_4.cu; do
    if [ -f "$f" ]; then
      echo
      echo "Compiling $f"
      ${TIME_BIN} "$NVCC" -O3 -lcublas -lcusparse -lcusolver -lcurand -c "$f" -o "${f%.cu}.o" || echo "Compilation failed for $f"
    else
      echo "File $f not found, skipping"
    fi
  done
fi

echo
echo "Done. Notes:"
echo "- The script measured Python sklearn runs directly and attempted to time CUDA compilations." 
echo "- Running the CUDA implementations end-to-end requires building executables or Python bindings that call the functions inside the .cu files;" 
echo "  I can add small driver programs that load your datasets (dense or CSR) and call the routines, then benchmark their runtime — tell me if you want me to implement these drivers and run them (requires nvcc + GPU)."
