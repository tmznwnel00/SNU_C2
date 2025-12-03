import subprocess
import time
import json
import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import os

############################################
# USER SETTINGS
############################################
DATA_FILE = "facebookG.txt"   # or Syn200G.txt
EXECUTABLE = "./cocluster"    # GPU pipeline exe
N_CLUSTERS = 10               # facebook=10, syn200=200
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

############################################
# 1. CPU BASELINE (Sklearn)
############################################
print("[CPU] Loading data...")

# load edge list
edges = np.loadtxt(DATA_FILE, dtype=int)
max_node = edges.max()
n = max_node + 1

# build adjacency matrix (n x n)
# warning: uses dense matrix; facebookG is ok (4039x4039)
X = np.zeros((n, n), dtype=float)
for u, v in edges:
    X[u, v] = 1.0
    X[v, u] = 1.0   # undirected

print("[CPU] Running SpectralCoclustering...")
t0 = time.time()
model = SpectralCoclustering(n_clusters=N_CLUSTERS, random_state=0)
model.fit(X)
t1 = time.time()

cpu_runtime = t1 - t0
row_cpu = model.row_labels_
col_cpu = model.column_labels_

print(f"[CPU] runtime = {cpu_runtime:.3f} sec")

############################################
# 2. GPU PIPELINE RUN
############################################
print("[GPU] Running GPU pipeline executable...")

t0 = time.time()
proc = subprocess.run(
    [EXECUTABLE, DATA_FILE],
    capture_output=True, text=True
)
t1 = time.time()
gpu_runtime = t1 - t0

if proc.returncode != 0:
    print(proc.stderr)
    raise RuntimeError("GPU pipeline failed.")

stdout = proc.stdout.splitlines()

# parse GPU output: ROW: ... , COL: ...
row_gpu = []
col_gpu = []

for line in stdout:
    if line.startswith("ROW:"):
        items = line.replace("ROW:", "").strip().split()
        row_gpu = list(map(int, items))
    if line.startswith("COL:"):
        items = line.replace("COL:", "").strip().split()
        col_gpu = list(map(int, items))

print(f"[GPU] runtime = {gpu_runtime:.3f} sec")

############################################
# 3. METRICS (NMI, AMI)
############################################

def safe_metric(a, b):
    if len(a) != len(b): return None, None
    return (
        normalized_mutual_info_score(a, b),
        adjusted_mutual_info_score(a, b)
    )

row_nmi, row_ami = safe_metric(row_cpu, row_gpu)
col_nmi, col_ami = safe_metric(col_cpu, col_gpu)

############################################
# 4. SAVE JSON RESULTS
############################################

result = {
    "data_file": DATA_FILE,
    "n_clusters": N_CLUSTERS,

    "cpu_runtime_sec": cpu_runtime,
    "gpu_runtime_sec": gpu_runtime,
    "speedup": cpu_runtime / gpu_runtime if gpu_runtime > 0 else None,

    "row_nmi": row_nmi,
    "row_ami": row_ami,
    "col_nmi": col_nmi,
    "col_ami": col_ami,

    "row_labels_cpu": row_cpu.tolist(),
    "col_labels_cpu": col_cpu.tolist(),
    "row_labels_gpu": row_gpu,
    "col_labels_gpu": col_gpu,
}

outfile = os.path.join(OUTPUT_DIR,
                       f"{os.path.splitext(DATA_FILE)[0]}_compare.json")

with open(outfile, 'w') as f:
    json.dump(result, f, indent=4)

print(f"[INFO] saved results to {outfile}")
print("Done.")
