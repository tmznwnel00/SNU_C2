import numpy as np
import time
import json
import os
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

##############################################
# 1. 사용자 설정
##############################################

# DATA_FILE = "facebookG.txt"     # "Syn200G.txt" 로 변경 가능
# N_CLUSTERS = 10                 # 데이터셋에 맞게 조정
DATA_FILE = "Syn200G.txt"     # "Syn200G.txt" 로 변경 가능
N_CLUSTERS = 200                 # 데이터셋에 맞게 조정
OUTPUT_DIR = "results"          # 결과 저장 폴더

##############################################
# 2. 환경 준비
##############################################

os.makedirs(OUTPUT_DIR, exist_ok=True)
basename = os.path.splitext(os.path.basename(DATA_FILE))[0]
log_file = os.path.join(OUTPUT_DIR, f"{basename}_spectral_results.json")

##############################################
# 3. 데이터 로드
##############################################

print(f"[INFO] Loading data from {DATA_FILE}...")
X = np.loadtxt(DATA_FILE)

# X shape 확인
print(f"[INFO] Data loaded. Shape = {X.shape}")

##############################################
# 4. CPU Spectral Co-clustering 실행
##############################################

print("[INFO] Running Spectral Co-clustering (CPU baseline)...")
start = time.time()

model = SpectralCoclustering(
    n_clusters=N_CLUSTERS,
    random_state=0,
    svd_method="randomized"
)

model.fit(X)

end = time.time()
runtime = end - start
print(f"[INFO] Done. Runtime = {runtime:.4f} sec")

##############################################
# 5. 평가 지표(NMI / AMI) 계산
##############################################

# sklearn SpectralCoClustering은 행/열 2개 클러스터링 결과를 제공함
row_labels = model.row_labels_
col_labels = model.column_labels_

# GT label이 없으므로 자기-일관성 기반 평가 (optional)
nmi = normalized_mutual_info_score(row_labels, row_labels)
ami = adjusted_mutual_info_score(row_labels, row_labels)

##############################################
# 6. 결과 로그 저장
##############################################

results = {
    "data_file": DATA_FILE,
    "shape": X.shape,
    "n_clusters": N_CLUSTERS,
    "runtime_sec": runtime,
    "row_labels": row_labels.tolist(),
    "col_labels": col_labels.tolist(),
    "metrics": {
        "self_NMI": float(nmi),
        "self_AMI": float(ami)
    }
}

with open(log_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"[INFO] Results saved to {log_file}")
print("[INFO] Test completed successfully!")
