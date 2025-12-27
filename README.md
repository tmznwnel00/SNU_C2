**Spectral Co-clustering (GPU-optimized)**

간단 소개
- 이 저장소는 Spectral Co-clustering 알고리즘을 GPU(CUDA)로 가속한 구현입니다. 전체 파이프라인(정규화 및 k-means)을 GPU에서 실행하도록 분리된 helper 모듈들로 구성되어 있습니다.

**저장소 구성**
- **소스코드:** [cocluster_main.cu](project/cocluster_main.cu), helper 파일들: [coclustering_1.cu](project/coclustering_1.cu), [coclustering_3.cu](project/coclustering_3.cu), [coclustering_4.cu](project/coclustering_4.cu) 등
- **실행 파일:** `cocluster` (빌드 후 생성)
- **입력 데이터(예시):** `facebookG.txt`, `Syn200G.txt`
- **벤치마크 스크립트:** `benchmark.sh` (CPU baseline vs GPU 구현 비교 실행)

**전제 조건(필수)**
- CUDA 툴킷 및 `nvcc`
- cuSPARSE, cuBLAS, cuSOLVER, cuRAND 개발 라이브러리
- Linux 환경에서 빌드 권장

**B. 빌드 방법**
프로젝트 루트 디렉토리에서 아래 명령으로 빌드합니다:

```bash
make clean && make all && make main
```

빌드가 완료되면 `cocluster` 실행 파일이 생성됩니다.

**C. 제공 데이터셋으로 프로그램 실행 방법**
예시 데이터셋을 이용해 실행하는 방법:

```bash
./cocluster facebookG.txt
./cocluster Syn200G.txt
```

파일 경로가 다르면 상대/절대 경로를 지정하여 실행하세요.

**D. 벤치마크 실행 방법 (CPU baseline vs. GPU)**
CPU baseline과 GPU 구현을 비교하는 벤치마크는 다음 명령으로 실행합니다:

```bash
bash benchmark.sh
```

**추가 참고 및 디버깅 팁**
- 런타임/링크 오류가 발생하면 CUDA 라이브러리 경로와 `LD_LIBRARY_PATH`를 확인하세요.
- 빌드 옵션이나 SVD/k-means 파라미터는 `cocluster_main.cu` 내 변수(k, p, q, iter 등)를 조정해 실험할 수 있습니다.
- GPU 메모리 부족 시 입력 그래프 크기 또는 `k` 값을 줄여 테스트하세요.
