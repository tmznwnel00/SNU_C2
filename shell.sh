cd SNU_C2
conda activate cocluster
conda deactivate

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

launch-shell 1 120 P3

nvcc -c -O3 coclustering_1.cu -o coc1.o -lcusparse -lcublas
nvcc -c -O3 coclustering_2.cu -o coc2.o -lcusparse -lcublas -lcusolver -lcurand
nvcc -c -O3 coclustering_3.cu -o coc3.o -lcublas -lcurand
nvcc -c -O3 coclustering_4.cu -o coc4.o -lcublas

nvcc cocluster_main.cu coc1.o coc2.o coc3.o coc4.o \
    -o cocluster \
    -lcusparse -lcublas -lcusolver -lcurand
./cocluster facebookG.txt

-----


(base) hjin@c02:~/SNU_C2$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

(base) hjin@c02:~/SNU_C2$ nvidia-smi
Wed Dec  3 01:41:48 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:05:00.0 Off |                  Off |
| 30%   29C    P8             17W /  450W |       0MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+