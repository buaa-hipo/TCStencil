## Require
 - A100 and V100
 - CUDA 11.0+
 - Nvidia Nsight Compute

 ## Getting Started
 1. run `nvidia-smi -lgc 1410` or `nvidia-smi -lgc 1380` on A100 and V100 respectively
 2. run `python3 script/nvcc_compile.py --arch A100` on A100 platform and `python3 script/nvcc_compile.py --arch V100` on V100 platform. Keep the generated `txt` file
 3. run `python3 script/process_data.py` to process data, and the resulting csv file appears under the `data` folder
 4. run `python3 draw.py`, generate related figures in the `figure` folder

