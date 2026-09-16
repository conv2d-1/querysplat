# hAlgorithm 

## Clone:
  ```
  git clone ssh://git@code.hesaitech.com:10022/TM_Algorithm/TMA.git -b algorithm
  cd TMA
  ```
  
## Install
  - 设置 cuda：
  ```
  export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
  ```
  - 创建 python10 环境
  ```
  pip install -r requirements/requirement_hAlgorithm_cu124.txt
  pip3 install git+ssh://git@code.hesaitech.com:10022/open-projects/diff-gaussian-rasterization-modified.git
  pip3 install git+ssh://git@code.hesaitech.com:10022/open-projects/fused-ssim.git
  pip3 install git+ssh://git@code.hesaitech.com:10022/open-projects/pytorch3d-0.7.8.git
  ```

## Run
- train:
  ```
  bash hAlgorithm/script/train_dist.sh
  ```

- test
  ```
  bash hAlgorithm/script/test_dist.sh

  # EvalDepth
  bash hAlgorithm/script/evaldepth/test.sh [test results path]
  ```
  

# FFGS & finetune GS
- 1.feedforward GS test for a GS ply

  ```
  bash hAlgorithm/scipt/test_real_dist.sh
  # 修改配置CONFIG="hAlgorithm/configs/ffgs_finetune/ffgs_lzh_250418/more_data_gs_pgsr_0509.py"

  ```

- 2.load and train with the ffgs ply

  ```
  由第1步得到的ply找到其路径:
  "results/test_gs_Zed2FT/0509ffgs/highres_132601/more_data_gs_pgsr_0509_20250512-113648/visualization/iter_000001/real_mf_ft/gaussians/000000/gaussians.ply"
  
  写进hAlgorithm/configs/ffgs_finetune/finetune_gs/250418_finetune_gs.py line98的pretrain_ply

  开始训练 # 注意修改CONFIG CONFIG="hAlgorithm/configs/ffgs_finetune/finetune_gs/250418_finetune_gs.py"
  bash hAlgorithm/scipt/train_real_dist.sh

  ```


# hBenchmark 

## Clone:
  ```
  git clone ssh://git@code.hesaitech.com:10022/TM_Algorithm/TMA.git -b benchmark
  cd TMA
  ```
  
## Install
  - 设置 cuda：
  ```
  export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
  ```
  - 创建 python10 环境
  ```
  pip install -r requirements/requirement_hAlgorithm_cu124.txt
  pip3 install git+ssh://git@code.hesaitech.com:10022/open-projects/diff-gaussian-rasterization-modified.git
  pip3 install git+ssh://git@code.hesaitech.com:10022/open-projects/fused-ssim.git
  pip3 install git+ssh://git@code.hesaitech.com:10022/open-projects/pytorch3d-0.7.8.git
  ```

## Eval

- eval
  ```
  bash hBenchmark/script/test_depth.sh
  bash hBenchmark/script/test_temporal.sh
  bash hBenchmark/script/test_resutruction.sh
  bash hBenchmark/script/test_pose.sh

  ```

