export CUDA_VISIBLE_DEVICES=3

export OMP_NUM_THREADS=1 
export PATH="/mnt/netdata/Team/SLAM/personal/lx/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/netdata/Team/SLAM/personal/lx/cuda-12.4/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=/mnt/netdata/Team/SLAM/personal/lx/cuda-12.4

python hAlgorithm/script/infer/mvfr_kosmo_demo.py \
    --config hAlgorithm/configs/open/da3/da3_251114_4f_giant_sv.py \
    --sv_conf_ratio 0.3 \
    --fake_sv \
    --process_res 504 \
    --cam 0 1 2 3 4 5 \
    --chunk 1 \
    --depth_name lidar_depth \
    --output_dir /mnt/nasTeam/Kosmo/processed_data/kosmo/20251231_精品/10F大厅/da3_2/ \
    --data /mnt/nasTeam/Kosmo/processed_data/kosmo/20251231_精品/10F大厅/raw_data/data_hgaussian_v1.0.1_normal_svtrt_v251117_sfm_colmap_v1.0.0_mask_mos_v1.0.0_depth_svtrt_v251117@20251222210000.json \
    # --nums 100 \
    # --sv_points \

