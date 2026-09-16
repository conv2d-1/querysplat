export CUDA_VISIBLE_DEVICES=0

python hAlgorithm/script/kosmo/vggt_normal_20251105.py \
    --fake_sv \
    --cam 0 1 2 3 \
    --depth_name lidar_depth \
    --output_dir ./results/kosmo_results/split_onnx/tensorrt_one \
    --data /mnt/nasTeam/Kosmo/processed_data/kosmo/20251227_4_会议室_tsh/劳特/raw_data/data_hgaussian_v1.0.1_normal_sv_v251225_sfm_colmap_v1.0.0_mask_mos_v1.0.0_depth_sv_v251225@20251227100001.json \
    --sv_points \
    --nums 100