export CUDA_VISIBLE_DEVICES=0

python hAlgorithm/script/infer/mvfr_kosmo_demo.py \
    --config results/da3_sv_ppt_all_251208/sv_da3b2_all_251207_bs4_500k_stage_kosmo_20251208-220808/sv_da3b2_all_251207_bs4_500k_stage_kosmo.py \
    --load_from latest \
    --fake_sv \
    --process_res 840 \
    --cam 0 1 2 3 4 5 6 7 \
    --chunk 1 \
    --depth_name lidar_depth \
    --output_dir /mnt/netdata/Team/AI/personal/ts/results/kosmo_results/20251227_4_会议室_tsh/ \
    --data /mnt/nasTeam/Kosmo/processed_data/kosmo/20251227_4_会议室_tsh/劳特/raw_data/data_hgaussian_v1.0.1_normal_sv_v251225_sfm_colmap_v1.0.0_mask_mos_v1.0.0_depth_sv_v251225@20251227100001.json \
    --sv_points \
    # --nums 4 \
    
