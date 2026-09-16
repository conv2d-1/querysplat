export CUDA_VISIBLE_DEVICES=0

chunk=100
overlap=0
step=1

config=results/da3_ppt_all_251223/mvfr_da3g_cam_ppt_all_251209_mvpdc_bs16_2f16f_50k_20251224-232949/mvfr_da3g_cam_ppt_all_251209_mvpdc_bs16_2f16f_50k.py
load_from=latest

python hAlgorithm/script/infer/mvfr_kosmo_demo.py \
    --config $config \
    --load_from $load_from \
    --cam 0 1 2 3 4 5 6 7 \
    --chunk $chunk \
    --overlap $overlap \
    --step $step \
    --model.model.glb_points_head None \
    --model.model.depth_head.chunk_size 1 \
    --depth_name pred_depth_mvfr \
    --conf_name pred_confidence_mvfr \
    --output_dir /mnt/netdata/Team/AI/personal/ts/results/kosmo_results/20251227_4_会议室_tsh_${chunk}_${overlap}_${step} \
    --data /mnt/home/stang/workspace/TMA2/results/kosmo_results/20251227_4_会议室_tsh/sv_da3b2_all_251207_bs4_500k_stage_kosmo_20251230-114911/data_hgaussian_v1.0.1_normal_sv_v251225_sfm_colmap_v1.0.0_mask_mos_v1.0.0_depth_sv_v251225@20251227100001.json \
    # --nums 10
    