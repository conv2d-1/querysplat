set -e -v

export CUDA_VISIBLE_DEVICES=0

JSON=$1

python EvalDepth/tools/test.py \
    --result_json $JSON \
    --process_type None \
    --test_type all \
    --data_root /mnt/netdata/Team/AI/datasets/TMD/ \
    --alignment None \
    --test_type all \
    --range 200.0 \
    --launcher None \
    --batch_size 1 \
    # --vis \
    # --show_dir ./ \

