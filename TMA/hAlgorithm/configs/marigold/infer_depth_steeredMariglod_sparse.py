data = dict(
    val="hAlgorithm/configs/dataset/diffuser/sparse/nyu_test.yaml",
    vis="hAlgorithm/configs/dataset/diffuser/sparse/nyu_test.yaml",
)

model = dict(
    pretrained_model_name_or_path="/mnt/netdata/Project/TMD/weights/Marigold/marigold-depth-lcm-v1-0",
    denoising_steps=4,
    processing_resolution=768,
)

trainer = dict(
    num_workers=1,
    batch_size=1,
)
