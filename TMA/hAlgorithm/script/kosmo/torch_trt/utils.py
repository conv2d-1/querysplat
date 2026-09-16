import os, sys
import torch
import tensorrt as trt
from datetime import datetime
from functools import partial

def torch_dtype_from_trt(dtype):
    import tensorrt as trt
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f"Unsupported TRT dtype: {dtype}")


def infer_trt_with_torch(engine, input_tensors_dict, output_shapes, global_context, stream):
    """
    input_tensors_dict: dict like {'image': tensor1, 'prompt_depth': tensor2, ...}
    Returns: dict of output tensors
    """
    # 1. 设置输入张量地址
    for name, tensor in input_tensors_dict.items():
        tensor = tensor.cuda()
        if not global_context.set_input_shape(name, tensor.shape):
            raise RuntimeError(f"Failed to set input shape for {name} to {tensor.shape}. required by {engine.get_tensor_shape(name)}")
        global_context.set_tensor_address(name, tensor.data_ptr())

    # 2. 为输出张量分配内存（使用预计算 shape）
    output_tensors = {}
    for name, shape in output_shapes.items():
        dtype = torch_dtype_from_trt(engine.get_tensor_dtype(name))
        # output_tensor = input_tensors_dict["image"].new_zeros(shape)#.to(dtype)
        output_tensor = torch.empty(shape, dtype=dtype, device='cuda')
        global_context.set_tensor_address(name, output_tensor.data_ptr())
        output_tensors[name] = output_tensor
    
    # 3. 执行推理
    global_context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return output_tensors


def load_engine(engine_file):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


class FileLogger(trt.ILogger):
    def __init__(self, severity=trt.Logger.INFO, log_file="trt_log.txt"):
        super().__init__()
        self.severity = severity
        self.log_file = log_file
        self._file_handle = None

    def __enter__(self):
        self._file_handle = open(self.log_file, "a", buffering=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()

    def log(self, severity, msg):
        if severity > self.severity or self._file_handle.closed:
            return
        s_str = self._get_severity_string(severity)
        full_msg = f"[{s_str}] {msg}"
        # 1. 打印到控制台（错误/警告用 stderr，其他用 stdout）
        if severity >= trt.Logger.ERROR:
            print(full_msg, file=sys.stderr)
        else:
            print(full_msg, file=sys.stdout)
        self._file_handle.write(full_msg + "\n")

    def _get_severity_string(self, severity):
        return {
            trt.Logger.INTERNAL_ERROR: "INTERNAL_ERROR",
            trt.Logger.ERROR: "ERROR",
            trt.Logger.WARNING: "WARNING",
            trt.Logger.INFO: "INFO",
            trt.Logger.VERBOSE: "VERBOSE"
        }.get(severity, "UNKNOWN")


def mark_dinov2_precision_v2(network):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name
        
        # 匹配 norm3 
        if (
            "/rgb_encoder/norm_3/ReduceMean" in name or 
            "/rgb_encoder/norm_3/Sub" in name or 
            "/rgb_encoder/norm_3/Mul" in name or 
            "/rgb_encoder/norm_3/Add_1" in name
        ):
            print(f"Setting {name} to fp32")
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
    breakpoint()


def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    fp16_mode: bool = False,
    max_workspace_size: int = 20 << 30,  # 20 GB
    custom_precision_func = None,
):
    """
    从 ONNX 构建 TensorRT 引擎（支持动态 shape + FP16）
    
    Args:
        onnx_file_path: ONNX 模型路径
        engine_file_path: 输出引擎路径
        fp16_mode: 是否启用 FP16
        max_workspace_size: Workspace 大小（bytes）
    """
    print(f"Start Building Engine from {onnx_file_path} to {engine_file_path}...")
    # 日志设置（VERBOSE 可帮助调试）
    # logger = trt.Logger(trt.Logger.VERBOSE)  # 或 trt.Logger.VERBOSE
    
    # 创建构建器
    with FileLogger(trt.Logger.VERBOSE, f"{engine_file_path[:-7]}_trt.log") as logger:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        
        # 设置 workspace
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        # 解析 ONNX
        parser = trt.OnnxParser(network, logger)
        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.log(trt.Logger.ERROR, parser.get_error(i).desc())
                raise RuntimeError("Failed to parse ONNX file")

        # 设置自定义精度
        if custom_precision_func is not None:
            custom_precision_func(network)

        # 启用 FP16（如果支持）
        if fp16_mode:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)  # 安全模式
                # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                logger.log(trt.Logger.INFO, "FP16 enabled")
            else:
                logger.log(trt.Logger.WARNING, "FP16 not supported on this platform")
                fp16_mode = False
        
        # 构建序列化引擎
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # 保存引擎
        with open(engine_file_path, "wb") as f:
            f.write(engine_bytes)
        
        print(f"✅ Engine saved to: {engine_file_path}")
        print(f"   FP16: {'Enabled' if fp16_mode else 'Disabled'}")
        print(f"Save engine to {engine_file_path}")
    return engine_file_path


@torch.no_grad()
def debug_dino_tokens(features, h, w, patch_start=0):
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    patch_size = 14
    
    out = []
    patch_h, patch_w = h // patch_size, w // patch_size
    for i, x in enumerate(features):
        x = x[:, patch_start:, ...]
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()
        out.append(x)
    
    batch_size = out[0].shape[0]
    if batch_size > 1:
        print(f"Warning: batch size > 1 ({batch_size}). Only visualizing the first sample in batch.")
    # 选择 batch 中的第一个样本
    feature_maps = [feat[0] for feat in out]  # List of [C, Hi, Wi]
    # 上采样到原始图像大小
    upsampled = []
    for feat in feature_maps:
        C, Hi, Wi = feat.shape

        # 展平为空间向量: [C, Hi*Wi] -> [Hi*Wi, C]
        feat_flat = feat.permute(1, 2, 0).reshape(-1, C).cpu().numpy()  # [N, C]

        # 执行 PCA，保留前3个主成分
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat_flat)  # [N, 3]

        # 归一化到 [0, 1] 以便转为图像
        feat_pca -= feat_pca.min(axis=0)
        feat_pca /= (feat_pca.max(axis=0) + 1e-8)
        feat_pca = np.clip(feat_pca, 0, 1)

        # 重塑回空间结构 [Hi, Wi, 3]
        rgb = feat_pca.reshape(Hi, Wi, 3)

        # 上采样到原始图像大小 (H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, Hi, Wi]
        rgb_up = torch.nn.functional.interpolate(
            rgb_tensor,
            size=(h, w),
            mode='nearest'
        ).squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]

        upsampled.append(rgb_up)

    # 转换为 numpy
    maps_np = upsampled
    # 绘制子图
    n = len(maps_np)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n):
        im = axes[i].imshow(maps_np[i], cmap='jet', aspect='auto')
        axes[i].set_title(f'Feature Map {i+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_aspect('equal', adjustable='box')

    # 隐藏多余的子图
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path = f'./debug/dinov2_features/debug.jpg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')



class TrtModelBase():
    def __init__(self, engine, onnx, model, use_fp16=True):
        self.engine = None
        model_name = os.path.basename(onnx)[:-5]
        if engine is not None and os.path.exists(engine):
            self.engine = load_engine(engine)
            if self.engine is None:
                print(f"Loading Engine From: {engine} Failed. Try to rebuild")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                engine = os.path.dirname(onnx) + f"/{model_name}_{timestamp}.engine"
            else:
                print(f"Loading Engine From: {engine}")
        
        if self.engine is None:
            assert onnx is not None and os.path.exists(onnx), "Onnx file or Engine File Must be Provided"
            if engine is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                engine = os.path.dirname(onnx) + f"/{model_name}_{timestamp}.engine"
            build_engine(onnx, engine, fp16_mode=use_fp16)
            self.engine = load_engine(engine)
        
        assert self.engine is not None, "Error Building Engine"
        
        self.global_context = self.engine.create_execution_context()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.stream = torch.cuda.Stream()
        self.infer_func = partial(
            infer_trt_with_torch,
            engine=self.engine,
            global_context=self.global_context,
            stream=self.stream,
        )
        print(f"Trt Inference Model init from {engine}")

    def __call__(self):
        raise NotImplementedError()

