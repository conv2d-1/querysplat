import argparse
import os
import time
import cv2
import numpy as np
from PIL import Image
from transparent_background import Remover

def parse_args():
    parser = argparse.ArgumentParser(description="批量分割图片并保存结果")
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./output_images', help='输出图片文件夹路径')
    parser.add_argument('--mode', type=str, default='fast', choices=['fast', 'base-nightly'], help='Remover模型模式')
    parser.add_argument('--jit', action='store_true', help='是否使用jit')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    parser.add_argument('--ckpt', type=str, default=None, help='模型权重路径')
    parser.add_argument('--img_exts', type=str, default='.jpg,.jpeg,.png,.bmp,.tiff', help='支持的图片后缀, 逗号分隔')
    parser.add_argument('--type', type=str, default='map', help='remover.process的type参数')
    return parser.parse_args()

def main():
    args = parse_args()

    # 处理图片后缀
    img_exts = [ext.strip().lower() for ext in args.img_exts.split(',') if ext.strip()]

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    remover_kwargs = {
        'mode': args.mode,
        'jit': args.jit,
        'device': args.device
    }
    if args.ckpt:
        remover_kwargs['ckpt'] = args.ckpt
    remover = Remover(**remover_kwargs)

    for fname in os.listdir(args.input_dir):
        if any(fname.lower().endswith(ext) for ext in img_exts):
            img_path = os.path.join(args.input_dir, fname)
            img = Image.open(img_path).convert('RGB')
            out = remover.process(img, type=args.type)
            # 将输出转换为二值uint8 mask
            out_np = np.array(out)
            # 如果输出有alpha通道，直接用alpha，否则转为灰度
            if out_np.shape[-1] == 4:
                mask = out_np[..., 3]
            else:
                mask = cv2.cvtColor(out_np, cv2.COLOR_RGB2GRAY)
            # 二值化
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask_bin = mask_bin.astype(np.uint8)
            # 保存为单通道PNG
            save_path = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}.png")
            cv2.imwrite(save_path, mask_bin)


if __name__ == '__main__':
    main()
