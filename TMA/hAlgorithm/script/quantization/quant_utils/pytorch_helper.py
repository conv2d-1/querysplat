import os, sys
sys.path.append(os.getcwd())

from itertools import islice
from tqdm import tqdm

import cv2
import torch
import pandas as pd

import pytorch_quantization
from pytorch_quantization import nn as quant_nn

from hAlgorithm.utils.util import eval_dict_to_text
from hAlgorithm.modules.pipelines.outputs import DepthOutput
from hAlgorithm.script.quantization.quant_utils.data_helper import (
    depth_to_point,
    filter_pointmap,
    summary_eval,
)

def get_layer(model, path):
    parts = path.split('/')
    current = model
    for part in parts:
        if part != "":
            sub_parts = part.split('.')
            if len(sub_parts)==1:
                current = getattr(current, sub_parts[0])
            else:
                current = getattr(current, sub_parts[0])[int(sub_parts[1])]
    return current

def get_layers(model, layers_list):
    non_quant_parts = []
    for layer_path in layers_list:
        layer = get_layer(model, layer_path)
        non_quant_parts.append(layer)
    return non_quant_parts

def model_eval2(model, dataloader, data_parser, eval_metrics, cfg):
    if 'torch_eval' in cfg:
        sample_num = cfg['torch_eval'].get('sample_num', -1)
        save_dir = cfg['torch_eval'].get('save_dir', None)
        save_ds = cfg['torch_eval'].get('save_ds', 1)
    else:
        sample_num = -1
        save_dir = None
        save_ds = 1

    if sample_num < 0:
        sample_num = len(dataloader)

    print("------------------------- Start Inference -----------------------------")
    frame_eval_results = []
    for i, batch in enumerate(islice(tqdm(dataloader), sample_num)):
        data = data_parser.parse_data(batch)
        image = data['image'].cuda()
        prompt_depth = data['prompt_depth'].cuda()
        prompt_scale = data['prompt_scale'].cuda()

        intrinsic = data['intrinsic']

        pointmap_pred, confidence = model(image, prompt_depth, prompt_scale)
        confidence = confidence[0, 0].cpu().numpy()

        if pointmap_pred.shape[-3]==1:
            pointmap_pred = depth_to_point(pointmap_pred, intrinsic, pointmap_pred.device)
        
        # Convert the predicted pointmap to a NumPy array and extract the depth channel.
        pointmap_pred = pointmap_pred.cpu().numpy().reshape(3, -1).transpose(1, 0)
        depth = (
            pointmap_pred[:, 2]
            .reshape((image.shape[-2], image.shape[-1])).clip(1e-3)
        )

        filtered_pointmap, filtered_pointmap_color = filter_pointmap(pointmap_pred, confidence, data['image_show'], cfg['model']['output_conf_thresh'])

        # If matching input resolution is required, resize the predicted pointmap and color pointmap to match the ground truth depth.
        if cfg['model']['match_input_res']:
            depth_gt = data['depth_gt'].squeeze().numpy()
            h, w = depth_gt.shape
            depth = cv2.resize(
                depth,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            confidence = cv2.resize(
                confidence,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        # Optionally convert the ground truth pointmap to a NumPy array.
        pointmap_gt = data['pointmap_gt']
        if pointmap_gt is not None:
            pointmap_gt = (
                pointmap_gt.cpu()
                .float()
                .squeeze(0)
                .numpy()
                .reshape(3, -1)
                .transpose(1, 0)
            )

        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        pointmap_color = (
            data['orig_image'].clone().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
        )
        pointmap_color = (pointmap_color + 1) * 0.5 * 255

        output = DepthOutput(
            depth_align=depth,
            pointmap=pointmap_pred,
            pointmap_gt=pointmap_gt,
            pointmap_color=pointmap_color,
            intrinsic=intrinsic,
            pointmap_h=image.shape[-2],
            pointmap_w=image.shape[-1],
            confidence=confidence,
            filtered_pointmap=filtered_pointmap,
            filtered_pointmap_color=filtered_pointmap_color,
        )

        if eval_metrics is not None:
            frame_eval_results.append(eval_metrics(batch, output))

    return frame_eval_results


def model_eval(model, dataloader, eval_metrics, cfg, save_dir=None, save_ds=50):
    frame_eval_results = []
    for i, batch in enumerate(tqdm(dataloader)):
        # 准备输入数据
        image = batch["image"].clone().cuda()
        # image = ((image + 1) * 0.5 - model._mean) / model._std

        intrinsic = batch.get("intrinsics", None)  # [n, 3, 3]

        prompt_depth = batch[model.prompt_name].cuda()
        prompt_scale = batch[model.prompt_scale_name][:, None, None, None].cuda()

        pointmap_pred, confidence = model(image, prompt_depth, prompt_scale)
        confidence = confidence[0, 0].cpu().numpy()
        
        if pointmap_pred.shape[-3]==1:
            pointmap_pred = depth_to_point(pointmap_pred, intrinsic, pointmap_pred.device)

        # Convert the predicted pointmap to a NumPy array and extract the depth channel.
        pointmap_pred = pointmap_pred.cpu().numpy().reshape(3, -1).transpose(1, 0)
        depth = (
            pointmap_pred[:, 2]
            .reshape((image.shape[-2], image.shape[-1]))
            .clip(1e-3)
        )

        filtered_pointmap, filtered_pointmap_color = filter_pointmap(pointmap_pred, confidence, batch["image_show"], cfg["model"]["output_conf_thresh"])

        # If matching input resolution is required, resize the predicted pointmap and color pointmap to match the ground truth depth.
        if model.match_input_res:
            depth_gt = batch[model.align_name].squeeze().numpy()
            h, w = depth_gt.shape
            depth = cv2.resize(
                depth,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            confidence = cv2.resize(
                confidence,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )

            if model.post_align:
                depth_gt_valid_mask = batch[model.align_mask_name].squeeze().numpy()
                depth = align_depth_least_square(
                    gt_arr=depth_gt,
                    pred_arr=depth,
                    valid_mask_arr=depth_gt_valid_mask,
                    return_scale_shift=False,
                    max_resolution=None,
                )

        # Optionally convert the ground truth pointmap to a NumPy array.
        pointmap_gt = batch.get(model.target_name, None)
        if pointmap_gt is not None:
            pointmap_gt = (
                pointmap_gt.cpu()
                .float()
                .squeeze(0)
                .numpy()
                .reshape(3, -1)
                .transpose(1, 0)
            )

        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        pointmap_color = (
            batch["image"].clone().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
        )
        pointmap_color = (pointmap_color + 1) * 0.5 * 255

        output = DepthOutput(
            depth_align=depth,
            pointmap=pointmap_pred,
            pointmap_gt=pointmap_gt,
            pointmap_color=pointmap_color,
            intrinsic=intrinsic,
            pointmap_h=image.shape[-2],
            pointmap_w=image.shape[-1],
            confidence=confidence,
            filtered_pointmap=filtered_pointmap,
            filtered_pointmap_color=filtered_pointmap_color,
        )

        if save_dir is not None and i%save_ds==0:
            model.visualize(output, batch["meta_data"], out_dir=save_dir)

        if eval_metrics is not None:
            frame_eval_results.append(eval_metrics(batch, output))
    return frame_eval_results

def model_eval_v1_3(model, data_loader, data_parser, postprocessor, eval_metrics=None, eval_cfg=None):
    if eval_cfg is None or 'data_num' not in eval_cfg:
        data_num = len(data_loader)
    else:
        data_num = eval_cfg['data_num'] if eval_cfg['data_num']>0 else len(data_loader)
    
    frame_eval_results = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(data_loader)):
            input_data = data_parser.get_inputs(batch)
            
            image = input_data['image'].cuda()
            prompt_depth_norm = input_data['prompt_depth_norm'].cuda()

            results = model(image, prompt_depth_norm, meta_data=batch["meta_data"])

            pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]
            gradient_pred = results.get("gradient", None)
            prompt_confidence_pred = results.get("prompt_confidence", None)

            # breakpoint()
            infer_results = {
                'pointmap_pred':pointmap_pred,
                'confidence_pred':confidence_pred,
                'gradient_pred':gradient_pred,
                'prompt_confidence_pred':prompt_confidence_pred,
            }
            
            if input_data['image'] is not None: 
                input_data['image'] = input_data['image'].to(device='cuda', dtype=torch.float32)
            if input_data['target'] is not None: 
                input_data['target'] = input_data['target'].to(device='cuda')
            if input_data['intrinsics'] is not None: 
                input_data['intrinsics'] = input_data['intrinsics'].to(device='cuda')
            if input_data['extrinsics'] is not None: 
                input_data['extrinsics'] = input_data['extrinsics'].to(device='cuda')
            if input_data['prompt_depth'] is not None: 
                input_data['prompt_depth'] = input_data['prompt_depth'].to(device='cuda', dtype=torch.float32)
            if input_data['prompt_scale'] is not None: 
                input_data['prompt_scale'] = input_data['prompt_scale'].to(device='cuda', dtype=torch.float32)
            if input_data['prompt_center'] is not None: 
                input_data['prompt_center'] = input_data['prompt_center'].to(device='cuda', dtype=torch.float32)

            post_results = postprocessor.postprocess(batch, input_data, infer_results)

            frame_eval_results.append(eval_metrics(batch, post_results))
    return frame_eval_results

class TorchModel:
    def __init__(self, model, half=False):
        self.half = half
        self.model = model
        if self.half:
            self.model = self.model.half()

    @torch.no_grad()
    def __call__(self, input_data, meta_data=None):
        image = input_data['image'].cuda()
        prompt_depth_norm = input_data['prompt_depth_norm'].cuda()

        if self.half:
            image = image.half()
            prompt_depth_norm = prompt_depth_norm.half()

        results = self.model(image, prompt_depth_norm, meta_data=meta_data)

        return results

def export_onnx(pipline, dataloader, data_parser, onnx_path, forward_func=None):
    from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers.attention import (
        Attention,
    )
    for block in pipline.model.rgb_encoder.dinov2.blocks:
        block.attn.__class__ = Attention

    pipline.model.rgb_encoder.dinov2.dinov2_attention_with_sdpa = False

    if forward_func is not None:
        pipline.__class__.forward = forward_func

    for i, batch in enumerate(dataloader):
        input_data = data_parser.get_inputs(batch)
        input_dict = {
            "image": input_data['image'].cuda(),
            "prompt_depth": input_data['prompt_depth'].cuda()[:,-1:,:,:], #[:,-1:,:,:]
            "prompt_scale": input_data['prompt_scale'].cuda(),
        }
        break
    input_names = []
    dummy_input = []
    for key, value in input_dict.items():
        input_names.append(key)
        dummy_input.append(value)
    dummy_input = tuple(dummy_input)

    output_names = ["depth", "confidence"]
    torch.onnx.export(pipline, dummy_input, onnx_path, 
                    input_names=input_names, output_names=output_names,
                    verbose=False, opset_version=13)

class TorchLayerAnalyser:
    def __init__(
        self, 
        model_wrapper, 
        dataloaders, 
        data_parser, 
        postprocessor, 
        eval_metrics, 
        evaluator, 
        analyse_cfg
    ):
        self.model_wrapper = model_wrapper
        self.dataloaders = dataloaders
        self.data_parser = data_parser
        self.postprocessor = postprocessor
        self.eval_metrics = eval_metrics
        self.evaluator = evaluator

        self.analyse_cfg = analyse_cfg
        self.output_dir = self.analyse_cfg['output_dir']

        self.all_results = {'multi_dataset':[]}

    def evaluate(self, eval_name):
        multi_dataset_frame_results = []
        for dataloader in self.dataloaders:
            dataset_name = (
                dataloader.dataset.name
                if hasattr(dataloader.dataset, "name")
                else "unnamed"
            )

            if dataset_name not in self.all_results:
                self.all_results[dataset_name] = []

            frame_eval_results = self.evaluator.eval(
                self.model_wrapper, 
                dataloader, 
                self.data_parser, 
                self.postprocessor, 
                self.eval_metrics
            )
            multi_dataset_frame_results += frame_eval_results

            dataset_results = summary_eval(frame_eval_results, self.eval_metrics)
            dataset_results['config'] = eval_name
            self.all_results[dataset_name].append(dataset_results)
        multi_dataset_results = summary_eval(multi_dataset_frame_results, self.eval_metrics)
        eval_text = eval_dict_to_text(val_metrics=multi_dataset_results, dataset_name='multi_dataset')
        print(f"Evaluation results on multi_dataset {eval_name} metrics: {eval_text}")
        multi_dataset_results['config'] = eval_name
        self.all_results['multi_dataset'].append(multi_dataset_results)

        for dataset_name, single_records in self.all_results.items():
            os.makedirs(f"{self.output_dir}/{dataset_name}", exist_ok=True) 
            df = pd.DataFrame(single_records)
            df.to_csv(f"{self.output_dir}/{dataset_name}/layer_analyse.csv", index=False)

    def analyse_layers(self):
        # disable quant layers
        for name, module in self.model_wrapper.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                disable_quant = False
                for to_search_module in self.analyse_cfg['to_search_modules']:
                    if to_search_module in name:
                        disable_quant = True
                for to_search_module in self.analyse_cfg['fix_quant_layers']:
                    if to_search_module in name:
                        disable_quant = False
                for to_search_module in self.analyse_cfg['non_quant_layers']:
                    if to_search_module in name:
                        disable_quant = True
                if disable_quant:
                    module.disable()
                    print(f"disable {name}")
        
        self.evaluate('start')

        for name, module in self.model_wrapper.model.named_modules():
            if not isinstance(module, quant_nn.TensorQuantizer):
                continue

            is_search_layer = False
            for to_search_module in self.analyse_cfg['to_search_modules']:
                if to_search_module in name:
                    is_search_layer = True

            if is_search_layer:
                module.enable()
                print(f"enable {name}")

                self.evaluate('start')

                module.disable()