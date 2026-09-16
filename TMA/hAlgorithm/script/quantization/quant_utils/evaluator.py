import os, sys
sys.path.append(os.getcwd())

from itertools import islice
from tqdm import tqdm

import torch
import cv2

from hAlgorithm.utils.util import eval_dict_to_text
from hAlgorithm.script.quantization.quant_utils.data_helper import summary_eval

from hAlgorithm.modules.pipelines.visualize import (
    save_depth_map,
    save_error,
    save_point_cloud,
)

class Evaluator:
    def __init__(
        self, 
        sample_num=-1, 
        out_dir=None, 
        vis_step=-1,
        output_conf_thresh = 0,
    ):
        self.sample_num = sample_num
        self.out_dir = out_dir
        self.vis_step = vis_step
        self.output_conf_thresh = output_conf_thresh

    def eval(self, model, dataloader, data_parser, postprocessor, eval_metrics=None):
        dataset_name = (
            dataloader.dataset.name
            if hasattr(dataloader.dataset, "name")
            else "unnamed"
        )

        vis_dir = None
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            if self.vis_step > 0:
                vis_dir = os.path.join(self.out_dir, dataset_name)
                os.makedirs(vis_dir, exist_ok=True)
        
        sample_num = len(dataloader)
        if self.sample_num > 0:
            sample_num = self.sample_num

        frame_eval_results = []
        for i, batch in enumerate(islice(tqdm(dataloader), sample_num)):
            input_data = data_parser.get_inputs(batch)

            results = model(input_data, meta_data=batch['meta_data'])

            pointmap_pred = results['pointmap']
            confidence_pred = results['confidence']

            infer_results = {
                'pointmap_pred':pointmap_pred,
                'confidence_pred':confidence_pred,
                'gradient_pred':None,
                'prompt_confidence_pred':None,
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

            if vis_dir is not None and self.vis_step>0 and (i % self.vis_step)==0:
                self.visualize(post_results, i, vis_dir)

        eval_results = summary_eval(frame_eval_results, eval_metrics)
        eval_text = eval_dict_to_text(val_metrics=eval_results, dataset_name=dataset_name)
        print(f"Evaluation results on {dataset_name}: {eval_text}")

        if self.out_dir is not None:
            eval_text_save_path = os.path.join(self.out_dir, f"eval-{dataset_name}.txt")
            with open(eval_text_save_path, "w+") as f:
                f.write(eval_text)
            print(f"Saved evaluation results to: {eval_text_save_path}")

        return frame_eval_results

    def visualize(self, outputs, data_idx, out_dir, prefix=""):
        # Depth Map Visualization
        if outputs.depth_align is not None:
            save_depth_map(
                outputs.depth_align.copy(), out_dir, f"{prefix}depth", data_idx, info=False
            )

        # Point Cloud Visualization
        if outputs.pointmap is not None:
            save_point_cloud(
                outputs.pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}point",
                data_idx,
                info=False,
            )

         # Ground Truth Point Cloud Visualization
        if outputs.pointmap_gt is not None:
            save_point_cloud(
                outputs.pointmap_gt.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}point_gt",
                data_idx,
            )

            # Error Map Visualization
            pointmap_shape = (outputs.pointmap_h, outputs.pointmap_w)
            pred_depthmap = outputs.pointmap.copy()[:, 2].reshape(pointmap_shape)
            gt_depthmap = outputs.pointmap_gt.copy()[:, 2].reshape(pointmap_shape)
            save_error(pred_depthmap, gt_depthmap, out_dir, f"{prefix}error", data_idx)

        # Confidence Visualization
        if outputs.confidence is not None:
            confidence = outputs.confidence.copy()
            save_depth_map(confidence, out_dir, f"{prefix}conf", data_idx, info=False)
            confidence_mask = (confidence > self.output_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, f"{prefix}conf_mask", data_idx)

        # Filtered Point Cloud Visualization
        if outputs.filtered_pointmap is not None:
            save_point_cloud(
                outputs.filtered_pointmap.copy(),
                (
                    outputs.filtered_pointmap_color.copy()
                    if outputs.filtered_pointmap_color is not None
                    else None
                ),
                out_dir,
                f"{prefix}filtered_point",
                data_idx,
            )

        # Confidence Visualization
        if outputs.input_confidence is not None:
            confidence = outputs.input_confidence.copy()
            confidence_act = 1 / (1 + np.exp(-confidence))
            save_depth_map(
                confidence_act,
                out_dir,
                f"{prefix}prompt_conf",
                data_idx,
                min_val=0,
                max_val=1,
                info=False,
            )
            confidence_mask = (confidence > self.output_prompt_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, f"{prefix}prompt_conf_mask", data_idx)

        if outputs.prompt_pointmap is not None:
            if outputs.prompt_h != outputs.pointmap_h or outputs.prompt_w != outputs.pointmap_w:
                if outputs.pointmap_color is not None:
                    tmp_color = cv2.resize(
                        outputs.pointmap_color.copy().reshape(outputs.pointmap_h, outputs.pointmap_w, -1),
                        dsize=(outputs.prompt_w, outputs.prompt_h),
                        interpolation=cv2.INTER_LINEAR,
                    ).reshape(outputs.prompt_w * outputs.prompt_h, -1)
                else:
                    tmp_color = None

                save_point_cloud(
                    outputs.prompt_pointmap.copy(),
                    tmp_color,
                    out_dir,
                    "prompt_point",
                    data_idx,
                )
            else:
                save_point_cloud(
                    outputs.prompt_pointmap.copy(),
                    (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                    out_dir,
                    f"{prefix}prompt_point",
                    data_idx,
                )

            save_depth_map(
                outputs.prompt_pointmap.copy().reshape(outputs.prompt_h, outputs.prompt_w, 3)[:, :, 2], 
                out_dir, 
                "prompt_point", 
                data_idx, 
                info=False
            )

        if outputs.pointmap_align is not None:
            save_point_cloud(
                outputs.pointmap_align.copy(),
                (
                    outputs.pointmap_color_align.copy()
                    if outputs.pointmap_color_align is not None
                    else None
                ),
                out_dir,
                f"{prefix}point_align",
                data_idx,
            )

        if outputs.filtered_pointmap_align is not None:
            save_point_cloud(
                outputs.filtered_pointmap_align.copy(),
                (
                    outputs.filtered_pointmap_color_align.copy()
                    if outputs.filtered_pointmap_color_align is not None
                    else None
                ),
                out_dir,
                f"{prefix}filtered_point_align",
                data_idx,
            )