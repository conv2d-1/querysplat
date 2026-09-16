import datetime
import json
import logging
import os
import time
import traceback
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from hAlgorithm.modules.utils.optimizer import build_optimizer_with_cfg
from hAlgorithm.utils import (
    cuda_timing_context,
    instantiate_from_config,
    profiled_context,
)
from hAlgorithm.utils.tb_logger import tb_logger

from .base_trainer import BaseTrainer


class MogeTrainer(BaseTrainer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def register_schedule(self):
        """Registers the learning rate schedule or any other training schedules."""
        # Optimizer !should be defined after input layer is adapted
        if self.optimizer is not None:
            self.optimizer = build_optimizer_with_cfg(self.optimizer, self.model)

        # LR scheduler
        if self.lr_scheduler is not None:
            lr_func = instantiate_from_config(self.lr_scheduler)
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)


class MogeIterTrainer(MogeTrainer):
    def __init__(
        self,
        skip_error_step=False,
        logging_step=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.skip_error_step = skip_error_step
        self.logging_step = logging_step

    def train(self):
        """Main training loop that runs through the dataset for the specified number of epochs or iterations."""

        self.total_iter = self.start_iter

        # Visualize before continuing training (optional)
        if self.in_visualize:
            if self.accelerator.is_main_process:
                logging.info("Visualizing before continuing training.")
            self.visualize()

        # Perform evaluation before starting the training (optional)
        if self.in_evaluation:
            if self.accelerator.is_main_process:
                logging.info("Evaluating before continuing training.")
            self.validate(save_best=False, save_tb=True)

        if self.accelerator.is_main_process:
            logging.info("------ Start training ------")
            logging.info(f"output_dir: {self.output_dir}")

        grad_norm = None  # To store the gradient norm for monitoring
        accumulated_step = 0  # Count of gradient accumulation steps
        accumulated_loss = defaultdict(float)  # Store accumulated losses during training

        # Progress bar for tracking training iterations
        progress_bar = tqdm(
            range(0, self.max_iter),
            initial=self.start_iter,
            desc="steps",
            disable=not self.accelerator.is_local_main_process,
        )

        # Initialize training variables
        self.start_time = time.time()
        self.cur_time = self.start_time
        self.callback_total_time = 0
        self.epoch = 0
        logging.debug(f"[MogeIterTrainer] start_iter {self.start_iter}")

        steps = (self.max_iter - self.start_iter + 1) * self.gradient_accumulation_steps
        train_dataloader_iter = iter(self.train_dataloader)

        # Turn on Torch profiler if self.enable_profile is True
        with profiled_context(steps, enable=self.enable_profile) as total_steps:
            for bi in total_steps:
                with cuda_timing_context("train_dataloader_iter", self.timing):
                    batch = next(train_dataloader_iter)  # Get the next batch

                # Adjust batch size dynamically if needed for distributed training
                if self.accelerator_dynamic_batch and self.accelerator.deepspeed_plugin is not None:
                    self.accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = batch[self.accelerator_dynamic_batch_key].shape[0]

                # Add the current iteration to the batch dictionary
                batch["total_iter"] = self.total_iter

                # Perform forward pass and compute loss
                with cuda_timing_context("model_train_step", self.timing):
                    if self.skip_error_step:
                        try:
                            loss, loss_dict = self.model.train_step(batch)
                        except Exception as e:
                            torch.cuda.empty_cache()
                            traceback.print_exc()
                            logging.error(e)
                            self.total_iter += 1
                            continue
                    else:
                        loss, loss_dict = self.model.train_step(batch)

                loss = loss / self.gradient_accumulation_steps  # Normalize loss for gradient accumulation

                # Accumulate gradients and losses
                accumulated_step += 1
                accumulated_loss["loss"] += loss.item()
                if loss_dict is not None:
                    for key, val in loss_dict.items():
                        accumulated_loss[key] += (val / self.gradient_accumulation_steps)

                # Perform backward pass to compute gradients
                with cuda_timing_context("backward", self.timing):
                    self.accelerator.backward(loss)  # Perform backward pass with gradient clipping

                # Update model weights when enough gradients have been accumulated
                if accumulated_step >= self.gradient_accumulation_steps:
                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.get_train_parameters(), self.max_grad_norm)
                        if grad_norm is not None and (torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item()):
                            logging.warning(f"{batch['meta_data']['name'][0]}, Grad is NaN or Inf, setting grad to zero.")
                            if isinstance(batch["meta_data"]["data_info"][0], dict):
                                if "scene" in batch["meta_data"]["data_info"][0]:
                                    logging.warning(f"{batch['meta_data']['data_info'][0]['scene']}, {batch['meta_data']['data_info'][0]['rgb']}.")
                                else:
                                    logging.warning(f"{batch['meta_data']['data_info'][0]['rgb']}.")
                            else:
                                logging.warning(f"{batch['meta_data']['data_info'][0][0]['scene']}, {[data_info['rgb'] for data_info in batch['meta_data']['data_info'][0]]}.")
                            self.optimizer.zero_grad()
                        elif grad_norm is not None and self.skip_grad_norm is not None and grad_norm > self.skip_grad_norm:
                            logging.warning(f"{batch['meta_data']['name'][0]}, Grad > {self.skip_grad_norm}, set grad zero")
                            if isinstance(batch["meta_data"]["data_info"][0], dict):
                                if "scene" in batch["meta_data"]["data_info"][0]:
                                    logging.warning(f"{batch['meta_data']['data_info'][0]['scene']}, {batch['meta_data']['data_info'][0]['rgb']}.")
                                else:
                                    logging.warning(f"{batch['meta_data']['data_info'][0]['rgb']}.")
                            else:
                                logging.warning(f"{batch['meta_data']['data_info'][0][0]['scene']}, {[data_info['rgb'] for data_info in batch['meta_data']['data_info'][0]]}.")
                            self.optimizer.zero_grad()

                    last_lr = self.lr_scheduler.get_last_lr()
                    self.optimizer.step()
                    self.lr_scheduler.step(self.total_iter)  # Update learning rate
                    self.optimizer.zero_grad()  # Reset gradients after weight update

                    # Log metrics and training information
                    if self.accelerator.is_main_process:
                        if (self.logging_step == 1) or (self.total_iter % self.logging_step == 0):
                            for i, lr in enumerate(last_lr):
                                self.tb_logger.writer.add_scalar(f"lr/lr{i}", lr, global_step=self.total_iter)
                            for key, val in accumulated_loss.items():
                                self.tb_logger.writer.add_scalar(f"train/{key}", val, global_step=self.total_iter)
                                if self.tb_dataset_split:
                                    self.tb_logger.writer.add_scalar(
                                        f"data_{key}/{batch['meta_data']['name'][0]}",
                                        val,
                                        global_step=self.total_iter,
                                    )

                            if grad_norm is not None:
                                self.tb_logger.writer.add_scalar(
                                    "train/grad_norm",
                                    grad_norm.item(),
                                    global_step=self.total_iter,
                                )
                                if self.tb_dataset_split:
                                    self.tb_logger.writer.add_scalar(
                                        f"data_grad_norm/{batch['meta_data']['name'][0]}",
                                        grad_norm.item(),
                                        global_step=self.total_iter,
                                    )

                            # Log iteration information
                            logging.info("")
                            text = f"iter{self.total_iter:d}"
                            for key, val in accumulated_loss.items():
                                if key in ["bs", "view"]:
                                    text += f", {key}:{int(val)}"
                                else:
                                    text += f", {key}:{val:.4f}"
                            text += ", "
                            text += ", ".join([f"lr{i}:{lr:.3e}" for i, lr in enumerate(last_lr[:1])])
                            if grad_norm is not None:
                                text += f", grad_norm:{grad_norm.item():.4f}"

                            text += f", mem:{self.get_max_memory()}M, data:{batch['meta_data']['name'][0]}"
                            logging.info(text)

                            cur_time = time.time()
                            iter_time = (cur_time - self.cur_time) / self.logging_step
                            self.cur_time = cur_time
                            total_time = cur_time - self.start_time
                            mean_time = total_time / (self.total_iter - self.start_iter + 1)
                            eta_time = mean_time * (self.max_iter - self.total_iter)
                            total_time = str(datetime.timedelta(seconds=int(total_time)))
                            eta_time = str(datetime.timedelta(seconds=int(eta_time)))
                            logging.info(f"iter:{iter_time:.2f}, mean:{mean_time:.2f}, total:{total_time}, eta:{eta_time}")

                        if self.total_iter % 100 == 0:
                            logging.info(f"output_dir: {self.output_dir}")

                        # Update the progress bar
                        progress_bar.update(1)
                        progress_bar_logs = {
                            "epoch": self.epoch,
                            "loss": accumulated_loss["loss"],
                        }
                        progress_bar.set_postfix(**progress_bar_logs)

                    # Call callback function after each training step
                    callback_flag = self.train_step_callback()

                    if self.accelerator.is_main_process:
                        if (self.logging_step == 1) or (self.total_iter % self.logging_step == 0):
                            # Mark time info
                            self.tb_logger.writer.add_scalar(
                                f"time/iter_time",
                                iter_time,
                                global_step=self.total_iter,
                            )
                            self.tb_logger.writer.add_scalar(
                                f"time/total_time",
                                (cur_time - self.start_time) / 3600.0,
                                global_step=self.total_iter,
                            )
                            callback_time = time.time() - self.cur_time
                            if callback_flag:
                                self.tb_logger.writer.add_scalar(
                                    f"time/callback_time",
                                    callback_time,
                                    global_step=self.total_iter,
                                )
                                self.callback_total_time += callback_time
                                self.tb_logger.writer.add_scalar(
                                    f"time/callback_total_time",
                                    self.callback_total_time / 3600.0,
                                    global_step=self.total_iter,
                                )

                    accumulated_step = 0
                    accumulated_loss = defaultdict(float)

                    # Check if maximum iterations have been reached
                    if 0 < self.max_iter <= self.total_iter:
                        break

                    self.total_iter += 1

        logging.info(f"{self.accelerator.device}, Training Ended.")

        # Final callback after training ends
        self.train_step_callback(is_last=True)

        # Wait for all processes to complete
        self.accelerator.wait_for_everyone()

        # End the training process
        self.accelerator.end_training()

    def train_step_callback(self, is_last=False):
        try:
            return super().train_step_callback(is_last=is_last)
        except Exception as e:
            traceback.print_exc()
            logging.error(e)
            return False


class MogeClipTestTrainer(MogeIterTrainer):
    def __init__(
        self,
        clip_result_dir="clip_test",
        clip_result_filename="clip_metrics.jsonl",
        clip_summary_filename="summary.json",
        clip_save_outputs=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_result_dir = clip_result_dir
        self.clip_result_filename = clip_result_filename
        self.clip_summary_filename = clip_summary_filename
        self.clip_save_outputs = clip_save_outputs

    @staticmethod
    def _safe_to_python(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return [MogeClipTestTrainer._safe_to_python(v) for v in value]
        if isinstance(value, dict):
            return {k: MogeClipTestTrainer._safe_to_python(v) for k, v in value.items()}
        return value

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _parse_clip_name(self, batch, dataset_name, batch_idx):
        meta_data = batch.get("meta_data", {})
        data_idx = meta_data.get("data_idx", [batch_idx])
        if isinstance(data_idx, (list, tuple)) and len(data_idx) > 0:
            data_idx = data_idx[0]
        try:
            data_idx = int(data_idx)
        except Exception:
            data_idx = int(batch_idx)

        scene_name = "scene"
        data_info = meta_data.get("data_info", None)
        if isinstance(data_info, list) and len(data_info) > 0:
            first_info = data_info[0]
            if isinstance(first_info, dict):
                scene_name = first_info.get("scene", scene_name)
            elif isinstance(first_info, list) and len(first_info) > 0 and isinstance(first_info[0], dict):
                scene_name = first_info[0].get("scene", scene_name)
        elif isinstance(data_info, dict):
            scene_name = data_info.get("scene", scene_name)

        clip_name = f"{self._sanitize_name(scene_name)}_{data_idx:06d}"
        clip_meta = {
            "dataset": dataset_name,
            "scene": scene_name,
            "data_idx": data_idx,
            "clip_name": clip_name,
        }
        return clip_name, clip_meta

    def validate(self, save_best=False, save_tb=False, vis=False, save_outputs=False):
        for data_loader in self.val_dataloaders:
            _, summary_metrics, _ = self.validate_single_dataset(
                data_loader=data_loader,
                eval_metrics=self.eval_metrics,
                vis=vis,
                save=save_outputs,
            )
            dataset_name = getattr(data_loader.dataset, "name", "unnamed")
            if save_tb and summary_metrics is not None and len(summary_metrics) > 0:
                self.tb_logger.log_dic(
                    {f"val_{dataset_name}/{k}": v for k, v in summary_metrics.items()},
                    global_step=self.total_iter,
                )

    def validate_single_dataset(
        self,
        data_loader,
        eval_metrics=None,
        vis=False,
        save=False,
        return_frame_output=False,
        app=False,
    ):
        dataset_name = getattr(data_loader.dataset, "name", "unnamed")
        ckpt_name = self.get_ckpt_name()

        result_root = os.path.join(
            self.output_dir,
            self.clip_result_dir,
            ckpt_name,
            self._sanitize_name(dataset_name),
        )
        save_root = os.path.join(result_root, "outputs")

        if self.accelerator is not None and self.accelerator.is_main_process:
            os.makedirs(result_root, exist_ok=True)
            if save or self.clip_save_outputs:
                os.makedirs(save_root, exist_ok=True)
        elif self.accelerator is None:
            os.makedirs(result_root, exist_ok=True)
            if save or self.clip_save_outputs:
                os.makedirs(save_root, exist_ok=True)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        local_clip_results = []
        dist_ready = (
            self.accelerator is not None
            and self.dist_test
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        local_rank = 0 if self.accelerator is None else int(self.accelerator.process_index)
        results_path = os.path.join(result_root, self.clip_result_filename)
        if dist_ready:
            result_name, result_ext = os.path.splitext(self.clip_result_filename)
            local_results_path = os.path.join(
                result_root, f"{result_name}_rank{local_rank:02d}{result_ext or '.jsonl'}"
            )
        else:
            local_results_path = results_path
        with open(local_results_path, "w", encoding="utf-8"):
            pass

        is_local_main_process = (
            True if self.accelerator is None else self.accelerator.is_local_main_process
        )
        for batch_idx, batch in enumerate(
            tqdm(
                data_loader,
                desc=f"Clip test on {dataset_name}",
                disable=not is_local_main_process,
            )
        ):
            with torch.inference_mode():
                if self.accelerator is None:
                    batch["use_amp"] = self.use_amp
                    batch["amp_dtype"] = self.amp_dtype
                output = self.model.infer(**batch)

                if (
                    isinstance(output, list)
                    and isinstance(output[-1], dict)
                    and output[-1].get("type", None) is not None
                ):
                    batch = output.pop()

                if vis:
                    vis_out_dir = os.path.join(self.vis_dir, ckpt_name, dataset_name)
                    if self.accelerator is None or self.accelerator.is_main_process:
                        os.makedirs(vis_out_dir, exist_ok=True)
                    self.model.visualize(
                        output,
                        meta_data=batch["meta_data"],
                        out_dir=vis_out_dir,
                    )

            clip_name, clip_meta = self._parse_clip_name(batch, dataset_name, batch_idx)
            clip_dir = os.path.join(save_root, clip_name)
            if save or self.clip_save_outputs:
                os.makedirs(clip_dir, exist_ok=True)
                self.model.save_output(
                    output,
                    meta_data=batch["meta_data"],
                    out_dir=clip_dir,
                    output_meta_dict={},
                )

            metrics = {}
            if eval_metrics is not None:
                eval_dict = eval_metrics(batch, output)
                metrics = {k: self._safe_to_python(v) for k, v in eval_dict.items()}

            local_clip_results.append(
                {"iter": int(self.total_iter), "metrics": metrics, **clip_meta}
            )
            with open(local_results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(local_clip_results[-1], ensure_ascii=False) + "\n")
                f.flush()

        if dist_ready:
            gathered_results = [None for _ in range(self.accelerator.num_processes)]
            torch.distributed.all_gather_object(gathered_results, local_clip_results)
            merged_results = [item for sub in gathered_results for item in sub]
        else:
            merged_results = local_clip_results

        if self.accelerator is None or self.accelerator.is_main_process:
            if dist_ready:
                with open(results_path, "w", encoding="utf-8") as f:
                    for item in merged_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

            metric_pool = defaultdict(list)
            for item in merged_results:
                for key, value in item["metrics"].items():
                    if isinstance(value, (int, float)):
                        metric_pool[key].append(float(value))
            summary_metrics = {
                key: sum(values) / len(values) for key, values in metric_pool.items() if len(values) > 0
            }
            summary = {
                "dataset": dataset_name,
                "iter": int(self.total_iter),
                "num_clips": len(merged_results),
                "avg_metrics": summary_metrics,
            }
            summary_path = os.path.join(result_root, self.clip_summary_filename)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logging.info(
                f"[MogeClipTestTrainer] dataset={dataset_name}, clips={len(merged_results)}, "
                f"results={results_path}, summary={summary_path}"
            )

            return None, summary_metrics, summary_path

        return None, None, None