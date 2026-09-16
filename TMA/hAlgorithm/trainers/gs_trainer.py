import json
import logging
import os
import time
from collections import defaultdict
import random

import torch
from accelerate import Accelerator
from tqdm import tqdm

from hAlgorithm.utils import (
    cuda_timing_context,
    eval_dict_list_to_text,
    eval_dict_to_text,
    profiled_context,
)

from .iter_trainer import IterTrainer
from hAlgorithm.modules.utils.optimizer import build_optimizer_and_scheduler_with_cfg


class GSTrainer(IterTrainer):
    def __init__(
        self,
        densify_grad_threshold=1.0,  # 调大
        scene_extent=10.0,
        densify_until_iter=500,
        densification_interval=100,
        pruning_interval=100,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.densify_grad_threshold = densify_grad_threshold
        self.scene_extent = scene_extent
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.pruning_interval = pruning_interval
        self.device = torch.device("cuda")

        # NOTE: only train and val
        self.vis_dataloaders = self.val_dataloaders
        self.vis_datasets = self.val_datasets

    def train(self):
        """Main training loop that runs through the dataset for the specified number of epochs or iterations."""

        self.total_iter = self.start_iter

        # Perform evaluation before starting the training (optional)
        if self.in_evaluation or self.in_visualize:
            if self.accelerator.is_main_process:
                logging.info("Evaluating before continuing training.")
            # NOTE: 避免测评和可视化进行两侧 infer
            self.validate(save_best=False, save_tb=True, vis=self.in_visualize)

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
        # Iterate over epochs
        self.epoch = 0
        logging.debug(f"Starting epoch {self.epoch}, batches in epoch {self.n_batch_in_epoch}")

        # Calculate the number of batches in the epoch
        steps = self.max_iter - self.n_batch_in_epoch
        train_dataloader_iter = iter(self.train_dataloader)

        print("total batch are: ", steps)
        print("total epoch are:", steps // len(self.train_dataset) + 1)
        # Turn on Torch profiler if self.enable_profile is True
        with profiled_context(steps, enable=self.enable_profile) as total_steps:
            with cuda_timing_context("total time", self.timing):
                for bi in total_steps:
                    with cuda_timing_context("train_dataloader_iter", self.timing):
                        batch = next(train_dataloader_iter)  # Get the next batch

                    self.n_batch_in_epoch += 1

                    # Add the current iteration to the batch dictionary
                    batch["total_iter"] = self.total_iter

                    # Perform forward pass and compute loss
                    with cuda_timing_context("model_train_step", self.timing):
                        loss, loss_dict, results = self.model.train_step(batch, bi)

                    loss = (
                        loss / self.gradient_accumulation_steps
                    )  # Normalize loss for gradient accumulation
                    # Perform backward pass to compute gradients
                    with cuda_timing_context("backward", self.timing):
                        grad_norm = None
                        if isinstance(self.accelerator, Accelerator):
                            self.accelerator.backward(loss)
                            if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.model.get_train_parameters(), self.max_grad_norm
                                )
                        else:
                            loss.backward()

                            if self.max_grad_norm is not None:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.get_train_parameters(), self.max_grad_norm
                                )

                    with cuda_timing_context("Densification", self.timing):
                        with torch.no_grad():
                            # Densification
                            if bi < self.densify_until_iter:  # 3000
                                gaussians = results.get("gaussians", None)
                                gaussians.optimizer = self.optimizer

                                radii = results.get("radii", None)
                                radii = radii[:, 0]

                                visibility_filter = results.get("visibility_filter", None)
                                visibility_filter = visibility_filter[:, 0]

                                screenspace_points = results.get("screenspace_points", None)

                                self.scene_extent = batch["meta_data"]["radius"].item()

                                if (
                                    bi % 500 == 0
                                    and gaussians.active_sh_degree < gaussians.sh_degree
                                ):
                                    gaussians.active_sh_degree += 1

                                # Keep track of max radii in image-space for pruning
                                gaussians.max_radii2D[visibility_filter] = torch.max(
                                    gaussians.max_radii2D[visibility_filter],
                                    radii[visibility_filter],
                                )  # [1,n]

                                gaussians.add_densification_stats(
                                    screenspace_points, visibility_filter
                                )

                                if bi > 500 and bi % self.densification_interval == 0:
                                    size_threshold = 20 if (bi - 1) > 1000 else None
                                    gaussians.densify_and_prune(
                                        self.densify_grad_threshold,
                                        0.005,
                                        self.scene_extent,
                                        size_threshold,
                                        radii,
                                    )  # self.scene_extent=1.0
                                    logging.info(
                                        f"!!!gaussains nums after densify is: {gaussians._means.shape}"
                                    )

                                # 与PGSR不同的地方是这里没有删除out_observe的点

                                if bi != 1 and (bi - 1) % 1000 == 0:
                                    gaussians.reset_opacity()

                        if grad_norm is not None and (
                            torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item()
                        ):
                            logging.warning(
                                f"{batch['meta_data']['name'][0]}, Grad is NaN or Inf, setting grad to zero."
                            )
                            self.optimizer.zero_grad()
                        elif (
                            grad_norm is not None
                            and self.skip_grad_norm is not None
                            and grad_norm > self.skip_grad_norm
                        ):
                            logging.warning(
                                f"{batch['meta_data']['name'][0]}, Grad > {self.skip_grad_norm}, set grad zero"
                            )
                            self.optimizer.zero_grad()

                    # Accumulate gradients and losses
                    accumulated_step += 1
                    accumulated_loss["loss"] += loss.item()
                    if loss_dict is not None:
                        for key, val in loss_dict.items():
                            accumulated_loss[key] += val

                    # Update model weights when enough gradients have been accumulated
                    if accumulated_step >= self.gradient_accumulation_steps:
                        self.optimizer.step()
                        self.lr_scheduler.step(self.total_iter)  # Update learning rate
                        self.optimizer.zero_grad()  # Reset gradients after weight update

                        # Log metrics and training information
                        if self.accelerator.is_main_process:
                            # Update the progress bar
                            progress_bar.update(1)
                            progress_bar_logs = {
                                "loss": accumulated_loss["loss"],
                            }
                            progress_bar.set_postfix(**progress_bar_logs)

                        # Call callback function after each training step
                        with cuda_timing_context("callback", self.timing):
                            self.train_step_callback()

                        accumulated_step = 0
                        accumulated_loss = defaultdict(float)

                        # Check if maximum iterations have been reached
                        if 0 < self.max_iter <= self.total_iter:
                            break

                        self.total_iter += 1

        logging.info(f"{self.accelerator.device}, Training Ended.")

        if isinstance(self.accelerator, Accelerator):
            # Wait for all processes to complete
            self.accelerator.wait_for_everyone()

            # Final callback after training ends
            self.train_step_callback(is_last=True)

            # End the training process
            self.accelerator.end_training()

        else:
            # Final callback after training ends
            self.train_step_callback(is_last=True)

            # End the training process
            torch.cuda.empty_cache()

    def train_step_callback(self, is_last=False):
        """
        Executes after every iteration. Handles backup, validation, checkpoint saving, and visualization.

        :param is_last: Boolean flag indicating if this is the last iteration of training.
        """
        latest_saved = False  # Flag to indicate if the latest checkpoint was saved

        # Backup checkpoint at regular intervals (without training state)
        if self.accelerator.is_main_process:
            if self.backup_period > 0 and self.total_iter % self.backup_period == 0:
                self.save_checkpoint(ckpt_name=self.get_ckpt_name(), save_train_state=True)

        # Perform validation at regular intervals or at the last iteration
        # Use XOR (^) to ensure only one of the conditions is True, but not both
        if self.val_period is not None and isinstance(self.val_period, (list, tuple)):
            if (self.total_iter in self.val_period) ^ is_last:
                # Perform visualization if needed
                # NOTE: 避免测评和可视化进行两侧 infer
                if (self.vis_period > 0 and self.total_iter % self.vis_period == 0) ^ is_last:
                    self.validate(save_best=False, save_tb=True, vis=True)
                else:
                    self.validate(save_best=False, save_tb=True)
                if self.accelerator.is_main_process:
                    self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                    latest_saved = True
            elif (self.vis_period > 0 and self.total_iter % self.vis_period == 0) ^ is_last:
                self.visualize()
        else:
            if (self.val_period > 0 and self.total_iter % self.val_period == 0) ^ is_last:
                # Perform visualization if needed
                # NOTE: 避免测评和可视化进行两侧 infer
                if (self.vis_period > 0 and self.total_iter % self.vis_period == 0) ^ is_last:
                    self.validate(save_best=False, save_tb=True, vis=True)
                else:
                    self.validate(save_best=False, save_tb=True)
                if self.accelerator.is_main_process:
                    self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                    latest_saved = True
            elif (self.vis_period > 0 and self.total_iter % self.vis_period == 0) ^ is_last:
                self.visualize()

        # Save model checkpoint if needed
        if self.accelerator.is_main_process:
            if (
                self.save_period > 0
                and self.total_iter % self.save_period == 0
                and not latest_saved
            ):
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        return latest_saved

    def validate_single_dataset(
        self, data_loader, eval_metrics=None, vis=False, save=False, return_frame_output=False
    ):
        """
        Validates the model on a single dataset (single GPU/CPU) and returns the evaluation results.
        """
        with torch.inference_mode():
            frame_outputs = []
            eval_results = {}
            frame_eval_results = []
            eval_text_save_path = None
            save_meta_dict = dict()

            dataset_name = getattr(data_loader.dataset, "name", "unnamed")

            batch = [data for data in data_loader]
            output = self.model.infer(batch)

            if self.vis_dir is not None:
                # NOTE: 测评时 debug 的存储路径
                vis_out_dir = os.path.join(self.vis_dir, self.get_ckpt_name(), dataset_name)
                for i in range(len(batch)):
                    batch[i]["meta_data"]["vis_out_dir"] = vis_out_dir

            meta_data = [data["meta_data"] for data in batch]
            scene = meta_data[0]["scene"]

            if return_frame_output:
                frame_outputs.append({"output": output, "meta_data": meta_data})

            if eval_metrics is not None:
                frame_eval_results.append(eval_metrics(batch, output))

            if eval_metrics is not None and len(frame_eval_results) > 0:
                if isinstance(eval_metrics.metrics[0], str):
                    for metric_name in eval_metrics.metrics:
                        if metric_name in frame_eval_results[0]:
                            eval_results[metric_name] = sum(
                                result[metric_name].item() for result in frame_eval_results
                            ) / len(frame_eval_results)
                    eval_text = eval_dict_to_text(
                        val_metrics=eval_results,
                        dataset_name=dataset_name,
                        dataset_num=len(data_loader.dataset),
                    )
                else:
                    multi_eval_results = []
                    for metric_obj in eval_metrics.metrics:
                        eval_result = {}
                        for metric_name in metric_obj.metrics:
                            if metric_name in frame_eval_results[0]:
                                eval_result[metric_name] = sum(
                                    result[metric_name].item() for result in frame_eval_results
                                ) / len(frame_eval_results)
                        multi_eval_results.append(eval_result)
                    eval_text = eval_dict_list_to_text(
                        val_metrics=multi_eval_results,
                        dataset_name=dataset_name,
                        dataset_num=len(data_loader.dataset),
                    )
                    eval_results = {k: v for d in multi_eval_results for k, v in d.items()}

                eval_text = f"Scene: {scene}, Iter: {self.total_iter:d}\n" + eval_text
                eval_text_save_path = os.path.join(
                    self.eval_dir,
                    dataset_name,
                    f"eval-{dataset_name}-iter{self.total_iter:06d}.txt",
                )
                try:
                    os.makedirs(os.path.dirname(eval_text_save_path), exist_ok=True)
                    with open(eval_text_save_path, "w+") as f:
                        f.write(eval_text)
                except Exception:
                    logging.warning("save eval_text failed.")

                eval_text_latest_path = os.path.join(
                    self.eval_dir,
                    f"eval-{dataset_name}-latest.txt",
                )
                os.system(f"cp {eval_text_save_path} {eval_text_latest_path}")

                logging.info(f"Evaluation results on {dataset_name}: {eval_text}")
                logging.info(f"Saved evaluation results to: {eval_text_save_path}")
            else:
                eval_results = None

            if vis:
                vis_out_dir = os.path.join(self.vis_dir, self.get_ckpt_name(), dataset_name)
                os.makedirs(vis_out_dir, exist_ok=True)
                self.model.visualize(output, meta_data=meta_data, out_dir=vis_out_dir)

            if save:
                save_out_dir = os.path.join(
                    self.output_dir, "outputs", self.get_ckpt_name(), dataset_name
                )
                os.makedirs(save_out_dir, exist_ok=True)
                self.model.save_output(
                    output,
                    meta_data=meta_data,
                    out_dir=save_out_dir,
                    output_meta_dict=save_meta_dict,
                )

                if len(save_meta_dict) > 0:
                    new_data_path = os.path.join(save_out_dir, "data_info_with_depth.json")
                    with open(new_data_path, "w") as f:
                        json.dump(save_meta_dict, f, indent=2)
                    logging.info(f"output meta saved to: {new_data_path}")

            return frame_outputs if return_frame_output else None, eval_results, eval_text_save_path


class GSTrainerV2(GSTrainer):
    """GSTrainerV2, camera optimizer."""

    def __init__(
        self,
        logging_step=0,
        cam_optimizer=None,
        test_cam_optimizer_start=1000,
        cam_optimizer_end=None,
        cam_optimizer_iters=10,
        cam_pretrain=None,
        build_mesh_final=False,
        mesh_frame_step=1,
        **kwargs,
    ):

        self.logging_step = logging_step
        self.cam_optimizer = cam_optimizer
        self.test_cam_optimizer_start = test_cam_optimizer_start
        self.cam_optimizer_end = cam_optimizer_end
        self.cam_optimizer_iters = cam_optimizer_iters
        self.cam_pretrain = cam_pretrain
        self.build_mesh_final = build_mesh_final
        self.mesh_frame_step = mesh_frame_step

        model = kwargs["model"]
        train_dataset = kwargs["train_dataset"]
        val_datasets = kwargs["val_datasets"]
        device = torch.device("cuda")

        if self.cam_pretrain is not None:
            # NOTE: 离线测评可加载优化后的 camera.json
            with open(self.cam_pretrain, "r") as f:
                data = json.load(f)
            model.set_cameras(data, device=device)
        else:
            if train_dataset is not None:
                model.set_cameras(train_dataset.datasets[0].gs_total_datas, device=device)
            if val_datasets is not None:
                model.set_cameras(val_datasets[0].gs_total_datas, device=device)

        super().__init__(**kwargs)

    def register_schedule(self):
        """Registers the learning rate schedule or any other training schedules."""
        # Optimizer !should be defined after input layer is adapted
        self.optimizer, self.lr_scheduler = build_optimizer_and_scheduler_with_cfg(
            self.optimizer, self.lr_scheduler, self.model, skip_match_key=True
        )

        if self.cam_optimizer is not None:
            self.cam_optimizer, self.cam_lr_scheduler = build_optimizer_and_scheduler_with_cfg(
                self.cam_optimizer, None, self.model, skip_match_key=True
            )

    def train(self):
        """Main training loop that runs through the dataset for the specified number of epochs or iterations."""

        if self.cam_optimizer is None:
            return super().train()

        # Perform evaluation before starting the training (optional)
        if self.in_evaluation:
            if self.accelerator.is_main_process:
                logging.info("Evaluating before continuing training.")
            # NOTE: 避免测评和可视化进行两侧 infer
            self.validate(save_best=False, save_tb=True, vis=self.in_visualize)

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
        self.total_iter = self.start_iter
        self.start_time = time.time()
        self.cur_time = self.start_time
        self.callback_total_time = 0
        # Iterate over epochs
        self.epoch = 0
        logging.debug(f"Starting epoch {self.epoch}, batches in epoch {self.n_batch_in_epoch}")

        # Calculate the number of batches in the epoch
        steps = self.max_iter - self.n_batch_in_epoch
        train_dataloader_iter = iter(self.train_dataloader)

        print("total batch are: ", steps)
        print("total epoch are:", steps // len(self.train_dataset) + 1)
        # Turn on Torch profiler if self.enable_profile is True
        with profiled_context(steps, enable=self.enable_profile) as total_steps:
            with cuda_timing_context("total time", self.timing):
                for bi in total_steps:
                    with cuda_timing_context("train_dataloader_iter", self.timing):
                        batch = next(train_dataloader_iter)  # Get the next batch

                    self.n_batch_in_epoch += 1

                    # Add the current iteration to the batch dictionary
                    batch["total_iter"] = self.total_iter

                    # Perform forward pass and compute loss
                    with cuda_timing_context("model_train_step", self.timing):
                        loss, loss_dict, results = self.model.train_step(batch, bi)

                    loss = (
                        loss / self.gradient_accumulation_steps
                    )  # Normalize loss for gradient accumulation
                    # Perform backward pass to compute gradients
                    with cuda_timing_context("backward", self.timing):
                        grad_norm = None
                        if isinstance(self.accelerator, Accelerator):
                            self.accelerator.backward(loss)
                            if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.model.get_train_parameters(), self.max_grad_norm
                                )
                        else:
                            loss.backward()

                            if self.max_grad_norm is not None:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.get_train_parameters(), self.max_grad_norm
                                )

                    with cuda_timing_context("Densification", self.timing):
                        with torch.no_grad():
                            # Densification
                            if bi < self.densify_until_iter:  # 3000
                                gaussians = results.get("gaussians", None)
                                gaussians.optimizer = self.optimizer

                                radii = results.get("radii", None)
                                radii = radii[:, 0]

                                visibility_filter = results.get("visibility_filter", None)
                                visibility_filter = visibility_filter[:, 0]

                                screenspace_points = results.get("screenspace_points", None)

                                self.scene_extent = batch["meta_data"]["radius"].item()

                                if (
                                    bi % 500 == 0
                                    and gaussians.active_sh_degree < gaussians.sh_degree
                                ):
                                    gaussians.active_sh_degree += 1

                                # Keep track of max radii in image-space for pruning
                                gaussians.max_radii2D[visibility_filter] = torch.max(
                                    gaussians.max_radii2D[visibility_filter],
                                    radii[visibility_filter],
                                )  # [1,n]

                                gaussians.add_densification_stats(
                                    screenspace_points, visibility_filter
                                )

                                if bi > 500 and bi % self.densification_interval == 0:
                                    size_threshold = 20 if (bi - 1) > 1000 else None
                                    gaussians.densify_and_prune(
                                        self.densify_grad_threshold,
                                        0.005,
                                        self.scene_extent,
                                        size_threshold,
                                        radii,
                                    )  # self.scene_extent=1.0
                                    logging.info(
                                        f"!!!gaussains nums after densify is: {gaussians._means.shape}"
                                    )

                                # 与PGSR不同的地方是这里没有删除out_observe的点

                                if bi != 1 and (bi - 1) % 1000 == 0:
                                    gaussians.reset_opacity()

                        if grad_norm is not None and (
                            torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item()
                        ):
                            logging.warning(
                                f"{batch['meta_data']['name'][0]}, Grad is NaN or Inf, setting grad to zero."
                            )
                            self.optimizer.zero_grad()
                            self.cam_optimizer.zero_grad()
                        elif (
                            grad_norm is not None
                            and self.skip_grad_norm is not None
                            and grad_norm > self.skip_grad_norm
                        ):
                            logging.warning(
                                f"{batch['meta_data']['name'][0]}, Grad > {self.skip_grad_norm}, set grad zero"
                            )
                            self.optimizer.zero_grad()
                            self.cam_optimizer.zero_grad()

                    # Accumulate gradients and losses
                    accumulated_step += 1
                    accumulated_loss["loss"] += loss.item()
                    if loss_dict is not None:
                        for key, val in loss_dict.items():
                            accumulated_loss[key] += val

                    # Update model weights when enough gradients have been accumulated
                    if accumulated_step >= self.gradient_accumulation_steps:
                        last_lr = self.lr_scheduler.get_last_lr()
                        cam_last_lr = self.cam_lr_scheduler.get_last_lr()
                        self.optimizer.step()
                        if (
                            self.cam_optimizer_end is None
                            or self.total_iter <= self.cam_optimizer_end
                        ):
                            self.cam_optimizer.step()
                        self.lr_scheduler.step(self.total_iter)  # Update learning rate
                        self.cam_lr_scheduler.step(self.total_iter)  # Update learning rate
                        self.optimizer.zero_grad()  # Reset gradients after weight update
                        self.cam_optimizer.zero_grad()

                    # Log metrics and training information
                    if self.accelerator.is_main_process:
                        if (self.logging_step > 0) and (self.total_iter % self.logging_step == 0):
                            with cuda_timing_context("Log", self.timing):
                                for i, lr in enumerate(last_lr):
                                    self.tb_logger.writer.add_scalar(
                                        f"lr/lr{i}", lr, global_step=self.total_iter
                                    )
                                for key, val in accumulated_loss.items():
                                    self.tb_logger.writer.add_scalar(
                                        f"train/{key}", val, global_step=self.total_iter
                                    )

                                if grad_norm is not None:
                                    self.tb_logger.writer.add_scalar(
                                        "train/grad_norm",
                                        grad_norm.item(),
                                        global_step=self.total_iter,
                                    )

                                # Log iteration information
                                logging.info("")
                                text = f"iter{self.total_iter:d}, epoch{self.epoch:d}, nbatch{self.n_batch_in_epoch:d}, "
                                text += ", ".join(
                                    [f"{key}:{val:.4f}" for key, val in accumulated_loss.items()]
                                )
                                text += ", "
                                text += ", ".join(
                                    [f"lr{i}:{lr:.3e}" for i, lr in enumerate(last_lr)]
                                )
                                text += ", " + ", ".join(
                                    [f"cam_lr{i}:{lr:.3e}" for i, lr in enumerate(cam_last_lr)]
                                )
                                if grad_norm is not None:
                                    text += f", grad_norm:{grad_norm.item():.2f}"

                                text += f", mem:{self.get_max_memory()}M, data:{batch['meta_data']['name'][0]}"
                                logging.info(text)

                                cur_time = time.time()
                                iter_time = (cur_time - self.cur_time) / self.logging_step
                                self.cur_time = cur_time
                                self.tb_logger.writer.add_scalar(
                                    f"time/iter_time",
                                    iter_time,
                                    global_step=self.total_iter,
                                )

                        # Update the progress bar
                        progress_bar.update(1)
                        progress_bar_logs = {
                            "loss": accumulated_loss["loss"],
                        }
                        progress_bar.set_postfix(**progress_bar_logs)

                        # Call callback function after each training step
                        with cuda_timing_context("callback", self.timing):
                            self.train_step_callback()

                        accumulated_step = 0
                        accumulated_loss = defaultdict(float)

                        # Check if maximum iterations have been reached
                        if 0 < self.max_iter <= self.total_iter:
                            break

                        self.total_iter += 1

        logging.info(f"{self.accelerator.device}, Training Ended.")

        if isinstance(self.accelerator, Accelerator):
            # Final callback after training ends
            self.train_step_callback(is_last=True)

            if self.build_mesh_final:
                self.build_mesh()

            # Wait for all processes to complete
            self.accelerator.wait_for_everyone()

            # End the training process
            self.accelerator.end_training()

        else:
            # Final callback after training ends
            self.train_step_callback(is_last=True)

            if self.build_mesh_final:
                self.build_mesh()

            # End the training process
            torch.cuda.empty_cache()

    def validate_single_dataset(
        self, data_loader, eval_metrics=None, vis=False, save=False, return_frame_output=False
    ):
        """
        Validates the model on a single dataset (single GPU/CPU) and returns the evaluation results.
        """
        if self.cam_optimizer is None or self.total_iter < self.test_cam_optimizer_start:
            return super().validate_single_dataset(
                data_loader=data_loader,
                eval_metrics=eval_metrics,
                vis=vis,
                save=save,
                return_frame_output=return_frame_output,
            )

        if self.cam_optimizer_end is None or self.total_iter <= self.cam_optimizer_end:
            self.adjust_test_cams(data_loader)

        with torch.inference_mode():
            frame_outputs = []
            eval_results = {}
            frame_eval_results = []
            eval_text_save_path = None
            save_meta_dict = dict()

            dataset_name = getattr(data_loader.dataset, "name", "unnamed")

            batch = [data for data in data_loader]
            output = self.model.infer(batch)

            if self.vis_dir is not None:
                # NOTE: 测评时 debug 的存储路径
                vis_out_dir = os.path.join(self.vis_dir, self.get_ckpt_name(), dataset_name)
                for i in range(len(batch)):
                    batch[i]["meta_data"]["vis_out_dir"] = vis_out_dir

            meta_data = [data["meta_data"] for data in batch]
            scene = meta_data[0]["scene"]

            if return_frame_output:
                frame_outputs.append({"output": output, "meta_data": meta_data})

            if eval_metrics is not None:
                frame_eval_results.append(eval_metrics(batch, output))

            if eval_metrics is not None and len(frame_eval_results) > 0:
                if isinstance(eval_metrics.metrics[0], str):
                    for metric_name in eval_metrics.metrics:
                        if metric_name in frame_eval_results[0]:
                            eval_results[metric_name] = sum(
                                result[metric_name].item() for result in frame_eval_results
                            ) / len(frame_eval_results)
                    eval_text = eval_dict_to_text(
                        val_metrics=eval_results,
                        dataset_name=dataset_name,
                        dataset_num=len(data_loader.dataset),
                    )
                else:
                    multi_eval_results = []
                    for metric_obj in eval_metrics.metrics:
                        eval_result = {}
                        for metric_name in metric_obj.metrics:
                            if metric_name in frame_eval_results[0]:
                                eval_result[metric_name] = sum(
                                    result[metric_name].item() for result in frame_eval_results
                                ) / len(frame_eval_results)
                        multi_eval_results.append(eval_result)
                    eval_text = eval_dict_list_to_text(
                        val_metrics=multi_eval_results,
                        dataset_name=dataset_name,
                        dataset_num=len(data_loader.dataset),
                    )
                    eval_results = {k: v for d in multi_eval_results for k, v in d.items()}

                eval_text = f"Scene: {scene}, Iter: {self.total_iter:d}\n" + eval_text
                eval_text_save_path = os.path.join(
                    self.eval_dir,
                    dataset_name,
                    f"eval-{dataset_name}-iter{self.total_iter:06d}.txt",
                )
                try:
                    os.makedirs(os.path.dirname(eval_text_save_path), exist_ok=True)
                    with open(eval_text_save_path, "w+") as f:
                        f.write(eval_text)
                except Exception:
                    logging.warning("save eval_text failed.")

                eval_text_latest_path = os.path.join(
                    self.eval_dir,
                    f"eval-{dataset_name}-latest.txt",
                )
                os.system(f"cp {eval_text_save_path} {eval_text_latest_path}")

                logging.info(f"Evaluation results on {dataset_name}: {eval_text}")
                logging.info(f"Saved evaluation results to: {eval_text_save_path}")
            else:
                eval_results = None

            if vis:
                vis_out_dir = os.path.join(self.vis_dir, self.get_ckpt_name(), dataset_name)
                os.makedirs(vis_out_dir, exist_ok=True)
                self.model.visualize(output, meta_data=meta_data, out_dir=vis_out_dir)

            if save:
                save_out_dir = os.path.join(
                    self.output_dir, "outputs", self.get_ckpt_name(), dataset_name
                )
                os.makedirs(save_out_dir, exist_ok=True)
                self.model.save_output(
                    output,
                    meta_data=meta_data,
                    out_dir=save_out_dir,
                    output_meta_dict=save_meta_dict,
                )

                if len(save_meta_dict) > 0:
                    new_data_path = os.path.join(save_out_dir, "data_info_with_depth.json")
                    with open(new_data_path, "w") as f:
                        json.dump(save_meta_dict, f, indent=2)
                    logging.info(f"output meta saved to: {new_data_path}")

            return frame_outputs if return_frame_output else None, eval_results, eval_text_save_path

    def adjust_test_cams(self, data_loader):
        if self.accelerator.is_main_process:
            logging.info("------ Adjusting test cameras ------")

        grad_norm = None  # To store the gradient norm for monitoring
        accumulated_step = 0  # Count of gradient accumulation steps
        accumulated_loss = defaultdict(float)  # Store accumulated losses during training

        dataset_name = getattr(data_loader.dataset, "name", "unnamed")
        progress_bar = tqdm(
            data_loader, desc=f"Adjusting test cameras on {dataset_name}", total=len(data_loader)
        )

        for single_batch in data_loader:
            for i in range(self.cam_optimizer_iters):
                # Perform forward pass and compute loss
                loss, loss_dict, results = self.model.train_step(single_batch, 0)

                loss = loss / self.gradient_accumulation_steps
                # Perform backward pass to compute gradients
                grad_norm = None
                if isinstance(self.accelerator, Accelerator):
                    self.accelerator.backward(loss)
                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.get_train_parameters(), self.max_grad_norm
                        )
                else:
                    loss.backward()

                    if self.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.get_train_parameters(), self.max_grad_norm
                        )

                if grad_norm is not None and (
                    torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item()
                ):
                    logging.warning(
                        f"{single_batch['meta_data']['name'][0]}, Grad is NaN or Inf, setting grad to zero."
                    )
                    self.optimizer.zero_grad()
                    self.cam_optimizer.zero_grad()
                elif (
                    grad_norm is not None
                    and self.skip_grad_norm is not None
                    and grad_norm > self.skip_grad_norm
                ):
                    logging.warning(
                        f"{single_batch['meta_data']['name'][0]}, Grad > {self.skip_grad_norm}, set grad zero"
                    )
                    self.optimizer.zero_grad()
                    self.cam_optimizer.zero_grad()

                # Accumulate gradients and losses
                accumulated_step += 1
                accumulated_loss["loss"] += loss.item()
                if loss_dict is not None:
                    for key, val in loss_dict.items():
                        accumulated_loss[key] += val

                # Update model weights when enough gradients have been accumulated
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.zero_grad()
                    self.cam_optimizer.step()
                    self.cam_optimizer.zero_grad()

                    accumulated_step = 0
                    accumulated_loss = defaultdict(float)

            progress_bar.update(1)
        progress_bar.close()

    def validate(
        self, save_best=False, save_tb=False, vis=False, save_outputs=False, build_mesh=False
    ):
        # NOTE: gs render depth, and build mesh
        if build_mesh:
            self.build_mesh()
        else:
            super().validate(save_best=save_best, save_tb=save_tb, vis=vis, save_outputs=save_outputs)

    def build_mesh(self):
        with torch.inference_mode():
            dataset_name = getattr(self.val_dataloaders[0].dataset, "name", "unnamed")
            vis_out_dir = os.path.join(self.vis_dir, self.get_ckpt_name(), dataset_name)
            os.makedirs(vis_out_dir, exist_ok=True)

            batch = []
            # NOTE: 训练集只取 image, frame_id, view_id 和 object_mask
            for dataset in self.train_dataset.datasets:
                for data in dataset.gs_total_datas:
                    meta_data = data["meta_data"]
                    fix_data = dict(
                        image=data["image"][None, None],
                        meta_data=dict(
                            frame_id=meta_data["frame_id"],
                            view_id=meta_data["view_id"],
                        ),
                    )
                    if self.model.object_mask_name in data:
                        fix_data[self.model.object_mask_name] = data[self.model.object_mask_name][
                            None, None
                        ]
                    batch.append(fix_data)

            for data_loader in self.val_dataloaders:
                batch += [data for data in data_loader]

            batch = batch[::self.mesh_frame_step]
            output, cameras = self.model.infer(batch, return_cameras=True)

            self.model.build_mesh(outputs_list=output, camera_list=cameras, out_dir=vis_out_dir)
