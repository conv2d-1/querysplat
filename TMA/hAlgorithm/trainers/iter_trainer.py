import datetime
import logging
import time
from collections import defaultdict

import torch
from accelerate import Accelerator
from tqdm import tqdm

from hAlgorithm.modules.utils.optimizer import build_optimizer_and_scheduler_with_cfg
from hAlgorithm.utils import (
    cuda_timing_context,
    profiled_context,
)

from .base_trainer import BaseTrainer


class IterTrainer(BaseTrainer):
    def __init__(self, log_step=1, train_amp=False, **kwargs):
        self.log_step = log_step
        self.train_amp = train_amp

        if isinstance(kwargs["accelerator"], Accelerator):
            self.train_amp = False

        super(IterTrainer, self).__init__(**kwargs)

    def register_schedule(self):
        """Registers the learning rate schedule or any other training schedules."""
        # Optimizer !should be defined after input layer is adapted
        self.optimizer, self.lr_scheduler = build_optimizer_and_scheduler_with_cfg(
            self.optimizer, self.lr_scheduler, self.model
        )

    def accelerator_prepare(self):
        if isinstance(self.accelerator, Accelerator):
            super().accelerator_prepare()
        else:
            if self.train_amp:
                self.model.to(device=self.accelerator.device)
                self.model.device = self.accelerator.device
                self.model.dtype = torch.float32
                self.scaler = torch.amp.GradScaler()
            else:
                self.model.to(device=self.accelerator.device, dtype=self.accelerator.dtype)
                self.model.device = self.accelerator.device
                self.model.dtype = self.accelerator.dtype

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

        # Iterate over epochs
        self.epoch = 0
        logging.debug(f"Starting epoch {self.epoch}, batches in epoch {self.n_batch_in_epoch}")

        # Calculate the number of batches in the epoch
        steps = self.max_iter - self.n_batch_in_epoch
        train_dataloader_iter = iter(self.train_dataloader)

        # Turn on Torch profiler if self.enable_profile is True
        with profiled_context(steps, enable=self.enable_profile) as total_steps:
            for bi in total_steps:
                with cuda_timing_context("train_dataloader_iter", self.timing):
                    try:
                        batch = next(train_dataloader_iter)  # Get the next batch
                    except Exception:
                        logging.warning('cfg["trainer"]["sampler"] != "MixedMaxIterBatchSampler"')
                        break

                self.n_batch_in_epoch += 1

                # Adjust batch size dynamically if needed for distributed training
                if self.accelerator_dynamic_batch and self.accelerator.deepspeed_plugin is not None:
                    self.accelerator.state.deepspeed_plugin.deepspeed_config[
                        "train_micro_batch_size_per_gpu"
                    ] = batch[self.accelerator_dynamic_batch_key].shape[0]

                # Set up random seed for batch processing if global seed is provided
                # if self.seed is not None:
                #     local_seed = self.get_next_seed()
                #     generator = torch.Generator(device=self.accelerator.device)
                #     generator.manual_seed(local_seed)
                # else:
                #     generator = None
                # batch["generator"] = generator

                # Add the current iteration to the batch dictionary
                batch["total_iter"] = self.total_iter

                # Perform forward pass and compute loss
                if self.train_amp:
                    with torch.amp.autocast(
                        device_type=str(self.accelerator.device), dtype=self.accelerator.dtype
                    ):
                        with cuda_timing_context("model_train_step", self.timing):
                            loss, loss_dict = self.model.train_step(batch)
                else:
                    with cuda_timing_context("model_train_step", self.timing):
                        loss, loss_dict = self.model.train_step(batch)

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
                        if self.train_amp:
                            self.scaler.scale(loss).backward()
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
                    last_lr = self.lr_scheduler.get_last_lr()
                    if self.train_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.lr_scheduler.step(self.total_iter)  # Update learning rate
                    else:
                        self.optimizer.step()
                        self.lr_scheduler.step(self.total_iter)  # Update learning rate
                        self.optimizer.zero_grad()  # Reset gradients after weight update

                    # Log metrics and training information
                    if self.accelerator.is_main_process:
                        if self.total_iter % self.log_step == 0:
                            for i, lr in enumerate(last_lr):
                                self.tb_logger.writer.add_scalar(
                                    f"lr/lr{i}", lr, global_step=self.total_iter
                                )
                            for key, val in accumulated_loss.items():
                                self.tb_logger.writer.add_scalar(
                                    f"train/{key}", val, global_step=self.total_iter
                                )
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
                                self.tb_logger.writer.add_scalar(
                                    f"data_grad_norm/{batch['meta_data']['name'][0]}",
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
                            text += ", ".join([f"lr{i}:{lr:.3e}" for i, lr in enumerate(last_lr)])
                            if grad_norm is not None:
                                text += f", grad_norm:{grad_norm.item():.2f}"

                            text += f", mem:{self.get_max_memory()}M, data:{batch['meta_data']['name'][0]}"
                            logging.info(text)

                            cur_time = time.time()
                            iter_time = cur_time - self.cur_time
                            self.cur_time = cur_time
                            total_time = cur_time - self.start_time
                            mean_time = total_time / (self.total_iter - self.start_iter + 1)
                            eta_time = mean_time * (self.max_iter - self.total_iter)
                            total_time = str(datetime.timedelta(seconds=int(total_time)))
                            eta_time = str(datetime.timedelta(seconds=int(eta_time)))
                            logging.info(
                                f"iter:{iter_time:.2f}, mean:{mean_time:.2f}, total:{total_time}, eta:{eta_time}"
                            )

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

                    if self.accelerator.is_main_process and self.total_iter % self.log_step == 0:
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
