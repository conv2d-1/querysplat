import datetime
import logging
import time
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from hAlgorithm.modules.utils.optimizer import build_optimizer_with_cfg
from hAlgorithm.trainers.base_trainer import BaseTrainer
from hAlgorithm.utils import (
    cuda_timing_context,
    eval_dict_list_to_text,
    eval_dict_to_text,
    instantiate_from_config,
    profiled_context,
)


class DataTrainer(BaseTrainer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gradient_accumulation_steps = 1

    def train(self):
        """Main training loop that runs through the dataset for the specified number of epochs or iterations."""

        if self.accelerator.is_main_process:
            logging.info("------ Start training ------")
            logging.info(f"output_dir: {self.output_dir}")

        accumulated_loss = defaultdict(float)

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
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"Starting epoch {self.epoch}, batches in epoch {self.n_batch_in_epoch}")

            # Calculate the number of batches in the epoch
            steps = len(self.train_dataloader) - self.n_batch_in_epoch
            train_dataloader_iter = iter(self.train_dataloader)

            # Turn on Torch profiler if self.enable_profile is True
            with profiled_context(steps, enable=self.enable_profile) as total_steps:
                for bi in total_steps:
                    with cuda_timing_context("train_dataloader_iter", self.timing):
                        batch = next(train_dataloader_iter)  # Get the next batch
                    self.n_batch_in_epoch += 1

                    # Adjust batch size dynamically if needed for distributed training
                    if (
                        self.accelerator_dynamic_batch
                        and self.accelerator.deepspeed_plugin is not None
                    ):
                        self.accelerator.state.deepspeed_plugin.deepspeed_config[
                            "train_micro_batch_size_per_gpu"
                        ] = batch[self.accelerator_dynamic_batch_key].shape[0]

                    # Set up random seed for batch processing if global seed is provided
                    if self.seed is not None:
                        local_seed = self.get_next_seed()
                        generator = torch.Generator(device=self.accelerator.device)
                        generator.manual_seed(local_seed)
                    else:
                        generator = None
                    batch["generator"] = generator

                    # Add the current iteration to the batch dictionary
                    batch["total_iter"] = self.total_iter
                    batch["image_show"] = None

                    # Perform forward pass and compute loss
                    with cuda_timing_context("model_train_step", self.timing):
                        loss, loss_dict = self.model.train_step(batch)
                    loss = (
                        loss / self.gradient_accumulation_steps
                    )  # Normalize loss for gradient accumulation

                    # Accumulate gradients and losses
                    accumulated_step = self.gradient_accumulation_steps
                    accumulated_loss["loss"] += loss.item()
                    if loss_dict is not None:
                        for key, val in loss_dict.items():
                            accumulated_loss[key] += val

                    # Update model weights when enough gradients have been accumulated
                    if accumulated_step >= self.gradient_accumulation_steps:
                        last_lr = self.lr_scheduler.get_last_lr()
                        self.lr_scheduler.step(self.total_iter)  # Update learning rate
                        self.optimizer.zero_grad()  # Reset gradients after weight update

                        # Log metrics and training information
                        if self.accelerator.is_main_process:
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

                            # Log iteration information
                            logging.info("")
                            text = f"iter{self.total_iter:d}, epoch{self.epoch:d}, nbatch{self.n_batch_in_epoch:d}, "
                            text += ", ".join(
                                [f"{key}:{val:.4f}" for key, val in accumulated_loss.items()]
                            )
                            text += ", "
                            text += ", ".join([f"lr{i}:{lr:.3e}" for i, lr in enumerate(last_lr)])

                            text += f", mem:{self.get_max_memory()}M, data:{batch['meta_data']['name'][0]}"
                            text += f", data_idx:{batch['meta_data']['data_idx'][0]}"
                            text += f", rgb_path:{batch['meta_data']['data_info']['rgb'][0]}"
                            if "frame_id" in batch["meta_data"]["data_info"].keys():
                                text += f", scene:{batch['meta_data']['data_info']['scene'][0]}"
                                text += (
                                    f", frame_id:{batch['meta_data']['data_info']['frame_id'][0]}"
                                )
                                text += f", view_id:{batch['meta_data']['data_info']['view_id'][0]}"
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
                                "lr": lr,
                            }
                            progress_bar.set_postfix(**progress_bar_logs)

                        # Call callback function after each training step
                        # callback_flag = self.train_step_callback()

                        # if self.accelerator.is_main_process:
                        #     # Mark time info
                        #     self.tb_logger.writer.add_scalar(
                        #         f"time/iter_time",
                        #         iter_time,
                        #         global_step=self.total_iter,
                        #     )
                        #     self.tb_logger.writer.add_scalar(
                        #         f"time/total_time",
                        #         (cur_time - self.start_time) / 3600.0,
                        #         global_step=self.total_iter,
                        #     )
                        #     callback_time = time.time() - self.cur_time
                        #     if callback_flag:
                        #         self.tb_logger.writer.add_scalar(
                        #             f"time/callback_time",
                        #             callback_time,
                        #             global_step=self.total_iter,
                        #         )
                        #         self.callback_total_time += callback_time
                        #         self.tb_logger.writer.add_scalar(
                        #             f"time/callback_total_time",
                        #             self.callback_total_time / 3600.0,
                        #             global_step=self.total_iter,
                        #         )

                        accumulated_step = 0
                        accumulated_loss = defaultdict(float)

                        # Check if maximum iterations have been reached
                        if 0 < self.max_iter <= self.total_iter:
                            break

                        self.total_iter += 1

            # Reset batch counter for the next epoch
            self.n_batch_in_epoch = 0

            # End training if maximum iterations have been reached
            if 0 < self.max_iter <= self.total_iter:
                logging.info(f"{self.accelerator.device}, Training Ended.")
                break

        # Wait for all processes to complete
        self.accelerator.wait_for_everyone()

        # Final callback after training ends
        # self.train_step_callback(is_last=True)

        # End the training process
        self.accelerator.end_training()

    def register_schedule(self):
        """Registers the learning rate schedule or any other training schedules."""
        # Optimizer !should be defined after input layer is adapted
        self.optimizer = build_optimizer_with_cfg(self.optimizer, self.model)

        # LR scheduler
        lr_func = instantiate_from_config(self.lr_scheduler)
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)
