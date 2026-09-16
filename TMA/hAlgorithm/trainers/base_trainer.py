import datetime
import json
import logging
import math
import os
import shutil
import time
import traceback
from collections import defaultdict

import torch
import torch.distributed as dist
from tqdm import tqdm

from hAlgorithm.modules.utils.seeding import generate_seed_sequence
from hAlgorithm.utils import (
    cuda_timing_context,
    eval_dict_list_to_text,
    eval_dict_to_text,
    instantiate_from_config,
    profiled_context,
)
from hAlgorithm.utils.tb_logger import tb_logger


class BaseTrainer(object):
    """NOTE
    BaseTrainer(v2): A distributed trainer that extends BaseTrainer(v1) to support multi-GPU inference during evaluation.

    This trainer modifies the base evaluation logic to:
    1. Distribute inference workload across multiple GPUs
    2. Gather results from all GPUs to the main process
    3. The main process handle result aggregation and visualization output
    """

    val_distributed = True

    def __init__(
        self,
        accelerator,
        model,
        train_dataset,
        train_dataloader,
        val_datasets,
        val_dataloaders,
        vis_datasets,
        vis_dataloaders,
        max_epoch=None,
        max_iter=None,
        lr=None,
        lr_scheduler=None,
        optimizer=None,
        in_evaluation=False,
        in_visualize=False,
        gradient_accumulation_steps=1,
        eval_metrics=None,
        main_eval_metric=None,
        main_eval_metric_goal=None,
        backup_period=0,
        val_period=0,
        save_period=0,
        vis_period=0,
        resume=None,
        load_from=None,
        output_dir=None,
        ckpt_dir=None,
        tb_dir=None,
        eval_dir=None,
        vis_dir=None,
        accelerator_dynamic_batch=True,
        accelerator_dynamic_batch_key=None,
        seed=None,
        max_grad_norm=None,
        skip_grad_norm=None,
        timing=False,
        enable_profile=False,
        load_lr_scheduler=True,
        load_optimizer=True,
        dist_test=True,
        logging_test_batch_results=False,
        tb_dataset_split=False,
        mixed_precision=None,
        **kwargs,
    ):
        """
        Initializes the trainer and sets up basic properties.

        :param model: The model instance to be trained.
        :param train_dataset: Training dataset.
        :param train_dataloader: DataLoader for the training dataset.
        :param val_datasets: List of validation datasets.
        :param val_dataloaders: List of DataLoaders for validation datasets.
        :param vis_datasets: List of visualization datasets (optional).
        :param vis_dataloaders: List of DataLoaders for visualization datasets (optional).
        :param output_dir: Root directory for all outputs.
        :param ckpt_dir: Directory for saving checkpoints.
        :param tb_dir: Directory for TensorBoard logs.
        :param eval_dir: Directory for evaluation outputs.
        :param vis_dir: Directory for visualization outputs.
        :param max_epoch: Maximum number of epochs to train.
        :param max_iter: Maximum number of iterations to train.
        :param gradient_accumulation_steps: Number of steps for gradient accumulation.
        :param backup_period: Period (in epochs) for creating backups (0 means no backup).
        :param val_period: Period (in epochs) for validation (0 means no validation).
        :param save_period: Period (in epochs) for saving checkpoints (0 means no saving).
        :param vis_period: Period (in epochs) for visualization (0 means no visualization).
        :param lr: Learning rate for the optimizer.
        :param lr_scheduler: Learning rate for the optimizer.
        :param optimizer: The optimizer used to update the model's parameters.
        :param in_evaluation: Flag indicating whether to perform evaluation before training.
        :param in_visualize: Flag indicating whether to perform visualization before training.
        :param resume: Path to a checkpoint to resume training from.
        :param load_from: Path to a pre-trained model to load weights from.
        :param kwargs: Additional keyword arguments (e.g., custom configurations).
        :param accelerator: Accelerator object for distributed training or mixed precision.
        :param accelerator_dynamic_batch: Whether to use dynamic batch size with the accelerator.
        :param accelerator_dynamic_batch_key: Get the batch size of data through the key.
        :param max_grad_norm: Max gradient norm.
        """

        # Initialize basic properties
        self.accelerator = accelerator
        self.mixed_precision = mixed_precision
        self.is_main_process = (self.accelerator is None ) or self.accelerator.is_main_process

        if self.accelerator is None:
            self.use_amp = True
            if self.mixed_precision == "fp16":
                self.amp_dtype = torch.float16
            elif self.mixed_precision == "bf16":
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                else:
                    logging.warning("bf16 is not supported on this device. Using fp16 instead.")
                    self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.float32
                self.use_amp = False

        self.seed = seed
        self.global_seed_sequence = []

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_datasets = val_datasets
        self.val_dataloaders = val_dataloaders
        self.vis_datasets = vis_datasets or []
        self.vis_dataloaders = vis_dataloaders or []

        self.in_evaluation = in_evaluation
        self.in_visualize = in_visualize

        # Initialize epoch and iteration counters
        self.epoch = 1
        self.start_iter = 1
        self.total_iter = 1
        self.n_batch_in_epoch = 0  # Reset at the start of each epoch
        self.best_metric = float("inf") if main_eval_metric_goal == "minimize" else float("-inf")
        self.best_metric_total = None
        if self.val_dataloaders is not None:
            self.best_metric_others = [self.best_metric] * len(self.val_dataloaders)

        # Set learning rate and gradient accumulation steps
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.max_grad_norm = max_grad_norm
        self.skip_grad_norm = skip_grad_norm
        self.eval_metrics = instantiate_from_config(eval_metrics) if eval_metrics else None

        # Set periods for various operations
        self.backup_period = backup_period
        self.val_period = val_period
        self.save_period = save_period
        self.vis_period = vis_period

        # Set directories for outputs
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir
        self.tb_dir = tb_dir
        self.eval_dir = eval_dir
        self.vis_dir = vis_dir

        # Load checkpoints
        self.resume = resume
        self.load_from = load_from
        self.load_lr_scheduler = load_lr_scheduler
        self.load_optimizer = load_optimizer

        # Set main validation metric and goal
        self.main_eval_metric = main_eval_metric
        self.main_eval_metric_goal = main_eval_metric_goal

        # Register the learning rate schedule and other training schedules
        self.model = model
        self.register_schedule()
        self.load_checkpoint()

        # Prepare the accelerator (this method should set up the environment for distributed training, etc.)
        if self.accelerator is not None:
            self.accelerator_dynamic_batch = accelerator_dynamic_batch
            self.accelerator_dynamic_batch_key = accelerator_dynamic_batch_key
            self.accelerator_prepare()
        else:
            self.model = self.model.cuda()
            self.model.device = torch.device("cuda")
            self.model.dtype = self.amp_dtype

        # Initialize TensorBoard logger
        self.tb_logger = tb_logger
        if self.tb_dir is not None and os.path.exists(self.tb_dir) and self.is_main_process:
            self.tb_logger.set_dir(self.tb_dir)

        self.timing = self.is_main_process & timing
        self.enable_profile = self.is_main_process & enable_profile

        self.dist_test = dist_test
        self.logging_test_batch_results = logging_test_batch_results
        self.tb_dataset_split = tb_dataset_split

    def train(self):
        """Main training loop that runs through the dataset for the specified number of epochs or iterations."""

        self.total_iter = self.start_iter

        # Visualize before continuing training (optional)
        if self.in_visualize:
            if self.is_main_process:
                logging.info("Visualizing before continuing training.")
            self.visualize()

        # Perform evaluation before starting the training (optional)
        if self.in_evaluation:
            if self.is_main_process:
                logging.info("Evaluating before continuing training.")
            self.validate(save_best=False, save_tb=True)

        if self.is_main_process:
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

                    # Perform forward pass and compute loss
                    with cuda_timing_context("model_train_step", self.timing):
                        try:
                            loss, loss_dict = self.model.train_step(batch)
                        except Exception as e:
                            logging.warning(e)
                            logging.warning("Train step error, skip this step")
                            torch.cuda.empty_cache()
                            self.optimizer.zero_grad()
                            continue
                    loss = (
                        loss / self.gradient_accumulation_steps
                    )  # Normalize loss for gradient accumulation

                    # Perform backward pass to compute gradients
                    with cuda_timing_context("backward", self.timing):
                        self.accelerator.backward(
                            loss
                        )  # Perform backward pass with gradient clipping
                        if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
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
                        self.optimizer.step()
                        self.lr_scheduler.step(self.total_iter)  # Update learning rate
                        self.optimizer.zero_grad()  # Reset gradients after weight update

                        # Log metrics and training information
                        if self.is_main_process:
                            for i, lr in enumerate(last_lr):
                                self.tb_logger.writer.add_scalar(
                                    f"lr/lr{i}", lr, global_step=self.total_iter
                                )
                            for key, val in accumulated_loss.items():
                                self.tb_logger.writer.add_scalar(
                                    f"train/{key}", val, global_step=self.total_iter
                                )
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
                                "lr": lr,
                            }
                            progress_bar.set_postfix(**progress_bar_logs)

                        # Call callback function after each training step
                        callback_flag = self.train_step_callback()

                        if self.is_main_process:
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

            # Reset batch counter for the next epoch
            self.n_batch_in_epoch = 0

            # End training if maximum iterations have been reached
            if 0 < self.max_iter <= self.total_iter:
                logging.info(f"{self.accelerator.device}, Training Ended.")
                break

        # Wait for all processes to complete
        self.accelerator.wait_for_everyone()

        # Final callback after training ends
        self.train_step_callback(is_last=True)

        # End the training process
        self.accelerator.end_training()

    def train_step_callback(self, is_last=False):
        """
        Executes after every iteration. Handles backup, validation, checkpoint saving, and visualization.

        :param is_last: Boolean flag indicating if this is the last iteration of training.
        """
        latest_saved = False  # Flag to indicate if the latest checkpoint was saved

        # Backup checkpoint at regular intervals (without training state)
        if self.is_main_process:
            if self.backup_period > 0 and self.total_iter % self.backup_period == 0:
                self.save_checkpoint(ckpt_name=self.get_ckpt_name(), save_train_state=True)

        # Perform validation at regular intervals or at the last iteration
        # Use XOR (^) to ensure only one of the conditions is True, but not both
        if (self.val_period > 0 and self.total_iter % self.val_period == 0) ^ is_last:
            self.validate(save_best=True, save_tb=True)
            if self.is_main_process:
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                latest_saved = True

        # Save model checkpoint if needed
        if self.is_main_process:
            if (
                self.save_period > 0
                and self.total_iter % self.save_period == 0
                and not latest_saved
            ):
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Perform visualization if needed
        if (self.vis_period > 0 and self.total_iter % self.vis_period == 0) ^ is_last:
            self.visualize()

        return latest_saved

    def visualize(self):
        """Performs visualization using the visualization datasets."""
        for data_loader in self.vis_dataloaders:
            assert data_loader.batch_size == 1, "Batch size must be 1 for this visualization."
            self.validate_single_dataset(
                data_loader=data_loader, eval_metrics=None, vis=True, save=False
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
        """
        Validates the model on a single dataset using multiple GPUs and returns the evaluation results.
        """
        assert data_loader.batch_size == 1, "Batch size must be 1 for this evaluation."

        # A single-process Accelerator does not initialize torch.distributed.
        # Treat it as a normal local evaluation even if the config enables
        # distributed testing, otherwise dist.barrier/all_gather will fail.
        dist_test = (
            self.dist_test
            and (not app)
            and dist.is_available()
            and dist.is_initialized()
        )

        num_processes = self.accelerator.num_processes if self.accelerator is not None else 1

        frame_outputs = []  # Store model outputs for visualization or saving
        eval_results = {}  # Store evaluation metrics
        frame_eval_results = []  # Store frame-level evaluation results
        eval_text_save_path = None  # Path to save evaluation results
        save_meta_dict = dict()

        # Get the dataset name for logging
        dataset_name = (
            data_loader.dataset.name if hasattr(data_loader.dataset, "name") else "unnamed"
        )

        # Generate a seed sequence for reproducibility if a global seed is provided
        # if self.seed is not None:
        #     seed_sequence = generate_seed_sequence(self.seed, len(data_loader))
        #     if self.is_main_process:
        #         logging.info(f"Global seed sequence generated, length={len(seed_sequence)}")

        total_steps = math.ceil(len(data_loader.dataset) / data_loader.batch_size)

        # Show progress bar only in the main process
        if self.is_main_process:
            progress_bar = tqdm(
                data_loader,
                desc=f"Inference on {dataset_name}",
                total=total_steps,
                disable=not self.is_main_process,
            )

        # Create output directories (only on main process)
        if vis or save:
            if vis:
                vis_out_dir = os.path.join(self.vis_dir, self.get_ckpt_name(), dataset_name)
                if self.is_main_process:
                    os.makedirs(vis_out_dir, exist_ok=True)
            if save:
                save_out_dir = os.path.join(
                    self.output_dir, "outputs", self.get_ckpt_name(), dataset_name
                )
                if self.is_main_process:
                    os.makedirs(save_out_dir, exist_ok=True)

            # Synchronize all processes to ensure directories are created
            if dist_test:
                dist.barrier()

        # Unwrap the DDP model for evaluation to prevent DDP forward hooks from
        # causing hangs when ranks have uneven number of samples.  DDP's internal
        # state tracking (reducer, bucket rebuild, etc.) expects symmetric forward
        # calls across all ranks, which cannot be guaranteed during distributed eval.
        _original_inner_model = None
        if self.accelerator is not None and dist_test:
            unwrapped = self.accelerator.unwrap_model(self.model)
            # The Pipeline wraps its inner model with DDP via accelerator.prepare.
            # We temporarily replace pipeline.model with the unwrapped version.
            if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "module"):
                _original_inner_model = unwrapped.model
                unwrapped.model = unwrapped.model.module

        # Iterate over the dataset to perform inference
        for batch in data_loader:
            # Disable gradient computation during validation to save memory
            with torch.inference_mode():
                # if self.seed is not None:
                #     local_seed = seed_sequence.pop()
                #     generator = torch.Generator(device=self.accelerator.device)
                #     generator.manual_seed(local_seed)
                #     batch["generator"] = generator

                # Perform forward pass to get the output
                if self.accelerator is None:
                    batch["use_amp"] = self.use_amp
                    batch["amp_dtype"] = self.amp_dtype

                output = self.model.infer(**batch)

                if (
                    isinstance(output, list)
                    and isinstance(output[-1], dict)
                    and output[-1].get("type", None) is not None
                ):
                    # for gs, there are novel views data in the raw batch
                    # we need to use the re-organized batch for evaluation
                    batch = output.pop()

                # Save or visualize outputs directly on each GPU
                if vis:
                    self.model.visualize(
                        output,
                        meta_data=batch["meta_data"],
                        out_dir=vis_out_dir,
                    )

            if save:
                self.model.save_output(
                    output,
                    meta_data=batch["meta_data"],
                    out_dir=save_out_dir,
                    output_meta_dict=save_meta_dict,
                )

            # Store outputs only if needed for return_frame_output
            if return_frame_output:
                frame_outputs.append(output)

            # Compute evaluation metrics if provided
            if eval_metrics is not None:
                eval_results = eval_metrics(batch, output)
                if len(eval_results) > 0:
                    frame_eval_results.append(eval_results)
            
            if self.logging_test_batch_results:
                if isinstance(batch['meta_data']['data_info'][0], dict):
                    logging.info(f"{dataset_name}, {batch['meta_data']['data_idx'][0]}, {batch['meta_data']['data_info'][0]['scene']}")
                else:
                    logging.info(f"{dataset_name}, {batch['meta_data']['data_idx'][0]}, {batch['meta_data']['data_info'][0][0]['scene']}")
                logging.info({
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in frame_eval_results[-1].items()
                })

            # Update progress bar
            if self.is_main_process:
                remaining_steps = min(num_processes, total_steps - progress_bar.n)
                progress_bar.update(remaining_steps)

        # Ensure the progress bar is complete
        if self.is_main_process:
            if progress_bar.n < total_steps:
                progress_bar.update(total_steps - progress_bar.n)
            progress_bar.refresh()
            progress_bar.close()

        # Restore the DDP-wrapped model after evaluation
        if _original_inner_model is not None:
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.model = _original_inner_model

        # NOTE:Synchronize all processes
        if dist_test:
            dist.barrier()

        # Gather evaluation results from all processes
        if eval_metrics is not None and eval_metrics.metrics is not None and len(eval_metrics.metrics) > 0:
            # Convert GPU tensors to Python scalars before all_gather_object to
            # avoid pickle restoring tensors onto foreign GPUs (which creates
            # unwanted CUDA contexts and ~416MB overhead per cross-GPU pair).
            frame_eval_results_cpu = [
                {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
                for result in frame_eval_results
            ]

            if dist_test:
                gathered_frame_eval_results = [None for _ in range(num_processes)]
                dist.all_gather_object(gathered_frame_eval_results, frame_eval_results_cpu)
            else:
                gathered_frame_eval_results = [frame_eval_results_cpu]

            # De-duplicate results from padded ranks (NoDuplicateDistributedSampler
            # may assign duplicate samples to ranks that would otherwise have 0 data
            # when dataset size < world_size).  Determine which ranks are padded by
            # checking the sampler attribute.
            if dist_test and self.is_main_process:
                sampler = getattr(data_loader, "sampler", None)
                is_padded_sampler = hasattr(sampler, "is_padded")
                if is_padded_sampler and sampler.num_samples < num_processes:
                    # Only keep results from the first num_samples ranks
                    # (ranks 0..extra_samples-1 each contribute 1 result)
                    n_real = sampler.num_samples
                    truncated = []
                    for sublist in gathered_frame_eval_results:
                        truncated.extend(sublist if sublist else [])
                    gathered_frame_eval_results = [truncated[:n_real]]

            if self.is_main_process:
                all_frame_eval_results = [
                    result for sublist in gathered_frame_eval_results for result in sublist
                ]

                if len(all_frame_eval_results) > 0:
                    if isinstance(eval_metrics.metrics[0], str):
                        for metric_name in eval_metrics.metrics:
                            if metric_name in all_frame_eval_results[0]:
                                eval_results[metric_name] = sum(
                                    result[metric_name] for result in all_frame_eval_results
                                ) / len(all_frame_eval_results)
                        eval_text = eval_dict_to_text(
                            val_metrics=eval_results,
                            dataset_name=dataset_name,
                            dataset_num=len(data_loader.dataset),
                        )
                    else:
                        multi_eval_results = []
                        for metric_obj in eval_metrics.metrics:
                            eval_results = dict()
                            for metric_name in metric_obj.metrics:
                                if metric_name in all_frame_eval_results[0]:
                                    eval_results[metric_name] = [
                                        result[metric_name]
                                        for result in all_frame_eval_results if metric_name in result
                                    ]
                                    eval_results[metric_name] = sum(eval_results[metric_name]) / len(eval_results[metric_name])
                            multi_eval_results.append(eval_results)
                        eval_text = eval_dict_list_to_text(
                            val_metrics=multi_eval_results,
                            dataset_name=dataset_name,
                            dataset_num=len(data_loader.dataset),
                        )
                        eval_results = {k: round(v, 5) if isinstance(v, float) else v for d in multi_eval_results for k, v in d.items()}

                    logging.info(f"Evaluation results on {dataset_name}: {eval_results}")

                    # Save evaluation results to a text file
                    eval_text = f"Iter: {self.total_iter:d}\n" + eval_text
                    eval_text_save_path = os.path.join(
                        self.eval_dir,
                        dataset_name,
                        f"eval-{dataset_name}-iter{self.total_iter:06d}.txt",
                    )
                    try:
                        os.makedirs(os.path.dirname(eval_text_save_path), exist_ok=True)
                        with open(eval_text_save_path, "w+", encoding="utf-8") as f:
                            f.write(eval_text)
                    except Exception as e:
                        traceback.print_exc()
                        logging.error(e)

                    eval_text_latest_path = os.path.join(
                        self.eval_dir,
                        f"eval-{dataset_name}-latest.txt",
                    )
                    os.system(f"cp {eval_text_save_path} {eval_text_latest_path}")

                    logging.debug(f"Evaluation results on {dataset_name}: {eval_text}")
                    logging.info(f"Saved evaluation results to: {eval_text_save_path}")
        else:
            eval_results = None

        # Save meta data if needed
        if save:
            if dist_test:
                # Gather meta data from all processes — must be unconditional so that
                # every rank participates in the collective call even when it has 0
                # data items (e.g. fewer samples than GPUs with NoDuplicateDistributedSampler).
                gathered_meta_dicts = [None for _ in range(num_processes)]
                dist.all_gather_object(gathered_meta_dicts, save_meta_dict)
            else:
                gathered_meta_dicts = [save_meta_dict]

            # Only proceed with merge if any process actually had data
            has_any_data = any(len(d) > 0 for d in gathered_meta_dicts if d is not None)
            if has_any_data and self.is_main_process:
                # Merge meta data from all processes (skip empty dicts)
                merged_meta_dict = {}
                non_empty_dicts = [d for d in gathered_meta_dicts if d is not None and len(d) > 0]
                if non_empty_dicts and "files" in non_empty_dicts[0]:
                    merged_meta_dict["files"] = []
                    for meta_dict in non_empty_dicts:
                        merged_meta_dict["files"].extend(meta_dict["files"])
                elif non_empty_dicts:
                    merged_meta_dict["mf_files"] = dict()
                    for meta_dict in non_empty_dicts:
                        scene_infos = meta_dict["mf_files"]
                        for scene in scene_infos.keys():
                            if scene in merged_meta_dict["mf_files"]:
                                merged_meta_dict["mf_files"][scene].extend(scene_infos[scene])
                            else:
                                merged_meta_dict["mf_files"][scene] = scene_infos[scene]

                if merged_meta_dict:
                    save_out_dir = os.path.join(
                        self.output_dir, "outputs", self.get_ckpt_name(), dataset_name
                    )
                    new_data_path = os.path.join(save_out_dir, "data_info_with_depth.json")
                    with open(new_data_path, "w") as f:
                        json.dump(merged_meta_dict, f, indent=2, ensure_ascii=False)
                    logging.info(f"output meta save: {new_data_path}")

        # Handle return_frame_output if needed
        if return_frame_output:
            if dist_test:
                gathered_frame_outputs = [None for _ in range(num_processes)]
                dist.all_gather_object(gathered_frame_outputs, frame_outputs)
            else:
                gathered_frame_outputs = [frame_outputs]

            if self.is_main_process:
                frame_outputs = [
                    output for sublist in gathered_frame_outputs for output in sublist
                ]
            else:
                frame_outputs = None

        if self.is_main_process:
            return frame_outputs, eval_results, eval_text_save_path
        else:
            return frame_outputs, None, None

    def register_schedule(self):
        """Registers the learning rate schedule or any other training schedules."""
        raise

    def accelerator_prepare(self):
        """
        Prepares the accelerator for training, including setting up distributed training or mixed precision.
        This function configures the necessary components for efficient and scalable model training.
        """
        self.accelerator.even_batches = False
        # Check if dynamic batch size adjustment is enabled and DeepSpeed plugin is available
        if self.accelerator_dynamic_batch and self.accelerator.deepspeed_plugin is not None:
            # Dynamically set the micro batch size per GPU to 1
            # This is useful for gradient accumulation where effective batch size is larger than the micro batch size
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1

            # Ensure that a key for dynamic batch size adjustment has been provided
            assert (
                self.accelerator_dynamic_batch_key is not None
            ), "An accelerator dynamic batch key must be specified."

        self.optimizer, self.train_dataloader, self.lr_scheduler = self.model.accelerator_prepare(
            accelerator=self.accelerator,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            train_dataloader=self.train_dataloader,
        )

    def get_max_memory(self):
        mem = torch.cuda.max_memory_allocated(device=self.accelerator.device)
        mem_mb = int(mem) // (1024 * 1024)
        return mem_mb

    def get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def validate(self, save_best=False, save_tb=False, vis=False, save_outputs=False):
        """
        Performs validation on the validation datasets and optionally saves the best model.

        :param save_best: Boolean flag indicating whether to save the best model.
        :param save_tb: Boolean flag indicating whether to log results to TensorBoard.
        :param save_outputs: Boolean flag indicating whether to save results.
        """

        for i, data_loader in enumerate(self.val_dataloaders):
            dataset_name = getattr(data_loader.dataset, "name", "unnamed")

            _, eval_results, eval_text_save_path = self.validate_single_dataset(
                data_loader=data_loader,
                eval_metrics=self.eval_metrics,
                vis=vis,
                save=save_outputs,
            )

            if save_tb and eval_results is not None:
                # Log evaluation results to TensorBoard
                self.tb_logger.log_dic(
                    {f"val_{dataset_name}/{k}": v for k, v in eval_results.items()},
                    global_step=self.total_iter,
                )

            # Update main eval metric
            if save_best and 0 == i and eval_results is not None:
                if self.main_eval_metric in eval_results:
                    main_eval_metric = eval_results[self.main_eval_metric]
                    if (
                        "minimize" == self.main_eval_metric_goal
                        and main_eval_metric <= self.best_metric
                        or "maximize" == self.main_eval_metric_goal
                        and main_eval_metric >= self.best_metric
                    ):
                        self.best_metric = main_eval_metric
                        self.best_metric_total = eval_results
                        logging.info(
                            f"Best metric: {self.main_eval_metric} = {self.best_metric} at iteration {self.total_iter}"
                        )
                        best_path = os.path.join(self.ckpt_dir, "best")
                        if os.path.exists(best_path):
                            os.system(f"rm {best_path}/ckpt.pth {best_path}/trainer.ckpt")
                        self.save_checkpoint(
                            ckpt_name="best",
                            save_train_state=True,  # NOTE: resume training from best
                        )
                        best_eval_text_save_path = os.path.join(
                            self.eval_dir,
                            f"eval-{dataset_name}-best.txt",
                        )
                        shutil.copy(eval_text_save_path, best_eval_text_save_path)

            if save_best and i > 0 and eval_results is not None:
                if self.main_eval_metric in eval_results:
                    main_eval_metric = eval_results[self.main_eval_metric]
                    if (
                        "minimize" == self.main_eval_metric_goal
                        and main_eval_metric <= self.best_metric_others[i]
                        or "maximize" == self.main_eval_metric_goal
                        and main_eval_metric >= self.best_metric_others[i]
                    ):
                        self.best_metric_others[i] = main_eval_metric
                        logging.info(
                            f"Best metric: {self.main_eval_metric} = {self.best_metric_others[i]} at iteration {self.total_iter}"
                        )
                        best_eval_text_save_path = os.path.join(
                            self.eval_dir,
                            f"eval-{dataset_name}-best.txt",
                        )
                        shutil.copy(eval_text_save_path, best_eval_text_save_path)

    def load_checkpoint(self):
        """Loads a checkpoint to resume training."""
        if self.resume is not None and self.resume not in ["None", "none"]:
            assert os.path.exists(self.resume), self.resume
            self.model.load_checkpoint(self.resume)

            path = os.path.join(os.path.dirname(self.resume), "trainer.ckpt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.start_iter = checkpoint["total_iter"] + 1
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]

            if isinstance(checkpoint["best_metric"], dict):
                if self.main_eval_metric in checkpoint["best_metric"] and checkpoint["best_metric"][self.main_eval_metric] is not None:
                    self.best_metric = checkpoint["best_metric"][self.main_eval_metric]
                else:
                    self.best_metric = (
                        float("inf") if self.main_eval_metric_goal == "minimize" else float("-inf")
                    )
            elif checkpoint["best_metric"] is not None:
                self.best_metric = checkpoint["best_metric"]

            if "optimizer" in checkpoint and self.load_optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                if self.is_main_process:
                    logging.info(f"optimizer state is loaded from {path}")

            if "lr_scheduler" in checkpoint and self.load_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                if self.is_main_process:
                    logging.info(f"LR scheduler state is loaded from {path}")
        elif self.load_from is not None and self.load_from not in ["None", "none"]:
            assert os.path.exists(self.load_from), self.load_from
            self.model.load_checkpoint(self.load_from)

    def save_checkpoint(self, ckpt_name, save_train_state=False):
        """
        Saves a checkpoint to the specified directory.

        :param ckpt_name: Name of the checkpoint directory.
        :param save_train_state: Boolean flag indicating whether to save the training state.
        """
        if not self.is_main_process:
            return

        ckpt_dir = os.path.join(self.ckpt_dir, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")

        # Save model
        try:
            self.model.save_checkpoint(self.accelerator, ckpt_dir)

            if save_train_state:
                if hasattr(self.optimizer, "state_dict"):
                    optimizer = self.optimizer.state_dict()
                else:
                    optimizer = self.accelerator.get_state_dict(self.optimizer)

                if hasattr(self.lr_scheduler, "state_dict"):
                    lr_scheduler = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler = self.accelerator.get_state_dict(self.lr_scheduler)

                # Save training state
                train_state = {
                    # "optimizer": self.accelerator.get_state_dict(self.optimizer),
                    # "lr_scheduler": self.accelerator.get_state_dict(self.lr_scheduler),
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                    "total_iter": self.total_iter,
                    "epoch": self.epoch,
                    "n_batch_in_epoch": self.n_batch_in_epoch,
                    "best_metric": self.best_metric_total,
                }
                train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
                torch.save(train_state, train_state_path)
                logging.info(f"Trainer state saved to: {train_state_path}")

                with open(os.path.join(ckpt_dir, "history.txt"), "a") as f:
                    f.write(self.get_ckpt_name() + "\n")
        except Exception as e:
            traceback.print_exc()
            logging.error(e)
            logging.error("save_checkpoint error!!!!")

    def get_ckpt_name(self):
        return f"iter_{self.total_iter:06d}"
