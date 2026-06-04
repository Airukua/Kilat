from __future__ import annotations
import time
from datetime import datetime
import torch

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def init_wandb(report_to: str, run_name: str, args_dict: dict, model_config: dict) -> None:
    """
    Initialize Weights & Biases run if configured and available.
    
    Design decisions:
    - report_to="wandb" string check allows easy extension to other logging backends
      (e.g., TensorBoard, MLflow) without changing the function signature.
    - HAS_WANDB guard prevents crashes when wandb isn't installed but report_to is 
      set to something else (or during testing/CI environments without wandb).
    - resume="allow": If a previous run with the same name was interrupted, wandb 
      will attempt to resume logging to the same run. This prevents creating 
      duplicate runs when training is preempted and restarted. If the run doesn't 
      exist, a new one is created.
    - Merging args_dict and model_config into config provides a comprehensive 
      hyperparameter and architecture record for experiment tracking. This makes 
      comparing runs in the wandb dashboard straightforward.
    
    Parameters
    ----------
    report_to : str
        Logging backend identifier. Only "wandb" is currently supported.
    run_name : str
        Display name for the wandb run (useful for identifying experiments).
    args_dict : dict
        Training hyperparameters to log as wandb config.
    model_config : dict
        Model architecture configuration (from model.config.to_dict()) to log.
    """
    if report_to == "wandb" and HAS_WANDB:
        wandb.init(
            project="kilat-transformer",
            name=run_name,
            config={
                **args_dict,
                **model_config,
            },
            resume="allow",
        )


def log_training_metrics(
    global_step: int,
    total_steps: int,
    loss: float,
    ppl: float,
    lr: float,
    grad_norm: torch.Tensor,
    current_epoch: int,
    start_time: float,
    report_to: str,
) -> None:
    """
    Log training metrics to both stdout and WandB at specified intervals.
    
    This function is called periodically during training (every logging_steps).
    It computes throughput statistics (steps/sec, ETA) from the wall-clock time
    elapsed since training began, which is more accurate than per-batch timing
    since it smooths out variability.
    
    Throughput calculation:
    - steps_per_sec uses the full training duration, giving a realistic average
      that accounts for data loading, checkpointing, and evaluation overhead.
    - max(1.0, ...) prevents division by zero in edge cases (e.g., first step
      where elapsed time might be near zero).
    - remaining_sec is estimated using the average throughput, providing a 
      reasonable ETA even though throughput can vary throughout training.
    
    Parameters
    ----------
    global_step : int
        Current optimizer step count (after gradient accumulation).
    total_steps : int
        Total planned optimizer steps.
    loss : float
        Current training loss (averaged over accumulation steps).
    ppl : float
        Perplexity (exp(loss)). May be inf for very high losses.
    lr : float
        Current learning rate from scheduler.
    grad_norm : torch.Tensor
        Gradient norm after clipping (for monitoring gradient health).
    current_epoch : int
        Current epoch number (for epoch-based training context).
    start_time : float
        time.time() from when training started (used for elapsed calculation).
    report_to : str
        Logging backend identifier.
    """
    elapsed = time.time() - start_time
    steps_per_sec = global_step / max(1.0, elapsed)
    remaining_sec = (total_steps - global_step) / max(1.0, steps_per_sec)

    # Print to stdout for real-time monitoring in terminal/logs.
    # This is essential even with WandB because:
    # 1. Not all users have access to the WandB dashboard during training
    # 2. Log files capture stdout for post-hoc analysis
    # 3. Provides immediate feedback without refreshing a web interface
    _print_metrics(global_step, total_steps, loss, ppl, lr, grad_norm, steps_per_sec, remaining_sec)

    # WandB logging: Uses step=global_step for proper x-axis alignment in charts.
    # This ensures all metrics are synchronized even if logging frequency varies.
    if report_to == "wandb" and HAS_WANDB:
        wandb.log(
            {
                "train/loss": loss,
                "train/ppl": ppl,
                "train/learning_rate": lr,
                "train/grad_norm": grad_norm,
                "train/epoch": current_epoch,
                "train/steps_per_sec": steps_per_sec,
            },
            step=global_step,
        )


def _print_metrics(
    global_step: int,
    total_steps: int,
    loss: float,
    ppl: float,
    lr: float,
    grad_norm: torch.Tensor,
    steps_per_sec: float,
    remaining_sec: float,
) -> None:
    """
    Print a single formatted line of training metrics to stdout.
    
    Formatting decisions:
    - Timestamp included for correlating metrics with other log sources (e.g., 
      system logs, error messages). HH:MM:SS is sufficient since most training 
      runs span hours, not days.
    - Step counter formatted with commas for readability (e.g., "12,345").
    - PPL displayed as "inf" when >= 10,000 because:
      * Perplexities above 10k indicate catastrophic training failure
      * The number itself becomes meaningless at that scale
      * It prevents formatting issues with very large numbers
    - Fixed-width fields ensure column alignment, making it easy to scan metrics
      as training progresses.
    - Remaining time shown in minutes since that's the natural unit for training
      runs lasting tens of minutes to hours.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    remaining_min = remaining_sec / 60.0
    ppl_str = f"{ppl:.1f}" if ppl < 10_000 else "inf"
    print(
        f"[{timestamp}] "
        f"Step: {global_step:>6,}/{total_steps:>6,} | "
        f"Loss: {loss:>8.4f} | "
        f"PPL: {ppl_str:>8} | "
        f"LR: {lr:>8.2e} | "
        f"Grad: {grad_norm:>6.2f} | "
        f"Speed: {steps_per_sec:>6.1f} st/s | "
        f"Remaining: {remaining_min:>6.1f} min"
    )


def log_eval_summary(
    avg_eval_loss: float,
    eval_ppl: float,
    global_step: int,
    best_eval_loss: float,
    report_to: str,
) -> None:
    """
    Print and log evaluation results after a full eval pass.
    
    This is called after evaluation completes (every eval_steps or at epoch end).
    It provides a clear visual delimiter in the log output to distinguish eval
    results from training metrics.
    
    Best loss tracking:
    - Only displayed when best_eval_loss != float("inf"), indicating at least 
      one evaluation has occurred (initial value is inf).
    - This context helps users see if the current evaluation is a new best or not.
    
    Visual separators (dashed lines) create clear sections in the log output,
    making it easy to search for evaluation results in log files.
    
    Parameters
    ----------
    avg_eval_loss : float
        Mean loss across all eval batches.
    eval_ppl : float
        Validation perplexity.
    global_step : int
        Current training step at evaluation time.
    best_eval_loss : float
        Best validation loss achieved so far (for comparison).
    report_to : str
        Logging backend identifier.
    """
    ppl_str = f"{eval_ppl:.1f}" if eval_ppl < 10_000 else "inf"
    print(f"\n{'-'*40}")
    print(f"Evaluation Summary")
    print(f"{'-'*40}")
    print(f"Validation Loss: {avg_eval_loss:.4f}")
    print(f"Validation PPL:  {ppl_str}")
    if best_eval_loss != float("inf"):
        print(f"Best Val Loss:   {best_eval_loss:.4f}")
    print(f"Step:            {global_step}")
    print(f"{'-'*40}\n")

    if report_to == "wandb" and HAS_WANDB:
        wandb.log({
            "eval/loss": avg_eval_loss,
            "eval/ppl": eval_ppl,
            "eval/step": global_step,
        })


def log_final_summary(
    global_step: int,
    start_time: float,
    best_eval_loss: float,
    output_dir: str,
) -> None:
    """
    Print final training summary to stdout after training completion.
    
    This provides a concise summary of the entire training run, useful for:
    - Quick assessment without scrolling through full logs
    - Comparing runs by looking at the end of log files
    - Documentation purposes (pasting summaries into experiment tracking sheets)
    
    Best PPL calculation:
    - Only computed when best_eval_loss < 100 to avoid math overflow
    - Uses the same threshold as training PPL display for consistency
    - If no evaluation was performed (best_eval_loss = inf), best_ppl will be inf
    
    Parameters
    ----------
    global_step : int
        Final step count at training completion.
    start_time : float
        time.time() from training start (for elapsed time calculation).
    best_eval_loss : float
        Best validation loss across all evaluations.
    output_dir : str
        Directory where checkpoints and outputs are saved.
    """
    import math

    total_time = time.time() - start_time
    best_ppl = math.exp(best_eval_loss) if best_eval_loss < 100 else float("inf")
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"Total steps:       {global_step:,}")
    print(f"Total time:        {total_time / 60:.1f} min ({total_time:.1f} sec)")
    print(f"Avg steps/sec:     {global_step / max(1.0, total_time):.1f}")
    print(f"Best eval loss:    {best_eval_loss:.4f}")
    print(f"Best eval PPL:     {best_ppl:.1f}")
    print(f"Output directory:  {output_dir}")
    print(f"{'='*60}\n")


def print_training_header(
    output_dir: str,
    save_checkpoints: bool,
    training_mode: str,
    total_steps: int,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
    precision: str,
    scaler_enabled: bool,
    report_to: str,
    seed: int,
) -> None:
    """
    Print a comprehensive configuration header before training begins.
    
    This serves multiple purposes:
    1. Reproducibility: All hyperparameters are logged at the start, ensuring
       that the exact configuration is captured even if external config files change.
    2. Sanity check: Users can quickly verify parameters before a potentially
       expensive training run.
    3. Debugging: If training behaves unexpectedly, the header provides the exact
       configuration used, eliminating guesswork about what parameters were active.
    
    Effective batch size:
    Computed as per_device_batch_size * gradient_accumulation_steps. This is the
    batch size the model "sees" for each optimizer step, and it's the number that
    matters for training dynamics and hyperparameter scaling (e.g., learning rate
    should scale with sqrt of effective batch size under some scaling laws).
    
    Parameters
    ----------
    output_dir : str
        Directory for saved checkpoints and outputs.
    save_checkpoints : bool
        Whether checkpoint saving is enabled.
    training_mode : str
        "steps" or "epochs" - determines training termination condition.
    total_steps : int
        Total optimizer steps (computed from mode and dataset size).
    num_train_epochs : int
        Number of epochs (only relevant in "epochs" mode).
    per_device_train_batch_size : int
        Micro-batch size per device.
    gradient_accumulation_steps : int
        Number of micro-batches before one optimizer step.
    learning_rate : float
        Peak learning rate.
    warmup_steps : int
        Linear warmup duration in optimizer steps.
    weight_decay : float
        AdamW weight decay coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    device : torch.device
        Device being used for training.
    precision : str
        "fp16", "bf16", or "fp32".
    scaler_enabled : bool
        Whether GradScaler is active (only for FP16).
    report_to : str
        Logging backend.
    seed : int
        Random seed for reproducibility.
    """
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"KilatTransformer Training")
    print(f"{'='*60}")
    print(f"Output dir:          {output_dir}")
    print(f"Save checkpoints:    {save_checkpoints}")
    print(f"Training mode:       {training_mode}")
    if training_mode == "steps":
        print(f"Total target steps:  {total_steps:,}")
    else:
        print(f"Total epochs:        {num_train_epochs}")
        print(f"Total target steps:  {total_steps:,}")
    print(f"Batch size (per GPU):{per_device_train_batch_size}")
    print(f"Gradient accum:      {gradient_accumulation_steps}")
    print(f"Effective batch:     {effective_batch}")
    print(f"Learning rate:       {learning_rate:.2e}")
    print(f"Warmup steps:        {warmup_steps}")
    print(f"Weight decay:        {weight_decay}")
    print(f"Max grad norm:       {max_grad_norm}")
    print(f"Device:              {device}")
    print(f"Precision:           {precision.upper()}")
    print(f"GradScaler:          {'active' if scaler_enabled else 'disabled'}")
    print(f"Reporting:           {report_to}")
    print(f"Seed:                {seed}")
    print(f"{'='*60}\n")


def finish_wandb(report_to: str) -> None:
    """
    Safely close WandB logging at training completion.
    
    This should be called in all exit paths (normal completion, early stopping,
    keyboard interrupt) to ensure the wandb run is properly finalized.
    Failing to call finish() can result in:
    - Incomplete metric uploads
    - Runs showing as "running" indefinitely in the wandb dashboard
    - Resource leaks in some wandb versions
    
    The HAS_WANDB guard is necessary because finish_wandb is called from _finish()
    regardless of whether wandb was initialized. This provides a clean no-op when
    wandb is not installed or report_to is not "wandb".
    
    Parameters
    ----------
    report_to : str
        Logging backend identifier. Only "wandb" triggers cleanup.
    """
    if report_to == "wandb" and HAS_WANDB:
        wandb.finish()