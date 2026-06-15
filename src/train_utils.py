# src/train_utils.py

import os
import yaml
import wandb
import mlflow
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer

load_dotenv()


def _wandb_token():
    return os.getenv("WANDB_TOKEN") or os.getenv("WANDB_API_KEY")


def _close_active_runs():
    """Make notebook reruns safe by closing any active tracking runs."""
    if wandb.run is not None:
        wandb.finish()

    while mlflow.active_run() is not None:
        mlflow.end_run()


def init_trackers(technique_name):
    """
    Initialize WandB and MLflow experiment tracking.

    Args:
        technique_name (str): One of baseline, linear_probing, lora, qlora
    """
    _close_active_runs()

    wandb_token = _wandb_token()
    if wandb_token:
        os.environ["WANDB_API_KEY"] = wandb_token
        os.environ.pop("WANDB_MODE", None)
        wandb.login(key=wandb_token, relogin=True)
        wandb.init(
            project="medical-llm-peft-benchmark",
            name=technique_name
        )
    else:
        wandb.init(
            project="medical-llm-peft-benchmark",
            name=technique_name,
            mode="disabled"
        )

    mlflow.set_experiment("medical-llm-peft-benchmark")
    mlflow.start_run(run_name=technique_name)


def log_metrics(metrics):
    """
    Log final metrics to WandB and MLflow.

    Args:
        metrics (dict): Dictionary of metric name to value
    """
    if wandb.run is not None:
        wandb.log(metrics)
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)


def end_trackers():
    """
    Close WandB and MLflow runs cleanly.
    Call after training and evaluation are complete.
    """
    _close_active_runs()


def get_training_args(output_dir, config_path="configs/training_config.yaml"):
    """
    Load training arguments from YAML config file.

    Args:
        output_dir (str): Directory to save model checkpoints
        config_path (str): Path to training_config.yaml

    Returns:
        SFTConfig: TRL supervised fine-tuning configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    report_to = ["wandb"] if _wandb_token() else []

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        fp16=config["fp16"],
        seed=config["seed"],
        report_to=report_to,
        dataset_text_field="text",
        max_length=config["max_seq_length"],
        optim="adamw_torch"
    )


def get_trainer(model, tokenizer, train_dataset, training_args,
                config_path="configs/training_config.yaml"):
    """
    Create SFTTrainer for supervised fine-tuning.

    Args:
        model: PEFT or base model to train
        tokenizer: Llama 3.2 tokenizer
        train_dataset: Formatted training dataset with 'text' column
        training_args: TrainingArguments object
        config_path (str): Path to training_config.yaml

    Returns:
        SFTTrainer: Ready to train
    """
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )
