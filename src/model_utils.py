# src/model_utils.py

import os
import torch
import yaml
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def load_base_model():
    """
    Load Llama 3.2 1B Instruct in fp16.
    Used for Baseline and Linear Probing.

    Returns:
        AutoModelForCausalLM: Base model in fp16
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )
    model.config.use_cache = False
    return model


def load_quantized_model():
    """
    Load Llama 3.2 1B Instruct in 4-bit NF4 quantization.
    Used for QLoRA.

    Returns:
        AutoModelForCausalLM: Quantized model prepared for k-bit training
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model


def apply_linear_probing(model):
    """
    Freeze all layers except the LM head.

    Args:
        model: Base model

    Returns:
        Model with only lm_head trainable
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True
    return model


def get_lora_config(config_path="configs/lora_config.yaml"):
    """
    Load LoRA configuration from YAML file.

    Args:
        config_path (str): Path to lora_config.yaml

    Returns:
        LoraConfig: PEFT LoRA configuration object
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    lora_keys = ["r", "lora_alpha", "target_modules",
                 "lora_dropout", "bias", "task_type"]
    return LoraConfig(**{k: config[k] for k in lora_keys})


def apply_lora(model, config):
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base or quantized model
        config: LoraConfig object

    Returns:
        PEFT model with LoRA adapters applied
    """
    return get_peft_model(model, config)


def count_parameters(model):
    """
    Count total and trainable parameters.

    Args:
        model: Any PyTorch model

    Returns:
        dict: total_params, trainable_params, trainable_pct
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    pct       = round(trainable / total * 100, 4)
    return {
        "total_params":     total,
        "trainable_params": trainable,
        "trainable_pct":    pct
    }
