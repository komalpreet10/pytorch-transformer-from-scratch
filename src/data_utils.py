# src/data_utils.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

MODEL_ID      = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_DATASET = "lavita/ChatDoctor-HealthCareMagic-100k"
TEST_DATASET  = "lavita/medical-qa-datasets"
TEST_SUBSET   = "chatdoctor-icliniq"

TRAIN_SAMPLE  = 10000  # set to None for full 112K
TEST_SAMPLE   = 200    # set to None for full 7.3K


def get_tokenizer():
    """
    Load and configure tokenizer for Llama 3.2 1B Instruct.
    Sets pad token to eos token and padding to right side.

    Returns:
        AutoTokenizer: Configured Llama 3.2 tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def format_prompt(row, tokenizer):
    """
    Format a training dataset row into Llama 3.2 chat template format.

    Args:
        row (dict): Dataset row with keys instruction, input, output
        tokenizer: Llama 3.2 tokenizer

    Returns:
        str: Formatted prompt string with Llama 3.2 special tokens
    """
    messages = [
        {"role": "system",    "content": row["instruction"]},
        {"role": "user",      "content": row["input"]},
        {"role": "assistant", "content": row["output"]}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def load_train_data(tokenizer, sample=TRAIN_SAMPLE):
    """
    Load and format ChatDoctor-HealthCareMagic training dataset.

    Args:
        tokenizer: Llama 3.2 tokenizer
        sample (int, optional): Rows to load. Default 10K. None = full 112K.

    Returns:
        Dataset: With added 'text' column in Llama 3.2 format
    """
    dataset = load_dataset(TRAIN_DATASET, split="train")
    if sample is not None:
        dataset = dataset.select(range(sample))
    dataset = dataset.map(lambda x: {"text": format_prompt(x, tokenizer)})
    return dataset


def load_test_data(sample=TEST_SAMPLE):
    """
    Load ChatDoctor-iCliniq test dataset (raw, no formatting).
    Columns: input, answer_icliniq, answer_chatgpt, answer_chatdoctor

    Args:
        sample (int, optional): Rows to load. Default 200. None = full 7.3K.

    Returns:
        Dataset: With input and answer_icliniq columns
    """
    dataset = load_dataset(TEST_DATASET, TEST_SUBSET, split="test")
    if sample is not None:
        dataset = dataset.select(range(sample))
    return dataset