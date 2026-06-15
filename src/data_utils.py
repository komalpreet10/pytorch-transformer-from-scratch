# src/data_utils.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

MODEL_ID    = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_DATASET = "lavita/ChatDoctor-HealthCareMagic-100k"
TEST_DATASET  = "lavita/medical-qa-datasets"
TEST_SUBSET   = "chatdoctor-icliniq"


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def format_prompt(row, tokenizer):
    """
    Format a dataset row into Llama 3.2 chat template format.

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


def load_train_data(tokenizer):
    """
    Load and format ChatDoctor-HealthCareMagic training dataset.

    Args:
        tokenizer: Llama 3.2 tokenizer
    Returns:
        Dataset: With added 'text' column in Llama 3.2 format
    """
    dataset = load_dataset(TRAIN_DATASET, split="train")
    dataset = dataset.map(lambda x: {"text": format_prompt(x, tokenizer)})
    return dataset


def load_test_data():
    """
    Load ChatDoctor-iCliniq test dataset (raw, no formatting).

    Returns:
        Dataset: With instruction, input, output columns
    """
    dataset = load_dataset(TEST_DATASET, TEST_SUBSET, split="test")
    return dataset

if __name__ == "__main__":
    print("Testing data_utils.py...\n")

    tokenizer = get_tokenizer()
    print(f"Tokenizer loaded: vocab size = {tokenizer.vocab_size}")

    print("\nLoading full training dataset...")
    train = load_train_data(tokenizer)
    print(f"Train rows    : {len(train)}")
    print(f"Train columns : {train.column_names}")
    print(f"\nFormatted prompt preview:")
    print(train[0]['text'][:400])

    print("\nLoading full test dataset...")
    test = load_test_data()
    print(f"Test rows    : {len(test)}")
    print(f"Test columns : {test.column_names}")
    print(f"\nTest input preview:")
    print(test[0]['input'][:200])

    print("\ndata_utils.py working correctly!")
