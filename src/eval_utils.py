# src/eval_utils.py
import json
import os
import time
import torch
import evaluate
from bert_score import score as bert_score


def _model_device(model):
    return next(model.parameters()).device


def format_generation_prompt(row, tokenizer):
    messages = [
        {"role": "system", "content": row.get("instruction", "")},
        {"role": "user", "content": row["input"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_responses(model, tokenizer, dataset, max_new_tokens=200):
    """
    Generate model responses for test dataset.

    Args:
        model: Fine-tuned or base model
        tokenizer: Llama 3.2 tokenizer
        dataset: Test dataset with 'input' column
        max_new_tokens (int): Max tokens to generate per response

    Returns:
        tuple: (predictions, references) — both lists of strings
    """
    model.eval()
    predictions = []
    references  = []

    for row in dataset:
        prompt = format_generation_prompt(row, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(_model_device(model))

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only new tokens
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        predictions.append(generated)
        references.append(row["output"])

    return predictions, references


def compute_rouge(predictions, references):
    """
    Compute ROUGE scores between predictions and references.

    Args:
        predictions (list): Generated responses
        references (list): Ground truth responses

    Returns:
        dict: rouge1, rouge2, rougeL scores
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references
    )
    return {
        "rouge1": round(results["rouge1"], 4),
        "rouge2": round(results["rouge2"], 4),
        "rougeL": round(results["rougeL"], 4)
    }


def compute_bertscore(predictions, references):
    """
    Compute BERTScore F1 between predictions and references.
    Captures semantic similarity beyond exact word overlap.

    Args:
        predictions (list): Generated responses
        references (list): Ground truth responses

    Returns:
        dict: bertscore_f1 mean score
    """
    _, _, f1 = bert_score(
        predictions,
        references,
        lang="en",
        verbose=False
    )
    return {"bertscore_f1": round(f1.mean().item(), 4)}


def measure_latency(model, tokenizer, num_samples=50):
    """
    Measure average inference latency in milliseconds.

    Args:
        model: Model to evaluate
        tokenizer: Llama 3.2 tokenizer
        num_samples (int): Number of samples to average over

    Returns:
        dict: latency_ms average latency
    """
    sample_input = "I have a headache and fever. What should I do?"
    inputs = tokenizer(
        sample_input,
        return_tensors="pt"
    ).to(_model_device(model))

    latencies = []
    model.eval()

    with torch.no_grad():
        for _ in range(num_samples):
            start = time.time()
            model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )
            latencies.append((time.time() - start) * 1000)

    return {"latency_ms": round(sum(latencies) / len(latencies), 2)}


def measure_gpu_memory():
    """
    Measure peak GPU memory usage in MB.

    Returns:
        dict: gpu_memory_mb peak memory
    """
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        return {"gpu_memory_mb": round(memory_mb, 1)}
    return {"gpu_memory_mb": 0}


def save_metrics(metrics, path):
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics (dict): All evaluation metrics
        path (str): Output file path e.g. results/lora_metrics.json
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def evaluate_model(model, tokenizer, test_dataset, technique_name):
    """
    Run full evaluation pipeline and return all metrics.

    Args:
        model: Model to evaluate
        tokenizer: Llama 3.2 tokenizer
        test_dataset: iCliniq test dataset
        technique_name (str): baseline, linear_probing, lora, qlora

    Returns:
        dict: All metrics combined
    """
    print(f"Evaluating {technique_name}...")

    # Generate responses
    predictions, references = generate_responses(
        model, tokenizer, test_dataset
    )

    # Compute all metrics
    metrics = {}
    metrics.update(compute_rouge(predictions, references))
    metrics.update(compute_bertscore(predictions, references))
    metrics.update(measure_latency(model, tokenizer))
    metrics.update(measure_gpu_memory())

    print(f"Results for {technique_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics
