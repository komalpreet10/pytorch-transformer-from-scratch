# src/viz_utils.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"

TECHNIQUES = ["baseline", "linear_probing", "lora", "qlora"]


def load_all_results():
    """
    Load metrics from all technique JSON files.

    Returns:
        pd.DataFrame: One row per technique with all metrics
    """
    records = []
    for technique in TECHNIQUES:
        path = os.path.join(RESULTS_DIR, f"{technique}_metrics.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                metrics = json.load(f)
            metrics["technique"] = technique
            records.append(metrics)
        else:
            print(f"Warning: {path} not found — skipping")

    df = pd.DataFrame(records).set_index("technique")
    return df


def plot_rouge(df):
    """
    Bar chart comparing ROUGE-L across techniques.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df["rougeL"].plot(
        kind="bar",
        color="steelblue",
        figsize=(8, 5)
    )
    plt.title("ROUGE-L by Technique")
    plt.xlabel("Technique")
    plt.ylabel("ROUGE-L")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rouge_comparison.png"))
    plt.close()
    print("Saved rouge_comparison.png")


def plot_bertscore(df):
    """
    Bar chart comparing BERTScore F1 across techniques.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df["bertscore_f1"].plot(
        kind="bar",
        color="darkorange",
        figsize=(8, 5)
    )
    plt.title("BERTScore F1 by Technique")
    plt.xlabel("Technique")
    plt.ylabel("BERTScore F1")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bertscore_comparison.png"))
    plt.close()
    print("Saved bertscore_comparison.png")


def plot_latency(df):
    """
    Bar chart comparing inference latency across techniques.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df["latency_ms"].plot(
        kind="bar",
        color="seagreen",
        figsize=(8, 5)
    )
    plt.title("Inference Latency by Technique (ms)")
    plt.xlabel("Technique")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_comparison.png"))
    plt.close()
    print("Saved latency_comparison.png")


def plot_params(df):
    """
    Bar chart comparing trainable parameter % across techniques.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df["trainable_pct"].plot(
        kind="bar",
        color="mediumpurple",
        figsize=(8, 5)
    )
    plt.title("Trainable Parameters % by Technique")
    plt.xlabel("Technique")
    plt.ylabel("Trainable Parameters (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "params_comparison.png"))
    plt.close()
    print("Saved params_comparison.png")


def plot_rouge_vs_params(df):
    """
    Scatter plot — ROUGE-L vs trainable parameters.
    Shows accuracy-efficiency tradeoff.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for technique in df.index:
        ax.scatter(
            df.loc[technique, "trainable_pct"],
            df.loc[technique, "rougeL"],
            s=100,
            label=technique
        )
        ax.annotate(
            technique,
            (df.loc[technique, "trainable_pct"],
             df.loc[technique, "rougeL"]),
            textcoords="offset points",
            xytext=(8, 4)
        )

    ax.set_title("ROUGE-L vs Trainable Parameters")
    ax.set_xlabel("Trainable Parameters (%)")
    ax.set_ylabel("ROUGE-L")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rouge_vs_params.png"))
    plt.close()
    print("Saved rouge_vs_params.png")


def print_results_table(df):
    """
    Print final comparison table to console.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    print("\n" + "="*70)
    print("FINAL RESULTS TABLE")
    print("="*70)
    print(df[[
        "rouge1", "rouge2", "rougeL",
        "bertscore_f1", "trainable_pct",
        "gpu_memory_mb", "latency_ms"
    ]].to_string())
    print("="*70 + "\n")


def save_results_csv(df):
    """
    Save results dataframe to CSV.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "comparison_metrics.csv")
    df.to_csv(path)
    print(f"Saved comparison_metrics.csv")


def generate_all(df):
    """
    Run all plots and save results table.
    Call this from notebook 06.

    Args:
        df (pd.DataFrame): Results dataframe from load_all_results()
    """
    plot_rouge(df)
    plot_bertscore(df)
    plot_latency(df)
    plot_params(df)
    plot_rouge_vs_params(df)
    print_results_table(df)
    save_results_csv(df)
    print("\nAll plots and results saved.")