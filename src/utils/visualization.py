"""
Visualization Utilities
Reusable plotting functions for crime detection project.
"""

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from wordcloud import WordCloud

SEVERITY_LABELS  = ["Low", "Medium", "High"]
SEVERITY_COLORS  = {"Low": "#00d4aa", "Medium": "#ffb347", "High": "#ff6b6b"}
OUTPUT_DIR       = Path("images")


def _save(fig: plt.Figure, filename: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Crime-text plots
# ---------------------------------------------------------------------------

def plot_top_crimes(crime_data: pd.DataFrame, top_n: int = 10, output_dir: Path = OUTPUT_DIR):
    crime_counts = crime_data["PRIMARY DESCRIPTION"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=crime_counts.values, y=crime_counts.index, palette="viridis",
                edgecolor="black", ax=ax)
    for i, count in enumerate(crime_counts.values):
        ax.text(count + 50, i, f"{count:,}", va="center", fontweight="bold")
    ax.set_title(f"Top {top_n} Crime Types", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of Incidents")
    ax.set_ylabel("Crime Type")
    plt.tight_layout()
    _save(fig, "top_crimes.png", output_dir)
    plt.show()


def plot_crime_pie_chart(crime_data: pd.DataFrame, top_n: int = 10, output_dir: Path = OUTPUT_DIR):
    counts = crime_data["PRIMARY DESCRIPTION"].value_counts()
    top    = counts.head(top_n)
    others = counts.iloc[top_n:].sum()
    if others > 0:
        top["Others"] = others
    colors = sns.color_palette("Set2", len(top))
    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(
        top.values, labels=top.index, autopct="%1.1f%%",
        startangle=90, colors=colors,
        textprops={"fontsize": 10, "weight": "bold"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)
    ax.set_title(f"Top {top_n} Crime Types Distribution", fontsize=16, fontweight="bold")
    ax.axis("equal")
    plt.tight_layout()
    _save(fig, "crime_pie_chart.png", output_dir)
    plt.show()


def plot_crime_wordcloud(crime_data: pd.DataFrame, output_dir: Path = OUTPUT_DIR):
    text = " ".join(crime_data["PRIMARY DESCRIPTION"].dropna().str.lower())
    wc = WordCloud(width=1200, height=600, background_color="white",
                   colormap="viridis", max_words=100, relative_scaling=0.5).generate(text)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title("Crime Description Word Cloud", fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _save(fig, "crime_wordcloud.png", output_dir)
    plt.show()


def plot_severity_distribution(
    series: pd.Series,
    severity_map: Dict[str, int],
    title: str,
    filename: str,
    output_dir: Path = OUTPUT_DIR,
):
    counts = series.map(severity_map).fillna(0).astype(int).map(
        {0: "Low", 1: "Medium", 2: "High"}
    ).value_counts()
    for lbl in SEVERITY_LABELS:
        if lbl not in counts:
            counts[lbl] = 0
    counts = counts[SEVERITY_LABELS]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values,
                  color=[SEVERITY_COLORS[l] for l in counts.index],
                  edgecolor="black", linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 50,
                f"{int(h):,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Severity Level", fontweight="bold")
    ax.set_ylabel("Number of Incidents", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, filename, output_dir)
    plt.show()


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
    filename: str = "confusion_matrix.png",
):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS,
                linewidths=2, linecolor="black",
                vmin=0, vmax=cm.max(), cbar_kws={"label": "Count"}, ax=ax)
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha="center", va="center", color=color,
                    fontsize=20, fontweight="bold")
    ax.set_xlabel("Predicted Severity", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Severity",      fontsize=14, fontweight="bold")
    ax.set_title("Confusion Matrix – Conservative Fusion", fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save(fig, filename, output_dir)
    plt.show()


def plot_roc_curves(roc_data: List[Dict], output_dir: Path = OUTPUT_DIR):
    colors = ["#00d4aa", "#ffb347", "#ff6b6b"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, rd in enumerate(roc_data):
        ax.plot(rd["fpr"], rd["tpr"], lw=3, color=colors[i],
                label=f"{rd['class_name']} (AUC = {rd['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate",  fontsize=14, fontweight="bold")
    ax.set_title("ROC Curves – One-vs-Rest (Fused Model)", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "roc_curves.png", output_dir)
    plt.show()


def plot_training_history(history: Dict, output_dir: Path = OUTPUT_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc"), ("train_f1", "val_f1")],
        ["Loss", "Accuracy (%)", "Weighted F1"],
    ):
        ax.plot(history[key[0]], label="Train")
        ax.plot(history[key[1]], label="Val")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    _save(fig, "training_history.png", output_dir)
    plt.show()


def plot_ablation_results(ablation_results: Dict, output_dir: Path = OUTPUT_DIR):
    if not ablation_results:
        print("[Visualization] No ablation results to plot.")
        return

    df = pd.DataFrame.from_dict(ablation_results, orient="index")
    df = df.sort_values("f2_score", ascending=False)

    cmap = LinearSegmentedColormap.from_list("ablation", ["#4e79a7", "#f28e2b", "#e15759"])
    x     = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, name in enumerate(df.index):
        color = cmap(i / max(1, len(df) - 1))
        ax.bar(x[i] - width / 2, df.loc[name, "accuracy"],       width, color=color, edgecolor="white")
        ax.bar(x[i] + width / 2, df.loc[name, "f2_score"] * 100, width, color=color, alpha=0.75, edgecolor="white")
        ax.text(x[i] - width / 2, df.loc[name, "accuracy"] + 0.8,
                f"{df.loc[name, 'accuracy']:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(x[i] + width / 2, df.loc[name, "f2_score"] * 100 + 0.8,
                f"{df.loc[name, 'f2_score']:.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in df.index], rotation=15, ha="right")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Ablation Study: Impact of Modalities", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(1, len(df) - 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[0, max(1, len(df) - 1)], shrink=0.7)
    cbar.ax.set_yticklabels(["Best", "Worst"])

    plt.tight_layout()
    _save(fig, "ablation_results.png", output_dir)
    plt.show()
