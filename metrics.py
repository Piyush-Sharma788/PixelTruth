"""
Model Analytics Module - Deepfake Detection Metrics & Visualizations
This module provides helper functions to display model performance metrics,
confusion matrices, ROC curves, and dataset distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# ==================== THEME CONFIGURATION ====================
DARK_BG = '#0f172a'
DARK_CARD = '#1e293b'
TEXT_COLOR = '#e5e7eb'
BORDER_COLOR = '#475569'


def apply_dark_theme(fig, ax):
    """
    Applies consistent dark theme styling to matplotlib figures.
    
    Args:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_CARD)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    return fig, ax


# ==================== SAMPLE METRICS ====================
def get_sample_metrics():
    """
    Returns sample model performance metrics.
    These are realistic values based on typical deepfake detection models.
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1_score
    """
    return {
        "accuracy": 95.2,
        "precision": 94.8,
        "recall": 95.7,
        "f1_score": 95.2,
    }


# ==================== CONFUSION MATRIX ====================
def get_confusion_matrix_plot():
    """
    Generates a confusion matrix heatmap for binary classification
    (Real vs Fake deepfake detection).
    
    Returns:
        matplotlib figure object
    """
    # Sample confusion matrix: [True Negatives, False Positives]
    #                          [False Negatives, True Positives]
    cm = np.array([[475, 25],    # Real images: 475 correct, 25 misclassified
                   [23, 477]])   # Fake images: 23 misclassified, 477 correct
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    apply_dark_theme(fig, ax)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='RdYlGn',
        cbar=True,
        ax=ax,
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_ylabel('True Label', color=TEXT_COLOR)
    ax.set_xlabel('Predicted Label', color=TEXT_COLOR)
    ax.set_title('Confusion Matrix', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


def get_confusion_matrix_caption():
    """Returns explanatory caption for confusion matrix."""
    return (
        "Shows model predictions vs actual labels. Diagonal values (green) indicate correct predictions. "
        "Higher diagonal values demonstrate better model performance."
    )



# ==================== ROC CURVE ====================
def get_roc_curve_plot():
    """
    Generates a ROC (Receiver Operating Characteristic) curve.
    Shows the trade-off between True Positive Rate and False Positive Rate.
    
    Returns:
        matplotlib figure object
    """
    # Sample data: true labels (0=Real, 1=Fake) and predicted probabilities
    y_true = np.array([0]*500 + [1]*500)  # 500 real, 500 fake
    
    # Predicted probabilities (higher = more confident it's fake)
    y_prob = np.concatenate([
        np.random.beta(2, 5, 500),      # Real images: mostly low scores
        np.random.beta(5, 2, 500)       # Fake images: mostly high scores
    ])
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    apply_dark_theme(fig, ax)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#22c55e', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel('True Positive Rate', color=TEXT_COLOR, fontsize=11)
    ax.set_title('ROC Curve', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc="lower right", facecolor=DARK_CARD, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    ax.grid(True, alpha=0.2, color=BORDER_COLOR)
    
    plt.tight_layout()
    return fig


def get_roc_curve_caption():
    """Returns explanatory caption for ROC curve."""
    return (
        "Illustrates the model's ability to distinguish between Real and Fake images across different thresholds. "
        "AUC (Area Under Curve) closer to 1.0 indicates excellent discrimination. Line represents random guessing."
    )



# ==================== DATASET DISTRIBUTION ====================
def get_dataset_distribution_plot():
    """
    Generates a class distribution chart showing the breakdown of
    Real vs Fake images in the training/testing dataset.
    
    Returns:
        matplotlib figure object
    """
    # Sample dataset distribution
    labels = ['Real Images', 'Fake Images']
    sizes = [5000, 5000]  # Balanced dataset
    colors = ['#22c55e', '#ef4444']
    explode = (0.05, 0.05)
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        textprops={'color': TEXT_COLOR, 'fontsize': 10}
    )
    
    # Style percentage text
    for autotext in autotexts:
        autotext.set_color(DARK_BG)
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # Add legend with counts
    ax.legend(
        [f'{labels[0]} ({sizes[0]:,})', f'{labels[1]} ({sizes[1]:,})'],
        loc='upper left',
        bbox_to_anchor=(0.85, 1),
        facecolor=DARK_CARD,
        edgecolor=BORDER_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=10
    )
    
    ax.set_title('Class Distribution', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


def get_dataset_distribution_caption():
    """Returns explanatory caption for dataset distribution."""
    return (
        "Displays the balance between Real and Fake images in the dataset. "
        "Equal distribution (50/50) helps train unbiased models."
    )



# ==================== CLASS STATISTICS ====================
def get_class_statistics():
    """
    Returns statistics for each class (Real and Fake).
    
    Returns:
        dict: Statistics including samples, accuracy per class, etc.
    """
    return {
        "Real": {
            "total_samples": 5000,
            "correctly_classified": 4750,
            "misclassified": 250,
            "class_accuracy": 95.0,
        },
        "Fake": {
            "total_samples": 5000,
            "correctly_classified": 4760,
            "misclassified": 240,
            "class_accuracy": 95.2,
        },
    }
