"""
Model Analytics Module - Deepfake Detection Metrics & Visualizations

This module loads real evaluation results from ``metrics_cache.json``
(produced by ``evaluate.py``) and renders interactive Plotly charts for
the Streamlit analytics dashboard.

When no cache file is present, all data-source functions return ``None``
so the caller can display an appropriate warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ==================== THEME CONFIGURATION ====================
DARK_BG = '#0f172a'
DARK_CARD = '#1e293b'
TEXT_COLOR = '#e5e7eb'
BORDER_COLOR = '#475569'

# ==================== DEFAULT CACHE PATH ====================
DEFAULT_CACHE_PATH = "metrics_cache.json"


# ==================== CACHE LOADER ====================
def load_cached_metrics(cache_path: str = DEFAULT_CACHE_PATH) -> Optional[dict]:
    """Load evaluation results from the JSON cache file.

    Parameters
    ----------
    cache_path : str
        Path to the metrics cache JSON file.

    Returns
    -------
    dict or None
        Parsed evaluation results, or ``None`` if the file is missing or
        contains invalid JSON.
    """
    path = Path(cache_path)
    if not path.is_file():
        logger.info(f"Metrics cache not found at {cache_path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Basic schema validation — ensure required keys exist
        required_keys = {
            "accuracy", "precision", "recall", "f1",
            "confusion_matrix", "roc",
        }
        missing = required_keys - set(data.keys())
        if missing:
            logger.warning(
                f"Metrics cache is missing required keys: {missing}. "
                f"Re-run evaluate.py to regenerate."
            )
            return None

        return data

    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not load metrics cache: {exc}")
        return None


# ==================== SAMPLE METRICS ====================
def get_sample_metrics(cached_metrics: Optional[dict] = None) -> Optional[dict]:
    """Return model performance metrics from the evaluation cache.

    Parameters
    ----------
    cached_metrics : dict or None
        Pre-loaded cache dict. If ``None``, returns ``None``.

    Returns
    -------
    dict or None
        Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1_score``.
    """
    if cached_metrics is None:
        return None

    return {
        "accuracy": cached_metrics.get("accuracy", 0.0),
        "precision": cached_metrics.get("precision", 0.0),
        "recall": cached_metrics.get("recall", 0.0),
        "f1_score": cached_metrics.get("f1", 0.0),
    }


# ==================== CONFUSION MATRIX ====================
def get_confusion_matrix_plot(cached_metrics: Optional[dict] = None) -> Optional[go.Figure]:
    """Generate a confusion matrix heatmap from cached evaluation results.

    Parameters
    ----------
    cached_metrics : dict or None
        Pre-loaded cache dict. If ``None``, returns ``None``.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if cached_metrics is None:
        return None

    cm = np.array(cached_metrics["confusion_matrix"])

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=['Real', 'Fake'],
        y=['Real', 'Fake'],
        color_continuous_scale='RdYlGn',
        text_auto=True
    )

    # Enable explicit beautiful hover details
    fig.update_traces(
        hovertemplate="<b>True Label:</b> %{y}<br><b>Predicted Label:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>"
    )

    fig.update_layout(
        title={
            'text': "Confusion Matrix",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 15, 'color': TEXT_COLOR, 'family': 'sans-serif'}
        },
        xaxis=dict(side='bottom', color=TEXT_COLOR, gridcolor='#475569'),
        yaxis=dict(autorange='reversed', color=TEXT_COLOR, gridcolor='#475569'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_COLOR),
        margin=dict(l=60, r=40, t=80, b=60),
        height=380,
        autosize=True,
        dragmode='zoom'
    )

    return fig


def get_confusion_matrix_caption() -> str:
    """Returns explanatory caption for confusion matrix."""
    return (
        "Shows model predictions vs actual labels. Diagonal values (green) indicate correct predictions. "
        "Higher diagonal values demonstrate better model performance."
    )


# ==================== ROC CURVE ====================
def get_roc_curve_plot(cached_metrics: Optional[dict] = None) -> Optional[go.Figure]:
    """Generate an ROC curve from cached evaluation results.

    Parameters
    ----------
    cached_metrics : dict or None
        Pre-loaded cache dict. If ``None``, returns ``None``.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if cached_metrics is None:
        return None

    roc_data = cached_metrics.get("roc", {})
    fpr = np.array(roc_data.get("fpr", []))
    tpr = np.array(roc_data.get("tpr", []))

    if len(fpr) == 0 or len(tpr) == 0:
        logger.warning("ROC data is empty in metrics cache")
        return None

    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # Plot ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#22c55e', width=3),
        hovertemplate="<b>False Positive Rate (FPR):</b> %{x:.3f}<br><b>True Positive Rate (TPR):</b> %{y:.3f}<extra></extra>"
    ))

    # Plot Random line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='#94a3b8', width=2, dash='dash'),
        hovertemplate="<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title={
            'text': "ROC Curve",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 15, 'color': TEXT_COLOR, 'family': 'sans-serif'}
        },
        xaxis=dict(title='False Positive Rate', gridcolor='#475569', color=TEXT_COLOR),
        yaxis=dict(title='True Positive Rate', gridcolor='#475569', color=TEXT_COLOR),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        height=380,
        autosize=True,
        dragmode='zoom',  # Enables drag-to-zoom curves
        hovermode='closest'  # Ensures highly interactive hovering over the closest point
    )

    return fig


def get_roc_curve_caption() -> str:
    """Returns explanatory caption for ROC curve."""
    return (
        "Illustrates the model's ability to distinguish between Real and Fake images across different thresholds. "
        "AUC (Area Under Curve) closer to 1.0 indicates excellent discrimination. Line represents random guessing."
    )


# ==================== DATASET DISTRIBUTION ====================
def get_dataset_distribution_plot(cached_metrics: Optional[dict] = None) -> Optional[go.Figure]:
    """Generate a class distribution donut chart from cached evaluation results.

    Parameters
    ----------
    cached_metrics : dict or None
        Pre-loaded cache dict. If ``None``, returns ``None``.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if cached_metrics is None:
        return None

    dist = cached_metrics.get("dataset_distribution", {})
    n_real = dist.get("Real", 0)
    n_fake = dist.get("Fake", 0)

    labels = ['Real Images', 'Fake Images']
    sizes = [n_real, n_fake]
    colors = ['#22c55e', '#ef4444']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textinfo='percent+label',
        insidetextorientation='radial',
        hovertemplate="<b>%{label}</b><br><b>Samples:</b> %{value:,}<br><b>Percentage:</b> %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title={
            'text': "Test Set Class Distribution",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 15, 'color': TEXT_COLOR, 'family': 'sans-serif'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=380,
        autosize=True
    )

    return fig


def get_dataset_distribution_caption() -> str:
    """Returns explanatory caption for dataset distribution."""
    return (
        "Displays the balance between Real and Fake images in the test dataset. "
        "Equal distribution (50/50) helps produce unbiased evaluation metrics."
    )


# ==================== CLASS STATISTICS ====================
def get_class_statistics(cached_metrics: Optional[dict] = None) -> Optional[dict]:
    """Return per-class statistics from the evaluation cache.

    Parameters
    ----------
    cached_metrics : dict or None
        Pre-loaded cache dict. If ``None``, returns ``None``.

    Returns
    -------
    dict or None
        Nested dict with ``Real`` and ``Fake`` class-level stats.
    """
    if cached_metrics is None:
        return None

    return cached_metrics.get("class_statistics", {})


# ==================== METADATA HELPERS ====================
def get_evaluated_at(cached_metrics: Optional[dict] = None) -> Optional[str]:
    """Return the ISO timestamp of when evaluation was run.

    Returns
    -------
    str or None
    """
    if cached_metrics is None:
        return None
    return cached_metrics.get("evaluated_at")


def get_total_images(cached_metrics: Optional[dict] = None) -> Optional[int]:
    """Return the total number of images that were evaluated.

    Returns
    -------
    int or None
    """
    if cached_metrics is None:
        return None
    return cached_metrics.get("total_images")
