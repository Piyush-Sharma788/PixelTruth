"""
Model Analytics Module - Deepfake Detection Metrics & Visualizations
This module provides helper functions to display model performance metrics,
confusion matrices, ROC curves, and dataset distributions using Plotly.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# ==================== THEME CONFIGURATION ====================
DARK_BG = '#0f172a'
DARK_CARD = '#1e293b'
TEXT_COLOR = '#e5e7eb'
BORDER_COLOR = '#475569'


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
    (Real vs Fake deepfake detection) using Plotly.
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Sample confusion matrix: [True Negatives, False Positives]
    #                          [False Negatives, True Positives]
    cm = np.array([[475, 25],    # Real images: 475 correct, 25 misclassified
                   [23, 477]])   # Fake images: 23 misclassified, 477 correct
    
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


def get_confusion_matrix_caption():
    """Returns explanatory caption for confusion matrix."""
    return (
        "Shows model predictions vs actual labels. Diagonal values (green) indicate correct predictions. "
        "Higher diagonal values demonstrate better model performance."
    )


# ==================== ROC CURVE ====================
def get_roc_curve_plot():
    """
    Generates a ROC (Receiver Operating Characteristic) curve using Plotly.
    Shows the trade-off between True Positive Rate and False Positive Rate.
    
    Returns:
        plotly.graph_objects.Figure
    """
    #Set seed to keep it reproducible
    np.random.seed(42)
    y_true = np.array([0]*500 + [1]*500)  # 500 real, 500 fake
    
    # Predicted probabilities (higher = more confident it's fake)
    y_prob = np.concatenate([
        np.random.beta(2, 5, 500),      # Real images: mostly low scores
        np.random.beta(5, 2, 500)       # Fake images: mostly high scores
    ])
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
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


def get_roc_curve_caption():
    """Returns explanatory caption for ROC curve."""
    return (
        "Illustrates the model's ability to distinguish between Real and Fake images across different thresholds. "
        "AUC (Area Under Curve) closer to 1.0 indicates excellent discrimination. Line represents random guessing."
    )


# ==================== DATASET DISTRIBUTION ====================
def get_dataset_distribution_plot():
    """
    Generates a class distribution donut chart showing the breakdown of
    Real vs Fake images in the training/testing dataset using Plotly.
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Sample dataset distribution
    labels = ['Real Images', 'Fake Images']
    sizes = [5000, 5000]  # Balanced dataset
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
            'text': "Class Distribution",
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
