"""
Métricas de avaliação do classificador de alfabetização.

Centraliza o cálculo usado pelo treinamento (src/modeling/train.py) e pelos
notebooks, garantindo que treino, validação e relatórios usem exatamente a
mesma definição de cada métrica.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)


def metricas(y_true, y_pred, y_prob, sample_weight=None) -> dict:
    """Painel padrão: discriminação (AUCs), classificação no corte e calibração."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob, sample_weight=sample_weight),
        "pr_auc": average_precision_score(y_true, y_prob, sample_weight=sample_weight),
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "f1": f1_score(y_true, y_pred, sample_weight=sample_weight),
        "precision": precision_score(y_true, y_pred, sample_weight=sample_weight),
        "recall": recall_score(y_true, y_pred, sample_weight=sample_weight),
        "brier": brier_score_loss(y_true, y_prob, sample_weight=sample_weight),
    }
