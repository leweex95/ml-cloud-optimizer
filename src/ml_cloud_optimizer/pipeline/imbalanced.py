"""Handling imbalanced datasets with SMOTE, Focal Loss, and class weighting."""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SMOTEHandler:
    """Handle imbalanced data using SMOTE."""

    def __init__(self, sampling_strategy: float = 0.5, random_state: int = 42):
        """Initialize SMOTE handler.
        
        Args:
            sampling_strategy: Ratio of minority to majority class (0-1)
            random_state: Random seed
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=-1
        )

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Resampled (X, y)
        """
        original_counts = np.bincount(y)
        logger.info(f"Original distribution: {original_counts}")

        X_resampled, y_resampled = self.smote.fit_resample(X, y)

        resampled_counts = np.bincount(y_resampled)
        logger.info(f"Resampled distribution: {resampled_counts}")

        return X_resampled, y_resampled


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in range [0, 1] to balance positive vs negative
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        ce_loss = self.ce_loss(logits, targets.float())
        
        # Calculate focal loss
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss

        return loss.mean()


class ClassWeightCalculator:
    """Calculate class weights for imbalanced datasets."""

    @staticmethod
    def calculate_weights(y: np.ndarray, method: str = 'balanced') -> Dict[int, float]:
        """Calculate class weights.
        
        Args:
            y: Target vector
            method: Method for weight calculation ('balanced', 'inverse_frequency')
            
        Returns:
            Dictionary of class weights
        """
        if method == 'balanced':
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            return {int(c): float(w) for c, w in zip(classes, weights)}

        elif method == 'inverse_frequency':
            unique, counts = np.unique(y, return_counts=True)
            weights = 1.0 / counts
            weights = weights / weights.sum()
            return {int(c): float(w) for c, w in zip(unique, weights)}

        elif method == 'log':
            # Log scaling of inverse frequencies
            unique, counts = np.unique(y, return_counts=True)
            weights = 1.0 / np.log(counts + 1)
            weights = weights / weights.sum()
            return {int(c): float(w) for c, w in zip(unique, weights)}

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def get_sample_weights(y: np.ndarray, weights: Dict[int, float]) -> np.ndarray:
        """Get per-sample weights.
        
        Args:
            y: Target vector
            weights: Class weight dictionary
            
        Returns:
            Sample weights array
        """
        return np.array([weights[int(label)] for label in y])


class ImbalancedTrainer:
    """Train models on imbalanced datasets."""

    def __init__(self, model: Any, use_smote: bool = False, use_class_weights: bool = True):
        """Initialize imbalanced trainer.
        
        Args:
            model: Model to train
            use_smote: Whether to apply SMOTE
            use_class_weights: Whether to use class weights
        """
        self.model = model
        self.use_smote = use_smote
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.smote_handler = None

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Prepared (X, y)
        """
        if self.use_smote:
            self.smote_handler = SMOTEHandler(sampling_strategy=0.5)
            X, y = self.smote_handler.fit_resample(X, y)
            logger.info(f"Applied SMOTE: new shape {X.shape}")

        if self.use_class_weights:
            self.class_weights = ClassWeightCalculator.calculate_weights(y)
            logger.info(f"Calculated class weights: {self.class_weights}")

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train model on imbalanced data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Training results
        """
        # Prepare data
        X_prepared, y_prepared = self.prepare_data(X_train, y_train)

        # Train with class weights
        if self.use_class_weights and self.class_weights:
            sample_weights = ClassWeightCalculator.get_sample_weights(
                y_prepared, self.class_weights
            )
            self.model.fit(X_prepared, y_prepared, sample_weight=sample_weights)
        else:
            self.model.fit(X_prepared, y_prepared)

        logger.info("Model training completed")

        return {
            "model": self.model,
            "smote_applied": self.use_smote,
            "class_weights": self.class_weights,
        }


class ImbalancedEvaluator:
    """Evaluate performance on imbalanced datasets."""

    @staticmethod
    def evaluate_imbalanced(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
    ) -> Dict[str, float]:
        """Evaluate classification on imbalanced data.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
            balanced_accuracy_score,
        )

        metrics = {
            "accuracy": (y_true == y_pred).mean(),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics["roc_auc"] = None

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    # from sklearn.datasets import make_classification
    # from sklearn.ensemble import RandomForestClassifier
    #
    # # Generate imbalanced dataset
    # X, y = make_classification(
    #     n_samples=1000,
    #     n_features=20,
    #     n_informative=10,
    #     weights=[0.97, 0.03],
    #     random_state=42
    # )
    #
    # trainer = ImbalancedTrainer(
    #     RandomForestClassifier(),
    #     use_smote=True,
    #     use_class_weights=True
    # )
    # results = trainer.train(X, y)
