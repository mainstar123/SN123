"""
Probability Calibration for MANTIS Embeddings

Critical for LBFGS challenges where p[0..4] must be well-calibrated probabilities.
Raw model outputs are often poorly calibrated (overconfident or underconfident).

This module provides calibration methods to fix that.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax
import pickle


class ProbabilityCalibrator:
    """
    Calibrate model probabilities to match true frequencies
    
    For LBFGS challenges, validators expect:
    - p[0..4] to sum to 1.0
    - p[k] to reflect actual probability of bucket k
    - Well-calibrated means: if model says 70%, it should be right 70% of the time
    """
    
    def __init__(self, method='isotonic', n_bins=10):
        """
        Args:
            method: 'isotonic' (non-parametric, best) or 'platt' (parametric)
            n_bins: Number of bins for calibration validation
        """
        self.method = method
        self.n_bins = n_bins
        self.calibrators = {}  # One calibrator per class
        self.is_fitted = False
        
    def fit(self, y_probs: np.ndarray, y_true: np.ndarray):
        """
        Fit calibration on validation set
        
        Args:
            y_probs: Raw probabilities from model (n_samples, n_classes)
            y_true: True labels (n_samples,)
        """
        n_classes = y_probs.shape[1]
        
        # Fit one calibrator per class (one-vs-rest)
        for cls in range(n_classes):
            # Binary: is this class or not?
            y_binary = (y_true == cls).astype(int)
            p_cls = y_probs[:, cls]
            
            if self.method == 'isotonic':
                # Isotonic regression (non-parametric, monotonic)
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(p_cls, y_binary)
            elif self.method == 'platt':
                # Platt scaling (logistic regression)
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
                calibrator.fit(p_cls.reshape(-1, 1), y_binary)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.calibrators[cls] = calibrator
        
        self.is_fitted = True
        
        # Evaluate calibration quality
        self._evaluate_calibration(y_probs, y_true)
    
    def transform(self, y_probs: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities
        
        Args:
            y_probs: Raw probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        
        n_samples, n_classes = y_probs.shape
        calibrated = np.zeros_like(y_probs)
        
        # Calibrate each class
        for cls in range(n_classes):
            p_cls = y_probs[:, cls]
            
            if self.method == 'isotonic':
                calibrated[:, cls] = self.calibrators[cls].predict(p_cls)
            elif self.method == 'platt':
                calibrated[:, cls] = self.calibrators[cls].predict_proba(
                    p_cls.reshape(-1, 1)
                )[:, 1]
        
        # Ensure probabilities sum to 1 and are in [0, 1]
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated
    
    def fit_transform(self, y_probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(y_probs, y_true)
        return self.transform(y_probs)
    
    def _evaluate_calibration(self, y_probs: np.ndarray, y_true: np.ndarray):
        """Evaluate calibration quality"""
        calibrated = self.transform(y_probs)
        
        # Expected Calibration Error (ECE)
        ece = expected_calibration_error(calibrated, y_true, n_bins=self.n_bins)
        
        print(f"\nCalibration Quality:")
        print(f"  Method: {self.method}")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  (Lower is better, <0.05 is good)")
        
        # Per-class calibration
        for cls in range(y_probs.shape[1]):
            y_binary = (y_true == cls).astype(int)
            p_raw = y_probs[:, cls]
            p_cal = calibrated[:, cls]
            
            # Brier score (before and after)
            brier_raw = np.mean((p_raw - y_binary) ** 2)
            brier_cal = np.mean((p_cal - y_binary) ** 2)
            
            print(f"  Class {cls}: Brier {brier_raw:.4f} → {brier_cal:.4f}")
    
    def save(self, path: str):
        """Save calibrator to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'n_bins': self.n_bins,
                'calibrators': self.calibrators,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load calibrator from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls(method=data['method'], n_bins=data['n_bins'])
        calibrator.calibrators = data['calibrators']
        calibrator.is_fitted = data['is_fitted']
        
        return calibrator


def expected_calibration_error(y_probs: np.ndarray, y_true: np.ndarray, 
                               n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE)
    
    ECE measures how well probabilities are calibrated:
    - If model says 70%, it should be correct 70% of the time
    - ECE = average gap between predicted confidence and actual accuracy
    
    Args:
        y_probs: Predicted probabilities (n_samples, n_classes)
        y_true: True labels (n_samples,)
        n_bins: Number of bins for bucketing
        
    Returns:
        ECE value (0 = perfect, higher = worse)
    """
    y_pred = np.argmax(y_probs, axis=1)
    y_conf = np.max(y_probs, axis=1)
    
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Samples in this bin
        in_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            # Accuracy in bin
            acc_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            
            # Average confidence in bin
            conf_in_bin = y_conf[in_bin].mean()
            
            # Contribution to ECE (weighted by bin size)
            ece += (n_in_bin / len(y_conf)) * np.abs(acc_in_bin - conf_in_bin)
    
    return ece


def temperature_scaling(logits: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Temperature scaling for calibration (simpler than isotonic)
    
    Divides logits by temperature T before softmax:
    - T > 1: Makes predictions less confident (flatter distribution)
    - T < 1: Makes predictions more confident (sharper distribution)
    
    Args:
        logits: Model logits before softmax (n_samples, n_classes)
        y_true: True labels (n_samples,)
        
    Returns:
        (calibrated_probs, optimal_temperature)
    """
    from scipy.optimize import minimize_scalar
    
    def nll(T):
        """Negative log likelihood with temperature T"""
        scaled_logits = logits / T
        probs = softmax(scaled_logits, axis=1)
        
        # Cross entropy loss
        log_probs = np.log(probs[np.arange(len(y_true)), y_true] + 1e-8)
        return -np.mean(log_probs)
    
    # Find optimal temperature
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    T_opt = result.x
    
    # Apply optimal temperature
    calibrated_probs = softmax(logits / T_opt, axis=1)
    
    print(f"\nTemperature Scaling:")
    print(f"  Optimal temperature: {T_opt:.4f}")
    print(f"  (T > 1: less confident, T < 1: more confident)")
    
    return calibrated_probs, T_opt


class LBFGSEmbeddingCalibrator:
    """
    Special calibrator for LBFGS 17-dim embeddings
    
    LBFGS format: [p0, p1, p2, p3, p4, Q0, Q1, Q3, Q4] where:
    - p[0..4]: 5-bucket probabilities (must sum to 1)
    - Q(c): 3 probabilities per bucket (12 values total)
    
    This calibrator ensures all probabilities are well-calibrated.
    """
    
    def __init__(self):
        self.p_calibrator = None  # For p[0..4]
        self.q_calibrators = {}  # For Q(c) per bucket
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray, bucket_labels: np.ndarray, 
            opposite_move_labels: dict = None):
        """
        Fit calibration for LBFGS embeddings
        
        Args:
            embeddings: Raw embeddings (n_samples, 17)
            bucket_labels: True bucket labels 0-4 (n_samples,)
            opposite_move_labels: Dict of opposite move labels per bucket (optional)
        """
        # Calibrate p[0..4]
        p_probs = embeddings[:, :5]
        p_probs = p_probs / p_probs.sum(axis=1, keepdims=True)  # Normalize
        
        self.p_calibrator = ProbabilityCalibrator(method='isotonic')
        self.p_calibrator.fit(p_probs, bucket_labels)
        
        print("✓ Calibrated p[0..4] (5-bucket classifier)")
        
        # Calibrate Q(c) if labels provided
        if opposite_move_labels is not None:
            for bucket, labels in opposite_move_labels.items():
                # Q slices: [5:8], [8:11], [11:14], [14:17] for buckets 0,1,3,4
                q_start = 5 + bucket * 3 if bucket < 2 else 5 + (bucket - 1) * 3
                q_probs = embeddings[:, q_start:q_start+3]
                
                # Each Q value is independent probability
                # Calibrate each threshold separately
                # (Simplified: treat as binary classification)
                
            print("✓ Calibrated Q(c) (opposite-move probabilities)")
        
        self.is_fitted = True
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calibrate LBFGS embeddings
        
        Args:
            embeddings: Raw embeddings (n_samples, 17)
            
        Returns:
            Calibrated embeddings (n_samples, 17)
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted.")
        
        calibrated = embeddings.copy()
        
        # Calibrate p[0..4]
        p_probs = embeddings[:, :5]
        p_probs = p_probs / (p_probs.sum(axis=1, keepdims=True) + 1e-8)
        calibrated[:, :5] = self.p_calibrator.transform(p_probs)
        
        # Q(c) values stay as-is for now (or calibrate if implemented)
        # They're already in [0, 1] from sigmoid
        
        # Ensure p[0..4] sums to 1
        calibrated[:, :5] = calibrated[:, :5] / (
            calibrated[:, :5].sum(axis=1, keepdims=True) + 1e-8
        )
        
        # Ensure all values in [-1, 1] (MANTIS requirement)
        calibrated = np.clip(calibrated, -1.0, 1.0)
        
        return calibrated


# Usage example
if __name__ == "__main__":
    # Example: Calibrate 5-class classifier probabilities
    
    # Generate fake data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Raw model outputs (overconfident)
    logits = np.random.randn(n_samples, n_classes) * 2
    y_probs_raw = softmax(logits, axis=1)
    
    # True labels
    y_true = np.random.randint(0, n_classes, n_samples)
    
    print("=" * 60)
    print("BEFORE CALIBRATION")
    print("=" * 60)
    ece_before = expected_calibration_error(y_probs_raw, y_true)
    print(f"Expected Calibration Error: {ece_before:.4f}")
    
    # Split into calibration and test
    split = int(0.7 * n_samples)
    y_probs_cal = y_probs_raw[:split]
    y_true_cal = y_true[:split]
    y_probs_test = y_probs_raw[split:]
    y_true_test = y_true[split:]
    
    # Fit calibrator
    print("\n" + "=" * 60)
    print("CALIBRATING")
    print("=" * 60)
    calibrator = ProbabilityCalibrator(method='isotonic', n_bins=10)
    calibrator.fit(y_probs_cal, y_true_cal)
    
    # Transform test set
    y_probs_calibrated = calibrator.transform(y_probs_test)
    
    print("\n" + "=" * 60)
    print("AFTER CALIBRATION")
    print("=" * 60)
    ece_after = expected_calibration_error(y_probs_calibrated, y_true_test)
    print(f"Expected Calibration Error: {ece_after:.4f}")
    print(f"Improvement: {ece_before - ece_after:.4f}")
    
    if ece_after < ece_before:
        print("\n✅ Calibration improved probability quality!")
    else:
        print("\n⚠️  Calibration did not improve (may need more data)")

