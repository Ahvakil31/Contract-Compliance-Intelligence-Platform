# evaluator.py
import numpy as np
from sklearn.metrics import precision_recall_curve
from typing import Dict, List
from pydantic import BaseModel

class ClauseThreshold(BaseModel):
    clause_name: str
    optimal_threshold: float
    target_precision: float
    target_recall: float

class LegalModelEvaluator:
    def __init__(self, clause_labels: List[str]):
        self.clause_labels = clause_labels

    def calibrate_thresholds(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, ClauseThreshold]:
        """
        Finds the optimal probability threshold per clause by maximizing the F2-score 
        (which weights recall twice as heavily as precision).
        """
        # Ensure the evaluation arrays are not empty before proceeding
        if y_probs.size == 0 or y_true.size == 0:
            raise ValueError("Calibration failed: Input probability or ground-truth arrays are empty.")
            
        calibrated_map = {}
        
        for i, clause in enumerate(self.clause_labels):
            # Compute precisions, recalls, and thresholds per label column
            precisions, recalls, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
            
            # Avoid division by zero by adding a small epsilon
            # F2 Formula: $F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$
            f2_scores = (5 * precisions * recalls) / ((4 * precisions) + recalls + 1e-10)
            best_idx = np.argmax(f2_scores)
            
            # If best_idx is out of bounds for thresholds (length is len(thresholds) + 1),
            # fall back to the highest predicted probability for that class instead of a arbitrary 0.5 default
            thresh_val = thresholds[best_idx] if best_idx < len(thresholds) else float(np.max(y_probs[:, i]))
            
            calibrated_map[clause] = ClauseThreshold(
                clause_name=clause,
                optimal_threshold=float(thresh_val),
                target_precision=float(precisions[best_idx]),
                target_recall=float(recalls[best_idx])
            )
            
            # Print corrected variable placeholders
            print(f"📈 Calibrated [{clause}]: Threshold={thresh_val:.4f} | Target Precision={precisions[best_idx]:.4f} | Target Recall={recalls[best_idx]:.4f}")
            
        return calibrated_map

if __name__ == "__main__":
    # Added the 3rd class label to match your fine_tune script
    labels = ["Auto-Renewal", "Indemnification", "Confidentiality"]
    evaluator = LegalModelEvaluator(labels)
    
    # Mock ground truths and model raw probability scores configured across 3 label arrays
    mock_true = np.array([
        [1, 0, 0], 
        [0, 1, 0], 
        [1, 0, 0], 
        [0, 0, 1]
    ])
    mock_probs = np.array([
        [0.42, 0.12, 0.05], 
        [0.05, 0.88, 0.11], 
        [0.71, 0.02, 0.09], 
        [0.11, 0.15, 0.93]
    ])
    
    print("⏳ Analyzing model probability matrix distributions...")
    threshold_rules = evaluator.calibrate_thresholds(mock_true, mock_probs)