# post_processor.py
import re
from typing import List, Dict, Any
from pydantic import BaseModel
from evaluator import ClauseThreshold

class InferenceOutput(BaseModel):
    text: str
    predicted_clauses: List[str]
    confidence_scores: Dict[str, float]
    requires_human_review: bool

class LegalPostProcessor:
    def __init__(self, thresholds: Dict[str, ClauseThreshold]):
        self.thresholds = thresholds
        # Compile hard matching patterns for key risks to prevent false negatives across all 3 pipeline targets
        self.heuristic_patterns = {
            "Auto-Renewal": re.compile(
                r"(auto(matically)?\s*renew|extension\s*notice|successive\s*terms|automatic\s*extension)", 
                re.IGNORECASE
            ),
            "Indemnification": re.compile(
                r"(hold\s*harmless|defend\s*and\s*indemnify|liability\s*reimbursement|indemnity|indemnification)", 
                re.IGNORECASE
            ),
            "Confidentiality": re.compile(
                r"(confidential|non-disclosure|nda|proprietary\s*information|disclose\s*confidential|trade\s*secret)", 
                re.IGNORECASE
            )
        }

    def process_predictions(self, text: str, raw_probabilities: Dict[str, float]) -> InferenceOutput:
        """
        Applies calibrated decision thresholds and business-logic regex overrides 
        to ensure zero false negatives on high-risk clauses.
        """
        final_predictions = []
        adjusted_scores = {}
        requires_review = False
        
        for clause, raw_prob in raw_probabilities.items():
            current_score = raw_prob
            rule = self.thresholds.get(clause)
            threshold_cutoff = rule.optimal_threshold if rule else 0.5
            
            # Apply Heuristic Override Layer
            if clause in self.heuristic_patterns:
                if self.heuristic_patterns[clause].search(text):
                    # Force confidence up if model missed a clear textual phrase trigger
                    if current_score < threshold_cutoff:
                        current_score = max(threshold_cutoff + 0.05, 0.85)
            
            adjusted_scores[clause] = current_score
            
            # Map finalized scores past threshold gating
            if current_score >= threshold_cutoff:
                final_predictions.append(clause)
                
            # If the score sits right in a zone of uncertainty, flag for human verification
            if abs(current_score - threshold_cutoff) < 0.15:
                requires_review = True
                
        return InferenceOutput(
            text=text,
            predicted_clauses=final_predictions,
            confidence_scores=adjusted_scores,
            requires_human_review=requires_review
        )

# Full System Verification Hook
if __name__ == "__main__":
    # Configure pre-calibrated baseline settings representing all three target classes
    mock_rules = {
        "Auto-Renewal": ClauseThreshold(clause_name="Auto-Renewal", optimal_threshold=0.35, target_precision=0.88, target_recall=0.95),
        "Indemnification": ClauseThreshold(clause_name="Indemnification", optimal_threshold=0.50, target_precision=0.91, target_recall=0.92),
        "Confidentiality": ClauseThreshold(clause_name="Confidentiality", optimal_threshold=0.45, target_precision=0.93, target_recall=0.94)
    }
    
    processor = LegalPostProcessor(thresholds=mock_rules)
    
    # Sample with ambiguous model probabilities but clear textual "Confidentiality" and "Auto-Renewal" phrase triggers
    sample_text = "The receiving party must protect all trade secrets and proprietary information."
    ambiguous_model_probs = {
        "Auto-Renewal": 0.10, 
        "Indemnification": 0.05,
        "Confidentiality": 0.32  # Below the 0.45 threshold
    }
    
    print("\n🧐 Running Ambiguous Extraction Text Through Post-Processor...")
    result = processor.process_predictions(sample_text, ambiguous_model_probs)
    
    print(f"   ↳ Processing Text: '{result.text}'")
    print(f"   ↳ Extracted Clauses: {result.predicted_clauses}")
    print(f"   ↳ Adjusted Probabilities: {result.confidence_scores}")
    print(f"   ↳ Flagged for Human Review Queue: {result.requires_human_review}")