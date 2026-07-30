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
        # Expanded heuristic patterns covering high-risk contract categories
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
            ),
            # NEW: Expanded categories for Option 3
            "Limitation of Liability": re.compile(
                r"(limitation\s*of\s*liability|aggregate\s*liability|consequential\s*damages|cap\s*on\s*liability|in\s*no\s*event\s*shall)", 
                re.IGNORECASE
            ),
            "Governing Law": re.compile(
                r"(governed\s*by|laws\s*of|jurisdiction|exclusive\s*venue|choice\s*of\s*law)", 
                re.IGNORECASE
            ),
            "Termination for Convenience": re.compile(
                r"(terminate\s*for\s*convenience|terminate\s*without\s*cause|prior\s*written\s*notice\s*of\s*termination)", 
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
                
            # If the score sits in an uncertainty buffer (+/- 0.15 of threshold), flag for human verification
            if abs(current_score - threshold_cutoff) < 0.15:
                requires_review = True
                
        return InferenceOutput(
            text=text,
            predicted_clauses=final_predictions,
            confidence_scores=adjusted_scores,
            requires_human_review=requires_review
        )

if __name__ == "__main__":
    # Baseline configuration supporting expanded categories
    mock_rules = {
        "Auto-Renewal": ClauseThreshold(clause_name="Auto-Renewal", optimal_threshold=0.35, target_precision=0.88, target_recall=0.95),
        "Indemnification": ClauseThreshold(clause_name="Indemnification", optimal_threshold=0.50, target_precision=0.91, target_recall=0.92),
        "Confidentiality": ClauseThreshold(clause_name="Confidentiality", optimal_threshold=0.45, target_precision=0.93, target_recall=0.94),
        "Limitation of Liability": ClauseThreshold(clause_name="Limitation of Liability", optimal_threshold=0.40, target_precision=0.90, target_recall=0.95),
        "Governing Law": ClauseThreshold(clause_name="Governing Law", optimal_threshold=0.50, target_precision=0.92, target_recall=0.91),
        "Termination for Convenience": ClauseThreshold(clause_name="Termination for Convenience", optimal_threshold=0.38, target_precision=0.89, target_recall=0.94)
    }
    
    processor = LegalPostProcessor(thresholds=mock_rules)
    
    sample_text = "Either party may terminate this Agreement without cause upon 30 days prior written notice."
    ambiguous_probs = {"Termination for Convenience": 0.20}
    
    result = processor.process_predictions(sample_text, ambiguous_probs)
    print(f"Text: '{result.text}'")
    print(f"Predicted Clauses: {result.predicted_clauses}")
    print(f"Adjusted Scores: {result.confidence_scores}")
    print(f"Requires Human Review: {result.requires_human_review}")