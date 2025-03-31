"""
Standard evaluation metrics for chunking strategies.
"""

import re
from typing import List, Dict, Any, Set

class EvaluationMetrics:
    """Class for calculating evaluation metrics for retrieval systems."""
    
    @staticmethod
    def calculate_metrics(retrieved_chunks: List[str], 
                         relevant_excerpts: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate precision, recall, and IoU metrics.
        
        Args:
            retrieved_chunks: The chunks retrieved by the system
            relevant_excerpts: The ground truth relevant excerpts
            
        Returns:
            A dictionary of metric values
        """
        # Extract tokens from retrieved chunks
        retrieved_tokens = set()
        for chunk in retrieved_chunks:
            tokens = re.findall(r'\b\w+\b', chunk.lower())
            retrieved_tokens.update(tokens)
        
        # Extract tokens from relevant excerpts
        relevant_tokens = set()
        for excerpt in relevant_excerpts:
            content = excerpt.get('content', excerpt)
            if isinstance(content, str):
                tokens = re.findall(r'\b\w+\b', content.lower())
                relevant_tokens.update(tokens)
        
        # Calculate intersection
        intersection = retrieved_tokens.intersection(relevant_tokens)
        
        # Calculate metrics
        precision = len(intersection) / len(retrieved_tokens) if retrieved_tokens else 0
        recall = len(intersection) / len(relevant_tokens) if relevant_tokens else 0
        
        # Calculate IoU (Intersection over Union)
        union = retrieved_tokens.union(relevant_tokens)
        iou = len(intersection) / len(union) if union else 0
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1,
            'retrieved_tokens_count': len(retrieved_tokens),
            'relevant_tokens_count': len(relevant_tokens),
            'intersection_count': len(intersection)
        }