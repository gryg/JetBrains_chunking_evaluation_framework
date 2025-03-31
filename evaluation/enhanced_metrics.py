"""
Enhanced evaluation metrics including MTEB-style retrieval metrics.
This module extends the basic metrics with comprehensive measures 
used in the Massive Text Embedding Benchmark (MTEB).
"""

import re
import numpy as np
from typing import List, Dict, Any, Set, Union, Collection
from collections import defaultdict

class EnhancedEvaluationMetrics:
    """
    Evaluation metrics for retrieval systems including MTEB-style metrics.
    Extends the basic token-level metrics with additional retrieval metrics.
    """
    
    @staticmethod
    def calculate_metrics(retrieved_chunks: List[str], 
                         relevant_excerpts: List[Dict[str, Any]],
                         include_mteb: bool = True) -> Dict[str, float]:
        """
        Calculate precision, recall, IoU, and MTEB-style metrics.
        
        Args:
            retrieved_chunks: The chunks retrieved by the system
            relevant_excerpts: The ground truth relevant excerpts
            include_mteb: Whether to include MTEB-style metrics
            
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
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Basic metrics dictionary
        metrics = {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1,
            'retrieved_tokens_count': len(retrieved_tokens),
            'relevant_tokens_count': len(relevant_tokens),
            'intersection_count': len(intersection)
        }
        
        # Add MTEB-style metrics if requested
        if include_mteb:
            mteb_metrics = EnhancedEvaluationMetrics._calculate_mteb_metrics(
                retrieved_chunks, relevant_excerpts
            )
            metrics.update(mteb_metrics)
            
        return metrics
    
    @staticmethod
    def _calculate_mteb_metrics(retrieved_chunks: List[str], 
                               relevant_excerpts: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate MTEB-style retrieval metrics.
        
        Args:
            retrieved_chunks: The chunks retrieved by the system
            relevant_excerpts: The ground truth relevant excerpts
            
        Returns:
            A dictionary of MTEB-style metric values
        """
        # For MTEB-style metrics, we need to calculate:
        # 1. Precision@k (P@k)
        # 2. Normalized Discounted Cumulative Gain (nDCG)
        # 3. Mean Average Precision (MAP)
        # 4. Recall@k
        
        # Extract the content of relevant excerpts
        relevant_contents = []
        for excerpt in relevant_excerpts:
            if isinstance(excerpt, dict) and 'content' in excerpt:
                relevant_contents.append(excerpt['content'].lower())
            elif isinstance(excerpt, str):
                relevant_contents.append(excerpt.lower())
        
        # Check which chunks contain which relevant excerpts
        chunk_relevance = []
        for chunk in retrieved_chunks:
            chunk_lower = chunk.lower()
            is_relevant = False
            relevance_score = 0.0
            
            # For each relevant excerpt, check if it's contained in the chunk
            for content in relevant_contents:
                # Calculate token overlap for a more nuanced relevance score
                chunk_tokens = set(re.findall(r'\b\w+\b', chunk_lower))
                content_tokens = set(re.findall(r'\b\w+\b', content))
                
                if content_tokens and chunk_tokens:
                    overlap = len(chunk_tokens.intersection(content_tokens)) / len(content_tokens)
                    
                    # If there's significant overlap, consider it relevant
                    if overlap > 0.5:
                        is_relevant = True
                        relevance_score = max(relevance_score, overlap)  # Take highest relevance score
            
            chunk_relevance.append({
                'is_relevant': is_relevant,
                'score': relevance_score
            })
        
        # Calculate Precision@k (P@k) for various k values
        k_values = [1, 3, 5, 10]
        precision_at_k = {}
        
        for k in k_values:
            if k <= len(chunk_relevance):
                # Only consider first k retrieved chunks
                relevant_count = sum(1 for i in range(k) if chunk_relevance[i]['is_relevant'])
                precision_at_k[f'precision@{k}'] = relevant_count / k
            else:
                precision_at_k[f'precision@{k}'] = None
        
        # Calculate Normalized Discounted Cumulative Gain (nDCG)
        def dcg_at_k(relevance_scores, k):
            """Calculate DCG@k."""
            dcg = 0
            for i in range(min(len(relevance_scores), k)):
                dcg += relevance_scores[i] / np.log2(i + 2)  # i+2 because i is 0-indexed
            return dcg
        
        def idcg_at_k(relevance_scores, k):
            """Calculate Ideal DCG@k."""
            sorted_scores = sorted(relevance_scores, reverse=True)
            return dcg_at_k(sorted_scores, k)
        
        relevance_scores = [item['score'] for item in chunk_relevance]
        ndcg = {}
        
        for k in k_values:
            if k <= len(relevance_scores):
                dcg = dcg_at_k(relevance_scores, k)
                idcg = idcg_at_k(relevance_scores, k)
                ndcg[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0
            else:
                ndcg[f'ndcg@{k}'] = None
        
        # Calculate Mean Average Precision (MAP)
        def average_precision(relevant_chunks):
            """Calculate Average Precision."""
            if not any(chunk['is_relevant'] for chunk in relevant_chunks):
                return 0.0
                
            precision_sum = 0.0
            relevant_count = 0
            
            for i, chunk in enumerate(relevant_chunks):
                if chunk['is_relevant']:
                    relevant_count += 1
                    # Precision up to this point
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i
                    
            return precision_sum / relevant_count if relevant_count > 0 else 0.0
        
        map_score = average_precision(chunk_relevance)
        
        # Calculate Recall@k
        recall_at_k = {}
        total_relevant = sum(1 for item in chunk_relevance if item['is_relevant'])
        
        for k in k_values:
            if k <= len(chunk_relevance):
                # Count relevant chunks up to position k
                relevant_at_k = sum(1 for i in range(k) if chunk_relevance[i]['is_relevant'])
                recall_at_k[f'recall@{k}'] = relevant_at_k / len(relevant_contents) if relevant_contents else 0
            else:
                recall_at_k[f'recall@{k}'] = None
        
        # Combine all MTEB-style metrics
        mteb_metrics = {
            'map': map_score,
            **precision_at_k,
            **ndcg,
            **recall_at_k
        }
        
        return mteb_metrics


def compute_aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute aggregate metrics across all evaluation results.
    
    Args:
        all_results: List of evaluation result dictionaries
        
    Returns:
        Dictionary of aggregate metrics
    """
    # Initialize aggregation containers
    metrics_sum = defaultdict(float)
    metrics_count = defaultdict(int)
    
    # Collect metrics from all results
    for result in all_results:
        if 'metrics' in result:
            for metric_name, metric_value in result['metrics'].items():
                if isinstance(metric_value, (int, float)) and metric_value is not None:
                    metrics_sum[metric_name] += metric_value
                    metrics_count[metric_name] += 1
    
    # Calculate averages
    aggregate_metrics = {}
    for metric_name, metric_sum in metrics_sum.items():
        count = metrics_count[metric_name]
        if count > 0:
            aggregate_metrics[f'avg_{metric_name}'] = metric_sum / count
    
    return aggregate_metrics