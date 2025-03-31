"""
Helper utilities for chunking evaluation.
"""

import json
import os
from typing import List, Dict, Any

def save_results(results: List[Dict[str, Any]], output_file: str = "chunking_results.json") -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def create_summary_table(results: List[Dict[str, Any]], metrics: List[str] = None) -> str:
    """
    Create a markdown table summarizing evaluation results.
    
    Args:
        results: List of result dictionaries
        metrics: List of metrics to include in the table
        
    Returns:
        Markdown table as string
    """
    if not results:
        return "No results to display."
    
    if metrics is None:
        metrics = ['avg_precision', 'avg_recall', 'avg_iou', 'avg_f1']
    
    # Create table header
    header = "| Chunker | Size | Overlap | Retrieved | " + " | ".join(m.replace('avg_', '') for m in metrics) + " |"
    separator = "|" + "---|" * (4 + len(metrics))
    
    # Create table rows
    rows = []
    for result in results:
        chunker_type = result.get('chunker_type', 'Unknown')
        chunk_size = result.get('chunk_size', '-')
        chunk_overlap = result.get('chunk_overlap', '-')
        num_retrieved = result.get('num_retrieved', '-')
        
        metric_values = []
        for metric in metrics:
            if metric in result:
                metric_values.append(f"{result[metric]:.4f}")
            else:
                metric_values.append('-')
        
        row = f"| {chunker_type} | {chunk_size} | {chunk_overlap} | {num_retrieved} | " + " | ".join(metric_values) + " |"
        rows.append(row)
    
    # Combine table parts
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    
    return table

def compare_embedding_providers(results_by_provider: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Create a comparison table across different embedding providers.
    
    Args:
        results_by_provider: Dictionary mapping provider names to their results
        
    Returns:
        Markdown table as string
    """
    if not results_by_provider:
        return "No results to compare."
    
    # Find common configuration across providers for comparison
    common_configs = {}
    
    for provider, results in results_by_provider.items():
        for result in results:
            config_key = f"{result['chunker_type']}_{result['chunk_size']}_{result['chunk_overlap']}"
            if config_key not in common_configs:
                common_configs[config_key] = {
                    'chunker_type': result['chunker_type'],
                    'chunk_size': result['chunk_size'],
                    'chunk_overlap': result['chunk_overlap'],
                    'providers': {}
                }
            
            common_configs[config_key]['providers'][provider] = {
                'avg_precision': result.get('avg_precision', 0),
                'avg_recall': result.get('avg_recall', 0),
                'avg_iou': result.get('avg_iou', 0),
                'avg_f1': result.get('avg_f1', 0)
            }
            
            # Add MTEB metrics if available
            if 'avg_ndcg@5' in result:
                common_configs[config_key]['providers'][provider]['avg_ndcg@5'] = result['avg_ndcg@5']
            if 'avg_map' in result:
                common_configs[config_key]['providers'][provider]['avg_map'] = result['avg_map']
    
    # Create table header
    providers = list(next(iter(results_by_provider.values()))[0].get('providers', {}).keys())
    metrics = ['avg_precision', 'avg_recall', 'avg_iou', 'avg_f1']
    
    header = "| Chunker | Size | Overlap | Metric | " + " | ".join(providers) + " |"
    separator = "|" + "---|" * (4 + len(providers))
    
    # Create table rows
    rows = []
    for config_key, config in common_configs.items():
        chunker_type = config['chunker_type']
        chunk_size = config['chunk_size']
        chunk_overlap = config['chunk_overlap']
        
        for metric in metrics:
            metric_values = []
            for provider in providers:
                if provider in config['providers'] and metric in config['providers'][provider]:
                    metric_values.append(f"{config['providers'][provider][metric]:.4f}")
                else:
                    metric_values.append('-')
            
            metric_name = metric.replace('avg_', '')
            row = f"| {chunker_type} | {chunk_size} | {chunk_overlap} | {metric_name} | " + " | ".join(metric_values) + " |"
            rows.append(row)
    
    # Combine table parts
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    
    return table