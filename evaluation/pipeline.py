"""
Evaluation pipeline for chunking strategies.
"""

from typing import List, Dict, Any, Optional
import sys

from chunkers.base_chunker import BaseChunker
from embeddings.base_provider import BaseEmbeddingProvider
from embeddings.e5_provider import E5EmbeddingProvider
from .data_loader import DataLoader
from .metrics import EvaluationMetrics
from .enhanced_metrics import EnhancedEvaluationMetrics, compute_aggregate_metrics
from .retrieval import RetrievalSystem, E5RetrievalSystem


class EvaluationPipeline:
    """Pipeline for evaluating chunking strategies."""
    
    def __init__(self, 
                 corpus_path: str,
                 questions_path: str,
                 corpus_id: str,
                 chunker: BaseChunker,
                 embedding_provider: BaseEmbeddingProvider,
                 num_retrieved: int = 5,
                 use_enhanced_metrics: bool = False):
        """
        Initialize the evaluation pipeline.
        
        Args:
            corpus_path: Path to the corpus file
            questions_path: Path to the questions CSV file
            corpus_id: Corpus ID to filter questions
            chunker: The chunking strategy to evaluate
            embedding_provider: The provider for generating embeddings
            num_retrieved: Number of chunks to retrieve for each query
            use_enhanced_metrics: Whether to use enhanced MTEB metrics
        """
        self.corpus_path = corpus_path
        self.questions_path = questions_path
        self.corpus_id = corpus_id
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.num_retrieved = num_retrieved
        self.use_enhanced_metrics = use_enhanced_metrics
        
    def run(self) -> Dict[str, Any]:
        """
        Run the evaluation pipeline.
        
        Returns:
            A dictionary of evaluation results
        """
        # Load data
        corpus = DataLoader.load_corpus(self.corpus_path)
        questions = DataLoader.load_questions(self.questions_path, self.corpus_id)
        
        if not corpus or not questions:
            print(f"Failed to load corpus or questions for {self.corpus_id}")
            return None
            
        print(f"Found {len(questions)} questions for corpus {self.corpus_id}")
        
        # Chunk the corpus
        chunks = self.chunker.split_text(corpus)
        print(f"Created {len(chunks)} chunks")
        
        # Set up retrieval system
        if isinstance(self.embedding_provider, E5EmbeddingProvider):
            retrieval_system = E5RetrievalSystem(self.embedding_provider)
        else:
            retrieval_system = RetrievalSystem(self.embedding_provider)
            
        retrieval_system.index_chunks(chunks)
        
        # Evaluate on each question
        results = []
        for i, question in enumerate(questions):
            # Retrieve chunks
            top_indices = retrieval_system.retrieve(question['question'], k=self.num_retrieved)
            retrieved_chunks = [chunks[idx] for idx in top_indices]
            
            # Calculate metrics
            if self.use_enhanced_metrics:
                metrics = EnhancedEvaluationMetrics.calculate_metrics(
                    retrieved_chunks, 
                    question['references'],
                    include_mteb=True
                )
            else:
                metrics = EvaluationMetrics.calculate_metrics(
                    retrieved_chunks, question['references']
                )
            
            results.append({
                'question': question['question'],
                'metrics': metrics
            })
            
            print(f"Question {i+1}/{len(questions)}: \"{question['question'][:50]}...\"")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  IoU: {metrics['iou']:.4f}")
            if 'f1' in metrics:
                print(f"  F1: {metrics['f1']:.4f}")
        
        # Calculate average metrics
        avg_precision = sum(r['metrics']['precision'] for r in results) / len(results)
        avg_recall = sum(r['metrics']['recall'] for r in results) / len(results)
        avg_iou = sum(r['metrics']['iou'] for r in results) / len(results)
        avg_f1 = sum(r['metrics'].get('f1', 0) for r in results) / len(results)
        
        print('\nAverage metrics:')
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  IoU: {avg_iou:.4f}")
        print(f"  F1: {avg_f1:.4f}")
        
        # Build additional metrics if enhanced metrics were used
        additional_metrics = {}
        if self.use_enhanced_metrics:
            mteb_keys = [k for k in results[0]['metrics'].keys() 
                        if k not in ['precision', 'recall', 'iou', 'f1', 
                                    'retrieved_tokens_count', 'relevant_tokens_count', 
                                    'intersection_count']]
            
            for key in mteb_keys:
                avg_value = sum(r['metrics'].get(key, 0) for r in results) / len(results)
                additional_metrics[f'avg_{key}'] = avg_value
                print(f"  {key}: {avg_value:.4f}")
        
        # Return complete results
        return {
            'corpus_id': self.corpus_id,
            'chunker_type': type(self.chunker).__name__,
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.chunk_overlap,
            'num_retrieved': self.num_retrieved,
            'num_chunks': len(chunks),
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_iou': avg_iou,
            'avg_f1': avg_f1,
            'num_questions': len(results),
            'individual_results': results,
            **additional_metrics
        }