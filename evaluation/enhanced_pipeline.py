"""
Enhanced evaluation pipeline with MTEB-style metrics.
"""

from typing import List, Dict, Any, Optional
import sys

from chunkers.base_chunker import BaseChunker
from embeddings.base_provider import BaseEmbeddingProvider
from embeddings.e5_provider import E5EmbeddingProvider
from .data_loader import DataLoader
from .enhanced_metrics import EnhancedEvaluationMetrics, compute_aggregate_metrics
from .retrieval import RetrievalSystem, E5RetrievalSystem


class EnhancedEvaluationPipeline:
    """
    Enhanced evaluation pipeline with MTEB-style metrics.
    This pipeline extends the basic evaluation pipeline with additional metrics.
    """
    
    def __init__(self, 
                 corpus_path: str,
                 questions_path: str,
                 corpus_id: str,
                 chunker: BaseChunker,
                 embedding_provider: BaseEmbeddingProvider,
                 num_retrieved: int = 5):
        """
        Initialize the enhanced evaluation pipeline.
        
        Args:
            corpus_path: Path to the corpus file
            questions_path: Path to the questions CSV file
            corpus_id: Corpus ID to filter questions
            chunker: The chunking strategy to evaluate
            embedding_provider: The provider for generating embeddings
            num_retrieved: Number of chunks to retrieve for each query
        """
        self.corpus_path = corpus_path
        self.questions_path = questions_path
        self.corpus_id = corpus_id
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.num_retrieved = num_retrieved
        
    def run(self) -> Dict[str, Any]:
        """
        Run the enhanced evaluation pipeline.
        
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
            
            # Calculate metrics using enhanced metrics
            metrics = EnhancedEvaluationMetrics.calculate_metrics(
                retrieved_chunks, 
                question['references'],
                include_mteb=True
            )
            
            # Extract some metrics for display
            key_metrics = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'iou': metrics['iou'],
                'f1': metrics['f1']
            }
            
            # Add MTEB metrics for display if available
            if 'ndcg@5' in metrics:
                key_metrics['ndcg@5'] = metrics['ndcg@5']
            if 'map' in metrics:
                key_metrics['map'] = metrics['map']
            
            results.append({
                'question': question['question'],
                'metrics': metrics
            })
            
            print(f"Question {i+1}/{len(questions)}: \"{question['question'][:50]}...\"")
            for name, value in key_metrics.items():
                print(f"  {name}: {value:.4f}")
        
        # Calculate aggregate metrics
        aggregate_metrics = compute_aggregate_metrics(results)
        
        # Extract standard metrics for display
        avg_precision = aggregate_metrics.get('avg_precision', 0)
        avg_recall = aggregate_metrics.get('avg_recall', 0)
        avg_iou = aggregate_metrics.get('avg_iou', 0)
        avg_f1 = aggregate_metrics.get('avg_f1', 0)
        
        # Extract MTEB metrics for display
        avg_ndcg5 = aggregate_metrics.get('avg_ndcg@5', 0)
        avg_map = aggregate_metrics.get('avg_map', 0)
        
        print('\nAverage metrics:')
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  IoU: {avg_iou:.4f}")
        print(f"  F1: {avg_f1:.4f}")
        print(f"  nDCG@5: {avg_ndcg5:.4f}")
        print(f"  MAP: {avg_map:.4f}")
        
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
            'avg_ndcg@5': avg_ndcg5,
            'avg_map': avg_map,
            'num_questions': len(results),
            'all_metrics': aggregate_metrics,
            'individual_results': results
        }


def run_enhanced_evaluations(corpus_path: str, 
                            questions_path: str, 
                            corpus_id: str,
                            configurations: List[Dict[str, Any]],
                            embedding_provider: Optional[BaseEmbeddingProvider] = None) -> List[Dict[str, Any]]:
    """
    Run evaluations with MTEB metrics for multiple chunking configurations.
    
    Args:
        corpus_path: Path to the corpus file
        questions_path: Path to the questions CSV file
        corpus_id: Corpus ID to filter questions
        configurations: List of configuration dictionaries
        embedding_provider: Optional embedding provider to reuse
                        
    Returns:
        A list of evaluation result dictionaries
    """
    results = []
    
    # Create embedding provider (reused across evaluations) if not provided
    if embedding_provider is None:
        from ..embeddings.sentence_transformers import SentenceTransformerProvider
        embedding_provider = SentenceTransformerProvider()
    
    for config in configurations:
        # Create chunker
        chunker_type = config.get('chunker_type', 'FixedTokenChunker')
        
        # Check if the chunker class is imported
        chunker_class = None
        try:
            # Import dynamically based on chunker_type
            if chunker_type == 'FixedTokenChunker':
                from chunkers.fixed_token_chunker import FixedTokenChunker
                chunker_class = FixedTokenChunker
            elif chunker_type == 'RecursiveCharacterTextSplitter':
                from chunkers.recursive_chunker import RecursiveCharacterTextSplitter
                chunker_class = RecursiveCharacterTextSplitter
            elif chunker_type == 'SentenceChunker':
                from chunkers.sentence_chunker import SentenceChunker
                chunker_class = SentenceChunker
            else:
                print(f"Warning: Chunker class '{chunker_type}' not found.")
                continue
        except (ImportError, AttributeError):
            print(f"Warning: Chunker class '{chunker_type}' could not be imported.")
            continue
            
        chunker = chunker_class(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        # Run enhanced evaluation
        pipeline = EnhancedEvaluationPipeline(
            corpus_path=corpus_path,
            questions_path=questions_path,
            corpus_id=corpus_id,
            chunker=chunker,
            embedding_provider=embedding_provider,
            num_retrieved=config['num_retrieved']
        )
        
        result = pipeline.run()
        if result:
            results.append(result)
    
    # Create comparison table with additional MTEB metrics
    print('\n\nComparison of all configurations:')
    print('Chunker | Size | Overlap | Retrieved | Precision | Recall | IoU | F1 | nDCG@5 | MAP')
    print('------- | ---- | ------- | --------- | --------- | ------ | --- | -- | ------ | ---')
    
    for r in results:
        print(f"{r['chunker_type']} | {r['chunk_size']} | {r['chunk_overlap']} | {r['num_retrieved']} | "
              f"{r['avg_precision']:.4f} | {r['avg_recall']:.4f} | {r['avg_iou']:.4f} | "
              f"{r['avg_f1']:.4f} | {r['avg_ndcg@5']:.4f} | {r['avg_map']:.4f}")
    
    return results