"""
Main entry point for running chunking evaluations.
"""

import argparse
import sys
import json
import os
from typing import List, Dict, Any

from chunkers import (
    FixedTokenChunker,
    RecursiveCharacterTextSplitter,
    SentenceChunker
)
from embeddings import (
    SentenceTransformerProvider,
    HuggingFaceEmbeddingProvider,
    E5EmbeddingProvider
)
from evaluation import (
    EvaluationPipeline,
    EnhancedEvaluationPipeline,
    run_enhanced_evaluations
)
from utils import (
    save_results,
    create_summary_table,
    compare_embedding_providers
)


def create_configurations(args):
    """Create configurations based on command line arguments."""
    # Default configurations
    chunk_sizes = [200, 400, 600]
    overlaps = [0, 100, 200]
    num_retrieved = [5]
    chunker_types = ["FixedTokenChunker", "RecursiveCharacterTextSplitter", "SentenceChunker"]
    
    # Override with provided args if any
    if args.chunk_sizes:
        chunk_sizes = [int(size) for size in args.chunk_sizes.split(',')]
    if args.overlaps:
        overlaps = [int(overlap) for overlap in args.overlaps.split(',')]
    if args.num_retrieved:
        num_retrieved = [int(n) for n in args.num_retrieved.split(',')]
    if args.chunker_types:
        chunker_types = args.chunker_types.split(',')
    
    # Generate all combinations
    configurations = []
    
    for chunker_type in chunker_types:
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                for retrieved in num_retrieved:
                    # Skip if overlap is larger than chunk size
                    if overlap >= chunk_size:
                        continue
                    
                    configurations.append({
                        'chunker_type': chunker_type,
                        'chunk_size': chunk_size,
                        'chunk_overlap': overlap,
                        'num_retrieved': retrieved
                    })
    
    return configurations


def create_embedding_providers(args):
    """Create embedding providers based on command line arguments."""
    providers = []
    
    if args.providers:
        provider_names = args.provider_names.split(',') if args.provider_names else []
        model_names = args.provider_models.split(',') if args.provider_models else []
        
        for i, provider_type in enumerate(args.providers.split(',')):
            provider_config = {
                'type': provider_type,
                'name': provider_names[i] if i < len(provider_names) else provider_type,
                'model_name': model_names[i] if i < len(model_names) else None
            }
            providers.append(provider_config)
    else:
        # Default provider: SentenceTransformer with MiniLM
        providers.append({
            'type': 'SentenceTransformer',
            'name': 'Sentence Transformers (MiniLM)',
            'model_name': 'all-MiniLM-L6-v2'
        })
    
    return providers


def instantiate_embedding_provider(provider_config):
    """Instantiate an embedding provider from a configuration."""
    provider_type = provider_config['type']
    model_name = provider_config.get('model_name')
    
    if provider_type == 'SentenceTransformer':
        return SentenceTransformerProvider(model_name) if model_name else SentenceTransformerProvider()
    elif provider_type == 'HuggingFace':
        return HuggingFaceEmbeddingProvider(model_name) if model_name else HuggingFaceEmbeddingProvider()
    elif provider_type == 'E5':
        return E5EmbeddingProvider(model_name) if model_name else E5EmbeddingProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run chunking evaluation pipeline")
    
    # Required arguments
    parser.add_argument("--corpus-path", required=True, help="Path to the corpus file")
    parser.add_argument("--questions-path", required=True, help="Path to the questions CSV file")
    parser.add_argument("--corpus-id", required=True, help="Corpus ID to filter questions")
    
    # Chunking configuration options
    parser.add_argument("--chunk-sizes", help="Comma-separated list of chunk sizes to evaluate")
    parser.add_argument("--overlaps", help="Comma-separated list of chunk overlaps to evaluate")
    parser.add_argument("--num-retrieved", help="Comma-separated list of number of chunks to retrieve")
    parser.add_argument("--chunker-types", help="Comma-separated list of chunker types to evaluate")
    
    # Embedding provider options
    parser.add_argument("--providers", help="Comma-separated list of embedding provider types")
    parser.add_argument("--provider-names", help="Comma-separated list of embedding provider names")
    parser.add_argument("--provider-models", help="Comma-separated list of embedding provider model names")
    
    # Output options
    parser.add_argument("--output-prefix", default="chunking_results", help="Prefix for output files")
    parser.add_argument("--enhanced-metrics", action="store_true", help="Use enhanced MTEB metrics")
    
    return parser.parse_args()


def run_evaluations(args):
    """Run the evaluations based on command line arguments."""
    print("\n=== Running Chunking Evaluations ===\n")
    
    # Validate input files
    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_path}")
    if not os.path.exists(args.questions_path):
        raise FileNotFoundError(f"Questions file not found: {args.questions_path}")
    
    # Create configurations
    configurations = create_configurations(args)
    print(f"Created {len(configurations)} configurations to evaluate")
    
    # Create embedding providers
    provider_configs = create_embedding_providers(args)
    print(f"Using {len(provider_configs)} embedding providers")
    
    # Run evaluations for each provider
    results_by_provider = {}
    
    for provider_config in provider_configs:
        provider_name = provider_config['name']
        print(f"\n\n=== Evaluating with {provider_name} ===\n")
        
        # Instantiate the embedding provider
        embedding_provider = instantiate_embedding_provider(provider_config)
        
        # Run enhanced evaluations if requested
        if args.enhanced_metrics:
            print("Using enhanced MTEB metrics")
            results = run_enhanced_evaluations(
                corpus_path=args.corpus_path,
                questions_path=args.questions_path,
                corpus_id=args.corpus_id,
                configurations=configurations,
                embedding_provider=embedding_provider
            )
        else:
            # Use standard evaluation pipeline
            results = []
            for config in configurations:
                # Create chunker
                chunker_type = config['chunker_type']
                
                if chunker_type == 'FixedTokenChunker':
                    chunker = FixedTokenChunker(config['chunk_size'], config['chunk_overlap'])
                elif chunker_type == 'RecursiveCharacterTextSplitter':
                    chunker = RecursiveCharacterTextSplitter(config['chunk_size'], config['chunk_overlap'])
                elif chunker_type == 'SentenceChunker':
                    chunker = SentenceChunker(config['chunk_size'], config['chunk_overlap'])
                else:
                    print(f"Unknown chunker type: {chunker_type}, skipping")
                    continue
                
                # Run evaluation
                pipeline = EvaluationPipeline(
                    corpus_path=args.corpus_path,
                    questions_path=args.questions_path,
                    corpus_id=args.corpus_id,
                    chunker=chunker,
                    embedding_provider=embedding_provider,
                    num_retrieved=config['num_retrieved']
                )
                
                result = pipeline.run()
                if result:
                    results.append(result)
            
            # Create summary table
            print("\nSummary of results:")
            print(create_summary_table(results))
        
        # Store results for this provider
        results_by_provider[provider_name] = results
        
        # Save individual provider results
        output_file = f"{args.output_prefix}_{provider_name.lower().replace(' ', '_')}.json"
        save_results(results, output_file)
    
            # If multiple providers, create comparison
    if len(results_by_provider) > 1:
        print("\n=== Comparison Across Embedding Providers ===\n")
        comparison_table = compare_embedding_providers(results_by_provider)
        print(comparison_table)
        
        # Save combined results
        combined_results = {
            'results_by_provider': results_by_provider,
            'configurations': configurations,
            'provider_configs': provider_configs
        }
        
        combined_file = f"{args.output_prefix}_combined_results.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\nCombined results saved to {combined_file}")
    
    return results_by_provider


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        run_evaluations(args)
        print("\nEvaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())