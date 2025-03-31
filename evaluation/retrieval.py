"""
Retrieval systems for document chunks.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

from embeddings.base_provider import BaseEmbeddingProvider
from embeddings.e5_provider import E5EmbeddingProvider

class RetrievalSystem:
    """System for retrieving text chunks based on query embeddings."""
    
    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        """
        Initialize the retrieval system.
        
        Args:
            embedding_provider: The provider for generating embeddings
        """
        self.embedding_provider = embedding_provider
        self.chunks = None
        self.chunk_embeddings = None
        
    def index_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for the chunks and index them.
        
        Args:
            chunks: The text chunks to index
            
        Returns:
            The chunk embeddings
        """
        chunk_embeddings = self.embedding_provider.embed_texts(chunks)
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        return chunk_embeddings
    
    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """
        Retrieve the top k chunks most similar to the query.
        
        Args:
            query: The query text
            k: The number of chunks to retrieve
            
        Returns:
            A list of chunk indices
        """
        query_embedding = self.embedding_provider.embed_texts([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return top_indices.tolist()


class E5RetrievalSystem(RetrievalSystem):
    """Modified retrieval system for E5 embeddings that require special query handling."""
    
    def __init__(self, embedding_provider: E5EmbeddingProvider):
        """
        Initialize the E5 retrieval system with an E5 embedding provider.
        
        Args:
            embedding_provider: The E5 embedding provider
        """
        super().__init__(embedding_provider)
        
    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """
        Retrieve the top k chunks most similar to the query.
        Uses the E5-specific query embedding method.
        
        Args:
            query: The query text
            k: The number of chunks to retrieve
            
        Returns:
            A list of chunk indices
        """
        # Use the E5-specific query embedding method
        if hasattr(self.embedding_provider, 'embed_query'):
            query_embedding = np.array([self.embedding_provider.embed_query(query)])
        else:
            query_embedding = self.embedding_provider.embed_texts([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return top_indices.tolist()