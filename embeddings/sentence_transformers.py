"""
Embedding provider using SentenceTransformers models from HuggingFace.
"""

import numpy as np
from typing import List

from .base_provider import BaseEmbeddingProvider

class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding provider using SentenceTransformers models from HuggingFace."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the SentenceTransformer embedding provider.
        
        Args:
            model_name: Name of the SentenceTransformers model to use
        """
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"Successfully loaded SentenceTransformer model: {model_name}")
        except ImportError:
            print("SentenceTransformers not installed. Install with: pip install sentence-transformers")
            print("Falling back to simple embedding method.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Falling back to simple embedding method.")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using SentenceTransformers."""
        if self.model is None:
            return self._simple_embed_texts(texts)
        
        # Process in batches to avoid memory issues with large text collections
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # Get embeddings from the model
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch: {e}")
                # Fall back to simple embeddings for this batch
                simple_embeddings = self._simple_embed_texts(batch)
                all_embeddings.append(simple_embeddings)
        
        # Combine all batches
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        else:
            return np.vstack(all_embeddings)