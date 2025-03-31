"""
Base class for embedding providers.
All embedding providers must inherit from this class.
"""

import numpy as np
import re
from typing import List

class BaseEmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: The texts to embed
            
        Returns:
            A numpy array of embeddings
        """
        raise NotImplementedError("Subclasses must implement embed_texts method")
    
    def _simple_embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Simple bag-of-words embedding as a fallback.
        
        Args:
            texts: The texts to embed
            
        Returns:
            A numpy array of embeddings
        """
        # Create vocabulary
        vocab = set()
        tokenized_texts = []
        
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokenized_texts.append(tokens)
            vocab.update(tokens)
            
        vocab_list = sorted(list(vocab))
        vocab_dict = {word: i for i, word in enumerate(vocab_list)}
        
        # Create embedding matrix
        embeddings = np.zeros((len(texts), len(vocab_dict)))
        
        for i, tokens in enumerate(tokenized_texts):
            for token in tokens:
                embeddings[i, vocab_dict[token]] += 1
                
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings