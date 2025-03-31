"""
Embedding provider using Microsoft's E5 models from HuggingFace.
"""

import numpy as np
from typing import List

from .base_provider import BaseEmbeddingProvider

class E5EmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using Microsoft's E5 models from HuggingFace.
    E5 models are specifically designed for retrieval and have shown excellent performance.
    """
    
    def __init__(self, model_name: str = "intfloat/e5-small-v2"):
        """
        Initialize the E5 embedding provider.
        
        Args:
            model_name: Name of the E5 model to use. Options include:
                - "intfloat/e5-small-v2" (smaller, faster)
                - "intfloat/e5-base-v2" (balanced)
                - "intfloat/e5-large-v2" (larger, more accurate)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            # Store torch for later use
            self.torch = torch
            
            print(f"Successfully loaded E5 model: {model_name} on {self.device}")
        except ImportError:
            print("Transformers not installed. Install with: pip install transformers torch")
            print("Falling back to simple embedding method.")
        except Exception as e:
            print(f"Error loading E5 model: {e}")
            print("Falling back to simple embedding method.")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using E5 models.
        E5 models expect a specific prefix for different types of text.
        For retrieval, we use "passage: " for document chunks and "query: " for queries.
        """
        if self.model is None or self.tokenizer is None:
            return self._simple_embed_texts(texts)
        
        # Add the "passage:" prefix to all texts
        # Note: When embedding queries, you should use "query: " instead
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        # Process in batches to avoid memory issues
        batch_size = 8
        all_embeddings = []
        
        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i+batch_size]
            try:
                # Tokenize the batch
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                
                # Generate embeddings
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # E5 models use mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = self.torch.sum(token_embeddings * input_mask_expanded, 1) / self.torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Convert to numpy and normalize
                batch_embeddings = embeddings.cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
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
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query using the 'query:' prefix.
        
        Args:
            query: The query text to embed
            
        Returns:
            A numpy array containing the query embedding
        """
        # For queries, use the "query: " prefix
        prefixed_query = f"query: {query}"
        
        # Use the same embedding method, but with the query prefix
        return self.embed_texts([prefixed_query])[0]